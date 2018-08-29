import traceback
import rospy
import numpy as np

from giskardpy.symengine_wrappers import *
from giskardpy.input_system import FrameInput, Vector3Input
from giskardpy.qp_problem_builder import SoftConstraint as SC
from gebsyas.actions import PActionInterface, Action
from gebsyas.basic_controllers import run_ineq_controller, InEqController
from gebsyas.data_structures import SymbolicData, StampedData
from gebsyas.bc_controller_wrapper import BCControllerWrapper
from gebsyas.bullet_based_controller import InEqBulletController
from gebsyas.dl_reasoning import DLRigidGMMObject, DLRigidObject, DLDisjunction
from gebsyas.predicates import ClearlyPerceived, PInstance
from gebsyas.utils   import symbol_formatter, real_quat_from_matrix

from blessed import Terminal


class ObservationController(InEqBulletController):

    def init(self, context, proximity_frame, camera, data_id='searched_objects'):
        self.context = context
        self.data_id = data_id
        self.proximity_frame = proximity_frame
        self.current_cov = eye(6)
        self.current_weight = 0.0

        for x in range(3):
            for y in 'xyz'[x:]:
                name = 'v_{}{}'.format('xyz'[x], y)
                symbol = Symbol(name)
                setattr(self, name, symbol)
                self.current_subs[symbol] = 0.0

        self.evecs = [Vector3Input.prefix(symbol_formatter, 'eigen1'),
                     Vector3Input.prefix(symbol_formatter, 'eigen2'),
                     Vector3Input.prefix(symbol_formatter, 'eigen3')]

        self.obstruction = Vector3Input.prefix(symbol_formatter, 'obstruction')

        for i in 'xyz':
            for evec in self.evecs:
                self.current_subs[getattr(evec, i)] = 0

        v_e1 = self.evecs[0].get_expression()
        v_e2 = self.evecs[1].get_expression()
        v_e3 = self.evecs[2].get_expression()

        opt_obs_range   = 1.2 # Figure this out by semantic type and observations. 
        opt_obs_falloff = 0.2

        self.goal_obj_index = -1
        self.goal_gmm_index = -1
        self.search_objects = None

        self.frame_input = FrameInput.prefix_constructor('position', 'orientation', symbol_formatter)

        pose = self.frame_input.get_frame()

        view_dir = x_of(camera.pose)
        c2o      = pos_of(pose) - pos_of(camera.pose)
        z_dist   = norm(c2o)
        look_dir = c2o / z_dist
        obs_ctrl = 1 #1 - (z_dist / opt_obs_range) # ((opt_obs_range - z_dist) / opt_obs_falloff) ** 2
        in_view  = acos(dot(view_dir, c2o) / norm(c2o))
        proximity = norm(diag(1,1,0,1) * (pos_of(pose) - pos_of(proximity_frame))) 

        co_lin_x = dot(view_dir, v_e1)
        co_lin_y = dot(view_dir, v_e2)
        co_lin_z = dot(view_dir, v_e3)

        # ((camera.hfov - acos(in_view)) / camera.hfov) ** 2
        s_in_view    = SC(- in_view, - in_view, norm(v_e1) + norm(v_e2) + norm(v_e3), in_view)
        s_in_v_dist  = SC(- proximity, opt_obs_range + opt_obs_falloff - proximity, 1, proximity)
        s_avoid_near = SC(camera.near - z_dist, 100, 1, z_dist)
        
        s_v_e1 = SC(-co_lin_x, -co_lin_x, norm(v_e1), co_lin_x)
        s_v_e2 = SC(-co_lin_y, -co_lin_y, norm(v_e2), co_lin_y)
        s_v_e3 = SC(-co_lin_z, -co_lin_z, norm(v_e3), co_lin_z)

        soft_constraints = {#'igain'   : s_igain,
                             'near_clip': s_avoid_near,
                             'examine_1': s_v_e1,
                             'examine_2': s_v_e2,
                             'examine_3': s_v_e3,
                             'in_view' : s_in_view,  
                             'obs_dist': s_in_v_dist}

        for link, (m, b, n) in context.agent.robot.collision_avoidance_links.items():
            link_pos = pos_of(context.agent.robot.get_fk_expression('map', link))
            c2l = link_pos - pos_of(camera.pose)
            ray_dist = norm(cross(view_dir, c2l))
            d_dist   = dot(view_dir, c2l)
            #soft_constraints['{}_non-obstruction'.format(link)] = SC(sin(camera.hfov * 0.5) * (d_dist / cos(camera.hfov* 0.5)) - ray_dist, 100, 1, ray_dist)

        context.agent.add_data_cb(self.data_id, self.update_objects)
        super(ObservationController, self).init(soft_constraints)

    def get_cmd(self, nWSR=None):
        if self.search_objects is not None:
            self.re_eval_focused_object()
        if self.goal_obj_index > -1:
            return super(ObservationController, self).get_cmd(nWSR)
        self.print_fn('Waiting for updated object information')
        return {}

    def update_objects(self, gmm_objects):
        self.search_objects = gmm_objects
        self.update_object_terms()

    def re_eval_focused_object(self):
        new_obj_index = -1
        new_gmm_index = -1
        best_rating = 100000.0
        robot_pos = pos_of(self.proximity_frame).subs(self.current_subs)

        for n in range(len(self.search_objects.search_object_list)):
            gmm_object = self.search_objects.search_object_list[n]
            object_weight = self.search_objects.weights[n]
            for x in range(len(gmm_object.gmm)):
                if gmm_object.gmm[x].weight > 0.0:
                    rating = (norm(diag(1,1,0,1) * (robot_pos - pos_of(gmm_object.gmm[x].pose))) / gmm_object.gmm[x].weight) / object_weight
                    if rating < best_rating:
                        new_obj_index = n
                        new_gmm_index = x
                        best_rating = rating

        if new_obj_index == -1:
            return
        elif new_obj_index != self.goal_obj_index or new_gmm_index != self.goal_gmm_index:
            self.goal_obj_index = new_obj_index
            self.goal_gmm_index = new_gmm_index
            self.update_object_terms()

    def update_object_terms(self):
        if self.goal_obj_index == -1:
            return

        gmm_object = self.search_objects.search_object_list[self.goal_obj_index]
        pose = gmm_object.gmm[self.goal_gmm_index].pose
        #print('New best gmm: {}\nAt location:\n{}\nWeight: {}'.format(self.goal_gmm_index, str(pos_of(pose)), gmm_object.gmm[self.goal_gmm_index].weight))
        self.current_cov    = np.array(gmm_object.gmm[self.goal_gmm_index].cov.tolist(), dtype=float)
        self.current_weight = gmm_object.gmm[self.goal_gmm_index].weight
        self.current_subs[self.frame_input.x] = pose[0, 3]
        self.current_subs[self.frame_input.y] = pose[1, 3]
        self.current_subs[self.frame_input.z] = pose[2, 3]
        quat = real_quat_from_matrix(pose)
        self.current_subs[self.frame_input.qx] = quat[0]
        self.current_subs[self.frame_input.qy] = quat[1]
        self.current_subs[self.frame_input.qz] = quat[2]
        self.current_subs[self.frame_input.qw] = quat[3]
        w, v = np.linalg.eig(self.current_cov[:3, :3])
        pos_eig = w * v
        for x in range(len(self.evecs)):
            self.current_subs[self.evecs[x].x] = pos_eig[0, x] # if np.isreal(pos_eig[0, x]) else 0
            self.current_subs[self.evecs[x].y] = pos_eig[1, x] # if np.isreal(pos_eig[1, x]) else 0
            self.current_subs[self.evecs[x].z] = pos_eig[2, x] # if np.isreal(pos_eig[2, x]) else 0

    def get_current_object(self):
        if self.goal_obj_index != -1:
            return self.search_objects.search_object_list[self.goal_obj_index]
        return None

    def stop(self):
        self.context.agent.remove_data_cb(self.data_id, self.update_objects)
        super(ObservationController, self).stop()


class ActivePerceptionInterface(PActionInterface):
    def __init__(self):
        super(ActivePerceptionInterface, self).__init__(
            'Move{}'.format(ClearlyPerceived.P),
            [],
            [PInstance(ClearlyPerceived, ('a',), True)],
            2.0)

    """
    @brief      Symbolic motion action interface which instantiates a GenericMotionAction.
    """
    def instantiate_action(self, context, assignments):
        ineq_constraints = {}
        pred_state = context.agent.get_predicate_state()
        d_assignments =  {a: pred_state.map_to_data(s).data for a, s in assignments.items()}
        for k in d_assignments.keys():
            v = d_assignments[k]
            if type(v) == SymbolicData:
                d_assignments[k] = v.data

        for p, args in self.postcons.items():
            for at, value in args.items():
                fargs = [d_assignments[a] for a in at]
                ineq_constraints.update(p.fp(context, *fargs))

        return ActivePerceptionAction(assignments['a'])


class ActivePerceptionAction(Action):
    """
    @brief      This action converts a set of inequality constraints directly into a controller and runs it.
    """
    def __init__(self, object_id):
        super(ActivePerceptionAction, self).__init__('ActivePerception')
        self.object_id = object_id
        self.terminal = Terminal()

    def execute(self, context):
        try:
            motion_ctrl = ObservationController(context,
                                                context.agent.get_data_state().dl_data_iterator(DLDisjunction(DLRigidObject, DLRigidGMMObject)),
                                                set(),
                                                3,
                                                context.log) #context.log
            # motion_ctrl = ObservationController(context.agent.robot,
            #                                     self.clear_and_print) #context.log
            motion_ctrl.init(context, 
                             context.agent.robot.get_fk_expression('map', 'base_link') * translation3(0.1, 0, 0),
                             context.agent.robot.camera, 
                             self.object_id)

            motion_success, m_lf, t_log = run_observation_controller(context.agent.robot, motion_ctrl, context.agent, 0.02, 0.9)

            context.display.draw_robot_trajectory('motion_action', context.agent.robot, t_log)

            if motion_success:
                return 1.0
            else:
                context.log('Whoops, perception failed. I don\'t know what to do now, so I\'ll just abort...')
        except Exception as e:
            context.log(traceback.format_exc())

        return 0.0

    def clear_and_print(self, msg):
        print('{}{}'.format(self.terminal.clear(), msg))


ACTIONS = [ActivePerceptionInterface()]


class ObservationRunner(object):
    """This class runs an observation controller. It processes joint state updates and new commands.
       It also terminates controller execution when the searched object is clearly perceived. 
       It is assumed, that the object can always be found.
    """
    def __init__(self, robot, controller, f_send_command,
                 f_add_cb, variance=0.02, weight=0.9):
        """
        Constructor.
        Needs a robot,
        the controller to run,
        a total timeout,
        a timeout for the low activity commands,
        a function to send commands,
        a function to add itself as listener for joint states,
        the threshold for low activity,
        the names of the constraints to monitor for satisfaction
        """
        self.robot = robot
        if not isinstance(controller, ObservationController):
            raise Exception('Controller must be subclassed from ObservationController. Controller type: {}'.format(str(type(controller))))

        self.controller = controller
        self.f_send_command = f_send_command
        self.f_add_cb = f_add_cb
        self.last_feedback = 0
        self.last_update   = None
        self.t_variance = variance
        self.t_weight   = weight
        self.trajectory_log = []
        self.execution_start = None

    def run(self):
        """Starts the run of the controller."""
        now = rospy.Time.now()
        self.terminate = False
        self.execution_start = now

        self.f_add_cb(self.js_callback)

        while not rospy.is_shutdown() and not self.terminate:
            pass

        print('Runner terminating...')
        #self.controller.stop()
        return self.terminate, self.controller.current_weight

    def js_callback(self, joint_state):
        """Callback processing joint state updates, checking constraints and generating new commands."""
        if self.terminate: # Just in case
            return

        self.trajectory_log.append(StampedData(rospy.Time.from_sec((rospy.Time.now() - self.execution_start).to_sec()), joint_state.copy()))

        now = rospy.Time.now()
        self.controller.set_robot_js(joint_state)
        try:
            command = self.controller.get_cmd()
        except Exception as e:
            self.terminate = True
            traceback.print_exc()
            print(e)
            return
        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()
        self.f_send_command(command)
        if self.controller.goal_obj_index > -1:
            cov = self.controller.current_cov
            self.terminate = self.controller.current_weight >= self.t_weight and \
                             abs(cov[0,0]) <= self.t_variance and \
                             abs(cov[1,1]) <= self.t_variance and \
                             abs(cov[2,2]) <= self.t_variance

def run_observation_controller(robot, controller, agent, variance=0.02, weight=0.9):
    """Comfort function for easily instantiating and running an inequality runner."""
    runner = ObservationRunner(robot, controller, agent.act, agent.add_js_callback, variance, weight)
    constraints_met, lf = runner.run()
    return constraints_met, lf, runner.trajectory_log
