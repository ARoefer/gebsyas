import traceback
import rospy
import numpy as np
import pybullet as pb
import random

from multiprocessing import Lock

from iai_bullet_sim.utils import Frame

from gebsyas.actions import PActionInterface, Action
from gebsyas.basic_controllers import run_ineq_controller, InEqController
from gebsyas.bc_controller_wrapper import BCControllerWrapper
from gebsyas.bullet_based_controller import InEqBulletController
from gebsyas.constants import LBA_BOUND, UBA_BOUND
from gebsyas.data_structures import SymbolicData, StampedData, JointState
from gebsyas.dl_reasoning import DLRigidGMMObject, DLRigidObject, DLDisjunction
from gebsyas.headless_controller_runner import CollisionResolverRunner, LocalizationIntegrator
from gebsyas.predicates import ClearlyPerceived, PInstance
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.utils   import symbol_formatter, real_quat_from_matrix
from giskardpy.input_system import FrameInput, Vector3Input
from giskardpy.qp_problem_builder import SoftConstraint as SC
from giskardpy.symengine_wrappers import *
from geometry_msgs.msg import PoseStamped

from navigation_msgs.msg import NavToPoseActionGoal as ATPGoalMsg
from actionlib_msgs.msg import GoalID as GoalIDMsg

from std_msgs.msg import Float64 as Float64Msg

from blessed import Terminal

obs_dist_constraint = 'obs_dist'

# Switches the usage of vulcan. 
#   True:  Vulcan is only used as a fall when giskard gets stuck
#   False: Vulcan is used for all global navigation
VULCAN_FALLBACK = True
opt_obs_falloff = 0.2

class ObservationController(InEqBulletController):

    def init(self, context, proximity_frame, camera, data_id='searched_objects'):
        self.lock = Lock()
        self.context = context
        self.data_id = data_id
        self.proximity_frame = proximity_frame
        self.current_cov = eye(6)
        self.current_weight = 0.0
        self.pub_set_nav_goal = rospy.Publisher('/nav_to_pose/goal', ATPGoalMsg, queue_size=1, tcp_nodelay=True)
        self.pub_cancel       = rospy.Publisher('/nav_to_pose/cancel', GoalIDMsg, queue_size=1, tcp_nodelay=True)
        self.pub_obs_uba = rospy.Publisher('observation_controller/uba', Float64Msg, queue_size=1, tcp_nodelay=True)
        self.pub_jitter = rospy.Publisher('observation_controller/base_jitter', Float64Msg, queue_size=1, tcp_nodelay=True)
        self.global_nav_mode  = False

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

        opt_obs_range   = 1.0 # Figure this out by semantic type and observations. 

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
                             obs_dist_constraint : s_in_v_dist}


        self.global_base_controller = InEqController(context.agent.robot, 
                                                           self.print_fn,
                                                           True)
        base_constraints = {'in_view' : s_in_view,  
                            obs_dist_constraint : s_in_v_dist}
        self.bc_free_symbols = set()
        for sc in base_constraints.values():
            for f in sc:
                if hasattr(f, 'free_symbols'):
                    self.bc_free_symbols = self.bc_free_symbols.union(f.free_symbols)
        print(self.bc_free_symbols)
        self.bc_cpq = []
        for cpq in self.closest_point_queries:
            cac = cpq.generate_constraints()
            for ac in cac.values():
                if len(self.bc_free_symbols.intersection(ac.expression.free_symbols)) > 0:
                    base_constraints.update(cac)
                    self.bc_cpq.append(cpq)
                    break

        self.essential_base_constraints = set(base_constraints.keys())
        self.essential_base_constraints.remove('in_view')
        self.global_base_controller.init(base_constraints)
        self.base_integrator = LocalizationIntegrator(context.agent.robot._joints, context.agent.robot.get_joint_symbol_map().joint_map)

        for link, (m, b, n) in context.agent.robot.collision_avoidance_links.items():
            link_pos = pos_of(context.agent.robot.get_fk_expression('map', link))
            c2l = link_pos - pos_of(camera.pose)
            ray_dist = norm(cross(view_dir, c2l))
            d_dist   = dot(view_dir, c2l)
            soft_constraints['{}_non-obstruction'.format(link)] = SC(sin(camera.hfov * 0.5) * (d_dist / cos(camera.hfov* 0.5)) - ray_dist, 100, 1, ray_dist)

        self.log_length = 20
        self.base_ang_vels  = np.ones(self.log_length) * 1000
        self.base_lin_vels  = np.ones(self.log_length) * 1000
        self.obs_vels       = np.ones(self.log_length) * -10000
        self.log_cursor     = 0
        self.last_update = None
        self.last_obs_ub = 10000.0
        self.stuck_timeout = rospy.Time.now()
        self.stuck_duration = rospy.Duration(1.5)

        context.agent.add_data_cb(self.data_id, self.update_objects)
        super(ObservationController, self).init(soft_constraints, True)
        self.current_subs[self.s_base_weight] = 1.0


    def local_nav(self):
        self.pub_cancel.publish(GoalIDMsg())
        self.global_nav_mode = False
        self.current_subs[self.s_base_weight] = 1.0
        print('Taking base control from global navigation')

    def global_nav(self, goal_location):
        msg = ATPGoalMsg()
        msg.goal.pose.x = goal_location[0]
        msg.goal.pose.y = goal_location[1]
        msg.goal.pose.theta = goal_location[2]
        self.pub_set_nav_goal.publish(msg)
        self.global_nav_mode = True
        self.current_subs[self.s_base_weight] = 10000
        print('Giving base control to global navigation')

    def get_cmd(self, nWSR=None):
        if self.search_objects is not None:
            self.re_eval_focused_object()
        if self.goal_obj_index > -1:
            cmd = super(ObservationController, self).get_cmd(nWSR)
            obs_lb, obs_ub = self.qp_problem_builder.get_a_bounds(obs_dist_constraint)
            if VULCAN_FALLBACK:
                if not self.global_nav_mode:
                    if self.is_stuck(cmd):
                        self.find_global_pose()
            else:
                if obs_ub < UBA_BOUND and not self.global_nav_mode:
                    self.find_global_pose()

            if self.global_nav_mode:
                del cmd['base_angular_joint']
                del cmd['base_linear_joint']
                if obs_ub >= UBA_BOUND:
                    self.local_nav()
            return cmd
            #self.print_fn('Waiting for updated object information')
        return {}

    def is_stuck(self, last_cmd):
        return False
        now = rospy.Time.now()
        self.base_ang_vels[self.log_cursor] = last_cmd['base_angular_joint']
        self.base_lin_vels[self.log_cursor] = last_cmd['base_linear_joint']

        obs_lb, obs_ub = self.qp_problem_builder.get_a_bounds(obs_dist_constraint)
        if self.last_update != None:
            #print('Is stuck delta t: {}'.format((now - self.last_update).to_sec()))
            delta_t = (now - self.last_update).to_sec()
            v_obs = (obs_ub - self.last_obs_ub) / delta_t
            self.obs_vels[self.log_cursor] = v_obs

            avg_base_vel = np.average(self.base_lin_vels)
            avg_obs_vel = np.average(self.obs_vels)

            avg_ang_vel = np.average(self.base_ang_vels)
            abs_avg_ang_vel = np.average(np.abs(self.base_ang_vels))
            jitter_factor = abs_avg_ang_vel - avg_ang_vel

            msg = Float64Msg()
            msg.data = avg_obs_vel
            self.pub_obs_uba.publish(msg)
            
            msg = Float64Msg()
            msg.data = jitter_factor
            self.pub_jitter.publish(msg)

            if obs_ub < UBA_BOUND and \
               abs(obs_ub) - opt_obs_falloff > 0 and \
               avg_obs_vel < abs(obs_ub) - opt_obs_falloff and \
               (avg_ang_vel < 0.1 or jitter_factor > 0.3) and \
               avg_obs_vel < 0.1:
                print('Stuck:\n  avg vl: {}\n  avg va: {}\n  avg ov: {}\n      ov: {}\n  avg |va|: {}\n  jf: {}'.format(
                      avg_base_vel, avg_ang_vel, avg_obs_vel, abs(obs_ub), abs_avg_ang_vel, jitter_factor))
                return True
        
        self.last_obs_ub = obs_ub
        self.log_cursor = (self.log_cursor + 1) % self.log_length
        self.last_update = now


    def update_objects(self, gmm_objects):
        self.search_objects = gmm_objects
        self.update_object_terms()

    def reset_search(self):
        self.search_objects = None
        self.goal_obj_index = -1
        self.goal_gmm_index = -1
        self.last_t = None

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
            self.base_ang_vels  = np.ones(self.log_length) *  10000
            self.base_lin_vels  = np.ones(self.log_length) *  10000
            self.obs_vels       = np.ones(self.log_length) * -10000
            self.update_object_terms()

    def find_global_pose(self, iterations=100, time_step=0.5, samples=10, spread=2.0):
        if self.goal_obj_index == -1:
            return

        trajectory_log = []
        base_subs = {s: (self.current_subs[s] if s in self.current_subs else 0.0) for s in self.global_base_controller.free_symbols}
        self.global_base_controller.current_subs = base_subs
        #print('\n  '.join(['{}: {}'.format(str(s), v) for s, v in base_subs.items()]))
        for cpq in self.bc_cpq:
            cpq.reset_subs_dict(base_subs)

        good = False
        print('Doing initial search...')
        for x in range(iterations):
            next_cmd = self.global_base_controller.get_cmd()
            self.base_integrator.integrate(base_subs, next_cmd, time_step)
            #print('\n  '.join(['{}: {}'.format(str(s), v) for s, v in base_subs.items()]))
            #positions = {j: (JointState(base_subs[s], 0, 0) if s in base_subs else JointState(self.current_subs[s], 0,0)) for j, s in self.base_integrator.symbol_map.items()}
            #trajectory_log.append(StampedData(rospy.Time.from_sec(x * time_step), positions))
            # if self.visualizer != None:
            #     self.visualizer.begin_draw_cycle()
            #     self.visualizer.draw_robot_pose('runner', self.robot, {j: s.position for j, s in positions.items()})
            #     self.visualizer.render()

            if self.global_base_controller.qp_problem_builder.constraints_met(lbThreshold=LBA_BOUND, ubThreshold=UBA_BOUND, names=[obs_dist_constraint]):
                print('Terminated initial search prematurely after {} iterations'.format(x))
                break

        print('Doing collision search...')
        x = 0

        coll_subs = []
        for y in range(samples):
            c = base_subs.copy()
            c[self.base_integrator.symbol_map['localization_x']] += spread * (0.5 - random.random())
            c[self.base_integrator.symbol_map['localization_y']] += spread * (0.5 - random.random())
            coll_subs.append(c)



        while not good:
            x += 1
            # self.visualizer.begin_draw_cycle()
            for base_subs in coll_subs:
                self.global_base_controller.current_subs = base_subs
                self.bullet_bot.set_joint_positions({j: base_subs[s] for j, s in self.base_integrator.symbol_map.items() if s in base_subs})
                quat = pb.getQuaternionFromEuler((0,0,base_subs[self.base_integrator.symbol_map['localization_z_ang']]))
                self.bullet_bot.set_pose(Frame((base_subs[self.base_integrator.symbol_map['localization_x']],
                                                base_subs[self.base_integrator.symbol_map['localization_y']],
                                                0), quat))
                for cpq in self.bc_cpq:
                    cpq.update_subs_dict(self.simulator, base_subs, self.visualizer)

                # self.base_integrator.integrate(base_subs, self.global_base_controller.get_cmd(), time_step)
                # positions = {j: (JointState(base_subs[s], 0, 0) if s in base_subs else JointState(self.current_subs[s], 0,0)) for j, s in self.base_integrator.symbol_map.items()}
                #trajectory_log.append(StampedData(rospy.Time.from_sec(x * time_step), positions))
                # if self.visualizer != None:
                #     self.visualizer.draw_robot_pose('runner', self.robot, {j: s.position for j, s in positions.items()})

                if self.global_base_controller.qp_problem_builder.constraints_met(lbThreshold=LBA_BOUND, ubThreshold=UBA_BOUND, names=self.essential_base_constraints):
                    good = True
                    print('Terminated collision resolve search after {} iterations'.format(x))
                    break
            #self.visualizer.render()

        goal_x = base_subs[self.base_integrator.symbol_map['localization_x']]
        goal_y = base_subs[self.base_integrator.symbol_map['localization_y']]
        goal_theta = base_subs[self.base_integrator.symbol_map['localization_z_ang']]

        self.global_nav((goal_x, goal_y, goal_theta))
        self.visualizer.begin_draw_cycle('nav_goal')
        self.visualizer.draw_mesh('nav_goal', frame3_rpy(0,0,0, [goal_x, goal_y, 0]), [1.0] * 3, 'package://gebsyas/meshes/nav_arrow.dae', r=1.0)
        self.visualizer.render('nav_goal')


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
        self.pub_cancel.publish(GoalIDMsg())
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
        self.last_obs_ub  = None
        self.base_timeout_duration = rospy.Duration(1.0)
        self.external_navigation = False
        self.visualizer = ROSVisualizer('observation_runner', 'map')

    def run(self):
        """Starts the run of the controller."""
        now = rospy.Time.now()
        self.terminate = False
        self.base_timeout = now + self.base_timeout_duration
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
        if self.last_update != None:
            delta_t = (now - self.last_update).to_sec()
        self.controller.set_robot_js(joint_state)
        try:
            command = self.controller.get_cmd()
        except Exception as e:
            self.terminate = True
            traceback.print_exc()
            print(e)
            return

        # obs_lb, obs_ub = self.controller.qp_problem_builder.get_a_bounds(obs_dist_constraint)
        # if obs_ub < UBA_BOUND and self.last_update != None:
        #     v_obs = (obs_ub - self.last_obs_ub) / delta_t
        #     if v_obs >= 0.1:
        #         self.base_timeout = now + self.base_timeout_duration
        #     if now >= self.base_timeout and not self.external_navigation:
        #         print('vulcan would kick in now. v_obs: {}'.format(v_obs))
        #         symbol_map = self.robot.get_joint_symbol_map().joint_map
        #         runner = CollisionResolverRunner(self.controller.global_base_controller, LocalizationIntegrator(self.robot._joints, symbol_map), self.visualizer)
        #         self.visualizer.begin_draw_cycle('resolved_pose')
        #         if runner.run():
        #             self.visualizer.draw_robot_trajectory('resolved_pose', self.robot, runner.trajectory_log, tint=(0,1,0,1))
        #         else:
        #             self.visualizer.draw_robot_trajectory('resolved_pose', self.robot, runner.trajectory_log, tint=(1,0,0,1))
        #         self.visualizer.render()
        #         self.external_navigation = True

        # self.last_obs_ub = obs_ub

        # if self.external_navigation:
        #     if 'base_linear_joint' in command:
        #         del command['base_linear_joint']

        #     if 'base_angular_joint' in command:
        #         del command['base_angular_joint']


        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()
        self.f_send_command(command)
        if self.controller.goal_obj_index > -1:
            cov = self.controller.current_cov
            c_obj = self.controller.get_current_object()
            t_var = c_obj.good_variance if hasattr(c_obj, 'good_variance') else [self.t_variance] * 3
            self.terminate = self.controller.current_weight >= self.t_weight and \
                             abs(cov[0,0]) <= t_var[0] and \
                             abs(cov[1,1]) <= t_var[1] and \
                             abs(cov[2,2]) <= t_var[2]
        self.last_update = now

def run_observation_controller(robot, controller, agent, variance=0.02, weight=0.9):
    """Comfort function for easily instantiating and running an inequality runner."""
    runner = ObservationRunner(robot, controller, agent.act, agent.add_js_callback, variance, weight)
    constraints_met, lf = runner.run()
    return constraints_met, lf, runner.trajectory_log
