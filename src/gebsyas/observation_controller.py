import traceback
import rospy
import numpy as np

from giskardpy.symengine_wrappers import *
from giskardpy.input_system import FrameInput, Vector3Input
from giskardpy.qp_problem_builder import SoftConstraint as SC
from gebsyas.actions import PActionInterface, Action
from gebsyas.basic_controllers import run_ineq_controller
from gebsyas.data_structures import SymbolicData
from gebsyas.bc_controller_wrapper import BCControllerWrapper
from gebsyas.bullet_based_controller import InEqBulletController
from gebsyas.dl_reasoning import DLRigidGMMObject, DLRigidObject, DLDisjunction
from gebsyas.predicates import ClearlyPerceived, PInstance
from gebsyas.utils   import symbol_formatter

from blessed import Terminal


class ObservationController(InEqBulletController):

    def init(self, context, proximity_frame, camera, object_id):
        self.context = context
        self.object_id = object_id
        obj = context.agent.data_state[object_id].data
        if obj is None:
            raise Exception('Data with Id "{}" does not exist.'.format(object_id))

        if not DLRigidGMMObject.is_a(obj):
            raise Exception('Object "{}" is not probabilistic.'.format(object_id))


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

        self.frame_input = FrameInput.prefix_constructor('position', 'orientation', symbol_formatter)
        self.current_subs[self.frame_input.x] = obj.gmm[0].pose[0, 3]
        self.current_subs[self.frame_input.y] = obj.gmm[0].pose[1, 3]
        self.current_subs[self.frame_input.z] = obj.gmm[0].pose[2, 3]
        quat = quaternion_from_matrix(obj.gmm[0].pose)
        self.current_subs[self.frame_input.qx] = quat[0]
        self.current_subs[self.frame_input.qy] = quat[1]
        self.current_subs[self.frame_input.qz] = quat[2]
        self.current_subs[self.frame_input.qw] = quat[3]

        pose = obj.gmm[0].pose

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

        for link, (m, b) in context.agent.robot.collision_avoidance_links.items():
            link_pos = pos_of(context.agent.robot.get_fk_expression('map', link))
            c2l = link_pos - pos_of(camera.pose)
            ang = acos(dot(view_dir, c2l) / norm(c2l))
            soft_constraints['{}_non-obstruction'.format(link)] = SC(camera.hfov * 0.5 -ang, 3.14, 1, ang)

        context.agent.add_data_cb(object_id, self.update_object)
        super(ObservationController, self).init(soft_constraints)

    def update_object(self, gmm_object):
        pose = gmm_object.gmm[0].pose
        cov  = np.array(gmm_object.gmm[0].cov.tolist(), dtype=float)
        self.current_subs[self.frame_input.x] = pose[0, 3]
        self.current_subs[self.frame_input.y] = pose[1, 3]
        self.current_subs[self.frame_input.z] = pose[2, 3]
        quat = quaternion_from_matrix(pose)
        self.current_subs[self.frame_input.qx] = quat[0]
        self.current_subs[self.frame_input.qy] = quat[1]
        self.current_subs[self.frame_input.qz] = quat[2]
        self.current_subs[self.frame_input.qw] = quat[3]
        w, v = np.linalg.eig(cov[:3, :3])
        pos_eig = w * v
        for x in range(len(self.evecs)):
            self.current_subs[self.evecs[x].x] = pos_eig[0, x] # if np.isreal(pos_eig[0, x]) else 0
            self.current_subs[self.evecs[x].y] = pos_eig[1, x] # if np.isreal(pos_eig[1, x]) else 0
            self.current_subs[self.evecs[x].z] = pos_eig[2, x] # if np.isreal(pos_eig[2, x]) else 0


    def stop(self):
        self.context.agent.remove_data_cb(self.object_id, self.update_object)
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
                                                self.clear_and_print) #context.log
            motion_ctrl.init(context, 
                             context.agent.robot.get_fk_expression('map', 'base_link') * translation3(0.1, 0, 0),
                             context.agent.robot.camera, 
                             self.object_id)

            motion_success, m_lf, t_log = run_ineq_controller(context.agent.robot, motion_ctrl, 45.0, 3.5, context.agent, task_constraints=None)

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