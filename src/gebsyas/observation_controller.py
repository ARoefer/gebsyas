import traceback
import rospy
import numpy as np
import pybullet as pb
import random

import math

from multiprocessing import Lock

from iai_bullet_sim.utils import Frame
from iai_bullet_sim.basic_simulator import hsva_to_rgba

from gebsyas.actions import PActionInterface, Action
from gebsyas.basic_controllers import run_ineq_controller, InEqController
from gebsyas.bc_controller_wrapper import BCControllerWrapper
from gebsyas.bullet_based_controller import InEqBulletController
from gebsyas.constants import LBA_BOUND, UBA_BOUND
from gebsyas.data_structures import SymbolicData, StampedData, JointState
from gebsyas.dl_reasoning import DLRigidGMMObject, DLRigidObject, DLDisjunction
from gebsyas.headless_controller_runner import CollisionResolverRunner, LocalizationIntegrator
from gebsyas.predicates import ClearlyPerceived, PInstance
from gebsyas.observation_helpers import *
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.utils   import symbol_formatter, real_quat_from_matrix
from giskardpy.input_system import FrameInput, Vector3Input
from giskardpy.qp_problem_builder import SoftConstraint as SC
from giskardpy.symengine_wrappers import *
from geometry_msgs.msg import PoseStamped
from gop_gebsyas_msgs.msg import NonPromisingComponent as NPCMsg


from navigation_msgs.msg import NavToPoseActionGoal as ATPGoalMsg
from navigation_msgs.msg import NavToPoseActionResult as ATPResultMsg
from actionlib_msgs.msg import GoalID as GoalIDMsg
from actionlib_msgs.msg import GoalStatus as GoalStatusMsg

from std_msgs.msg import Float64 as Float64Msg
from std_msgs.msg import String as StringMsg

from blessed import Terminal

obs_dist_constraint = 'obs_dist'
yaw_constraint = 'occlusion_escape_yaw'

# Switches the usage of vulcan.
#   True:  Vulcan is only used as a fall when giskard gets stuck
#   False: Vulcan is used for all global navigation
VULCAN_FALLBACK = False
VULCAN_DIST     = 4.0
opt_obs_falloff = 0.2
rating_scale    = 1.0

def blank_pass(a):
    pass

class ObservationController(InEqBulletController):

    def init(self, context, proximity_frame, camera, data_id='searched_objects'):
        self.context = context
        self.data_id = data_id
        self.proximity_frame = proximity_frame
        self.current_cov = eye(6)
        self.current_cov_occluded = False
        self.current_weight = 0.0
        self.pub_set_nav_goal = rospy.Publisher('/nav_to_pose/goal', ATPGoalMsg, queue_size=0, tcp_nodelay=True)
        self.pub_cancel       = rospy.Publisher('/nav_to_pose/cancel', GoalIDMsg, queue_size=0, tcp_nodelay=True)
        self.pub_obs_uba = rospy.Publisher('observation_controller/uba', Float64Msg, queue_size=0, tcp_nodelay=True)
        self.pub_jitter = rospy.Publisher('observation_controller/base_jitter', Float64Msg, queue_size=0, tcp_nodelay=True)
        self.pub_bad_component  = rospy.Publisher('/bad_component', NPCMsg, queue_size=0, tcp_nodelay=True)
        self.pub_pursued_object = rospy.Publisher('/pursued_object', StringMsg, queue_size=0, tcp_nodelay=True)
        self.global_nav_mode    = False
        self.sub_nav_result     = rospy.Subscriber('/nav_to_pose/result', ATPResultMsg, self.handle_nav_result, queue_size=1)
        self.occlusion_maps = {}

        # PERCEIVING VARIANCES ACTIVELY
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
        v_e1_flat = diag(1,1,0,1) * v_e1
        v_e2_flat = diag(1,1,0,1) * v_e2
        v_e3_flat = diag(1,1,0,1) * v_e3

        opt_obs_range   = 1.2 # Figure this out by semantic type and observations.

        self.goal_obj_index = -1
        self.goal_gmm_index = -1
        self.search_objects = None

        self.frame_input = FrameInput.prefix_constructor('position', 'orientation', symbol_formatter)

        pose = self.frame_input.get_frame()

        self.camera_position = pos_of(camera.pose)
        view_dir = x_of(camera.pose)
        view_dir_flat = (diag(1,1,0,1) * view_dir) * (1 / (cos(asin(view_dir[2])) + 0.0001))
        c2o      = pos_of(pose) - pos_of(camera.pose)
        z_dist   = norm(c2o)
        look_dir = c2o / z_dist
        obs_ctrl = 1 #1 - (z_dist / opt_obs_range) # ((opt_obs_range - z_dist) / opt_obs_falloff) ** 2
        in_view  = dot(view_dir, look_dir)
        proximity = norm(diag(1,1,0,1) * (pos_of(pose) - pos_of(proximity_frame)))

        co_lin_x = norm(cross(view_dir_flat, v_e1_flat)) #dot(view_dir, v_e1)
        co_lin_y = norm(cross(view_dir_flat, v_e2_flat)) #dot(view_dir, v_e2)
        co_lin_z = norm(cross(view_dir_flat, v_e3_flat)) #dot(view_dir, v_e3)

        # ((camera.hfov - acos(in_view)) / camera.hfov) ** 2
        self.s_pitch_goal = Symbol('pitch_goal')
        self.s_yaw_goal   = Symbol('yaw_goal')
        self.s_occlusion_weight = Symbol('occlusion_weight')

        self.close_enough = (0.5 + 0.5 * tanh(6 - 4 * (proximity / opt_obs_range)))

        look_gain = 10

        s_in_view    = SC((0.95 - in_view) * look_gain,
                          (1 - in_view) * look_gain,
                          (1 + norm(v_e1) + norm(v_e2) + norm(v_e3)) * self.close_enough,
                          in_view)
        s_in_v_dist  = SC(0.5 - proximity, opt_obs_range + opt_obs_falloff - proximity, (1 - 0.25 * self.s_occlusion_weight), proximity)
        s_avoid_near = SC(camera.near - z_dist, 100, 1, z_dist)

        s_v_e1 = SC(-co_lin_x, -co_lin_x, norm(v_e1) * (1 - 0.5 * self.s_occlusion_weight) * self.close_enough, co_lin_x)
        s_v_e2 = SC(-co_lin_y, -co_lin_y, norm(v_e2) * (1 - 0.5 * self.s_occlusion_weight) * self.close_enough, co_lin_y)
        s_v_e3 = SC(-co_lin_z, -co_lin_z, norm(v_e3) * (1 - 0.5 * self.s_occlusion_weight) * self.close_enough, co_lin_z)

        self.proximity = proximity

        soft_constraints = {#'igain'   : s_igain,
                             'near_clip': s_avoid_near,
                             'examine_1': s_v_e1,
                             'examine_2': s_v_e2,
                             'examine_3': s_v_e3,
                             'in_view' : s_in_view,
                             obs_dist_constraint : s_in_v_dist}

        # DEALING WITH OCCLUSIONS
        o2c = diag(1,1,0,1) * (pos_of(proximity_frame) - pos_of(pose)) + diag(0,0,-1,0) * c2o
        o2c_flat = diag(1,1,0,1) * o2c
        o2c_flat_normed = o2c_flat / norm(o2c_flat)
        obs_dir_yaw   = acos(o2c_flat_normed[0]) * fake_sign(o2c_flat_normed[1])
        obs_dir_pitch = asin(o2c[2] / norm(o2c))
        self.obs_dir_pitch = obs_dir_pitch


        self.current_subs[self.s_yaw_goal] = 0
        self.current_subs[self.s_pitch_goal] = 0
        self.current_subs[self.s_occlusion_weight] = 0

        self.robot_can_strafe = 'base_strafe_joint' in self.robot.joint_states_input.joint_map

        if self.robot_can_strafe:
            # MOVE TO EDGE OF OCCLUSION BY USING ANGLES
            yaw_goal_angular = sym_c_dist(obs_dir_yaw, self.s_yaw_goal)
            s_escape_yaw_occlusion = SC(-yaw_goal_angular,
                                        -yaw_goal_angular,
                                        self.s_occlusion_weight,
                                        self.yaw_goal_angular)
            soft_constraints[yaw_constraint] = s_escape_yaw_occlusion
        else:
            goal_pos = diag(1,1,0,1) * (pos_of(pose) + point3(cos(self.s_yaw_goal) * opt_obs_range,
                                                              sin(self.s_yaw_goal) * opt_obs_range, 0))
            direct_drive_dist = norm(diag(1,1,0,1) * (goal_pos - pos_of(proximity_frame)))

            s_drive_base = SC(-direct_drive_dist,
                              -direct_drive_dist,
                              self.s_occlusion_weight,
                              direct_drive_dist)

            soft_constraints[yaw_constraint] = s_drive_base

        soft_constraints['occlusion_escape_pitch'] = SC(self.s_pitch_goal - obs_dir_pitch,
                                      self.s_pitch_goal - obs_dir_pitch,
                                      self.s_occlusion_weight,
                                      obs_dir_pitch)


        # HEADLESS CONTROLLER FOR FINDING POSES FOR EXTERNAL NAVIGATION
        self.global_base_controller = InEqController(context.agent.robot,
                                                           blank_pass,
                                                           True)
        self.s_area_border = Symbol('non_occluded_area_width')
        area_center = vector3(cos(self.s_yaw_goal), sin(self.s_yaw_goal), 0)
        area_ang = acos(dot(o2c_flat_normed, area_center))
        s_base_non_occlusion = SC(-self.s_area_border - area_ang, self.s_area_border - area_ang, self.s_occlusion_weight, area_ang)

        facing_dot = dot(x_of(proximity_frame), -o2c_flat_normed)
        s_restrict_rotation = SC(-facing_dot, 1 - facing_dot, 1, facing_dot)

        base_constraints = {#'in_view' : s_in_view,
                            'not_occluded': s_base_non_occlusion,
                            'dont_face_away': s_restrict_rotation,
                            obs_dist_constraint : s_in_v_dist}
        self.bc_free_symbols = set()
        for sc in base_constraints.values():
            for f in sc:
                if hasattr(f, 'free_symbols'):
                    self.bc_free_symbols = self.bc_free_symbols.union(f.free_symbols)
        #print(self.bc_free_symbols)

        # ADD CLOSEST POINT QUERIES RELATING TO BASE
        self.bc_cpq = []
        for cpq in self.closest_point_queries:
            cac = cpq.generate_constraints()
            for n, ac in cac.items():
                if n not in self.self_avoidance_constraints and len(self.bc_free_symbols.intersection(ac.expression.free_symbols)) > 0:
                    base_constraints.update(cac)
                    self.bc_cpq.append(cpq)
                    break

        self.essential_base_constraints = set(base_constraints.keys())
        #self.essential_base_constraints.remove('in_view')
        self.global_base_controller.init(base_constraints, False, False)
        self.base_integrator = LocalizationIntegrator(context.agent.robot._joints, context.agent.robot.get_joint_symbol_map().joint_map)

        # KEEP LIMBS FROM OBSTRUCTING THE CAMERA'S FIELD OF VIEW
        for link, (m, b, n) in context.agent.robot.collision_avoidance_links.items():
            link_pos = pos_of(context.agent.robot.get_fk_expression('map', link))
            c2l = link_pos - pos_of(camera.pose)
            ray_dist = norm(cross(view_dir, c2l))
            d_dist   = dot(view_dir, c2l)
            #soft_constraints['{}_non-obstruction'.format(link)] = SC(sin(camera.hfov * 0.5) * (d_dist / cos(camera.hfov* 0.5)) - ray_dist, 100, 1, ray_dist)

        # LOGGING FOR "STUCK"-DETECTION
        log_length = 20
        delay = 0.1
        self.base_ang_vels  = ValueLogger(log_length, 1000, delay)
        self.base_lin_vels  = ValueLogger(log_length, 1000, delay)
        self.obs_vels       = FirstDerivativeLogger(log_length, 1000, delay)
        self.occ_vels       = CircularFirstDerivativeLogger(log_length, 1000, delay)
        self.map_lin_vel    = ValueLogger(log_length, 1000, delay)
        self.map_ang_vel    = ValueLogger(log_length, 1000, delay)
        #self.external_vels  = FirstDerivativeLogger(self.log_length) * 1000
        self.main_draw_layers.add('gmm_ratings')

        context.agent.add_data_cb(self.data_id, self.update_objects)
        super(ObservationController, self).init(soft_constraints, True)
        self.current_subs[self.s_base_weight] = 1.0

    def local_nav(self):
        self.pub_cancel.publish(GoalIDMsg())
        self.global_nav_mode = False
        self.current_subs[self.s_base_weight] = 1.0
        self.reset_stuck_markers()
        self.context.log('Taking base control from global navigation.')

    def global_nav(self, goal_location):
        msg = ATPGoalMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.goal.pose.x = goal_location[0]
        msg.goal.pose.y = goal_location[1]
        msg.goal.pose.theta = goal_location[2]
        self.pub_set_nav_goal.publish(msg)
        self.global_nav_mode = True
        self.reset_stuck_markers()
        #self.current_subs[self.s_base_weight] = 10000
        self.context.log('Giving base control to global navigation')

    def handle_nav_result(self, msg):
        if self.global_nav_mode:
            if msg.status.status == GoalStatusMsg.SUCCEEDED or msg.status.status == GoalStatusMsg.PREEMPTED:
                if msg.status.status == GoalStatusMsg.SUCCEEDED:
                    self.context.log('Global nav reached its goal')
                else:
                    self.context.log('Global nav got terminated')
                self.local_nav()
            elif msg.status.status == GoalStatusMsg.ABORTED:
                self.context.log('Global nav aborted navigation')
                self.find_global_pose()

    def get_cmd(self, nWSR=None):
        if self.search_objects is not None:
            self.re_eval_focused_object()
        if self.goal_obj_index > -1:
            #print('Occlusion weight: {}'.format(self.current_subs[self.s_occlusion_weight]))
            cmd = super(ObservationController, self).get_cmd(nWSR)
            obs_lb, obs_ub = self.qp_problem_builder.get_a_bounds(obs_dist_constraint)
            if VULCAN_FALLBACK:
                if not self.global_nav_mode:
                    if self.is_stuck(cmd) or self.proximity.subs(self.current_subs) > VULCAN_DIST:
                        self.find_global_pose()
            else:
                if (obs_ub < UBA_BOUND or self.current_subs[self.s_occlusion_weight] > 0) and not self.global_nav_mode:
                    self.find_global_pose()

            if self.global_nav_mode:
                del cmd['base_angular_joint']
                del cmd['base_linear_joint']
                localization = self.context.agent.data_state['localization'].data
                self.map_lin_vel.log(localization.lvx)
                self.map_ang_vel.log(localization.avz)

                avg_lin_vel = self.map_lin_vel.avg()
                avg_ang_vel = abs(self.map_ang_vel.avg())

                if (obs_ub >= UBA_BOUND and self.current_subs[self.s_occlusion_weight] == 0):# or (avg_lin_vel < 0.02 and avg_ang_vel < 0.1):
                    self.context.log('Robot is close enough to target and currently not forced to avoid occlusions.')
                    self.local_nav()

            #self.context.log('Close enough: {}'.format(self.close_enough.subs(self.current_subs)))

            return cmd
            #self.print_fn('Waiting for updated object information')
        return {}

    def is_stuck(self, last_cmd):
        obs_lb, obs_ub = self.qp_problem_builder.get_a_bounds(obs_dist_constraint)
        updated = self.obs_vels.log(obs_ub)
        self.occ_vels.log(self.current_subs[self.s_yaw_goal])
        self.base_ang_vels.log(last_cmd['base_angular_joint'])
        self.base_lin_vels.log(last_cmd['base_linear_joint'])

        occ_mode = self.current_subs[self.s_occlusion_weight] > 0
        avg_obs_vel  = self.obs_vels.avg()
        avg_occ_vel  = self.occ_vels.avg()
        avg_base_vel = self.base_lin_vels.avg()
        avg_ang_vel  = self.base_ang_vels.avg()
        abs_avg_ang_vel = self.base_ang_vels.abs_avg()
        jitter_factor = abs_avg_ang_vel - avg_ang_vel

        # REGULAR BASE NAVIGATION
        if updated:
            if not occ_mode:
                if obs_ub < UBA_BOUND and \
                   abs(obs_ub) - opt_obs_falloff > 0 and \
                   avg_obs_vel < abs(obs_ub) - opt_obs_falloff and \
                   (avg_ang_vel < 0.1 or jitter_factor > 0.3) and \
                   avg_obs_vel < 0.1:
                    self.context.log('Stuck:\n  avg vl: {}\n  avg va: {}\n  avg ov: {}\n      ov: {}\n  avg |va|: {}\n  jf: {}'.format(
                          avg_base_vel, avg_ang_vel, avg_obs_vel, abs(obs_ub), abs_avg_ang_vel, jitter_factor))
                    return True
            elif not self.global_nav_mode: # NAVIGATION TO CIRCLE OBJECCT
                if avg_occ_vel < 0.03:
                    self.context.log('Stuck:\n  avg vl: {}\n  avg va: {}\n  avg ov: {}\n      ov: {}\n  avg |va|: {}\n  jf: {}'.format(
                          avg_base_vel, avg_ang_vel, avg_obs_vel, abs(obs_ub), abs_avg_ang_vel, jitter_factor))
                    return True
        return False


    def update_objects(self, gmm_objects):
        self.search_objects = gmm_objects
        self.update_object_terms()

    def reset_stuck_markers(self):
        self.obs_vels.reset()
        self.occ_vels.reset()
        self.base_lin_vels.reset()
        self.base_ang_vels.reset()
        self.map_lin_vel.reset()
        self.map_ang_vel.reset()

    def reset_search(self):
        self.search_objects = None
        self.goal_obj_index = -1
        self.goal_gmm_index = -1
        self.last_t = None

    def re_eval_focused_object(self):
        new_obj_index = -1
        new_gmm_index = -1
        best_rating = 100000.0
        draw_offset = vector3(0,0,2)
        flat_robot_pos = diag(1,1,0,1) * pos_of(self.proximity_frame).subs(self.current_subs)
        arrow_start = flat_robot_pos + draw_offset

        self.visualizer.begin_draw_cycle('gmm_ratings')
        for n in range(len(self.search_objects.search_object_list)):
            gmm_object = self.search_objects.search_object_list[n]
            object_weight = self.search_objects.weights[n]
            for x in range(len(gmm_object.gmm)):
                if gmm_object.gmm[x].weight > 0.0:
                    flat_gc_pos = diag(1,1,0,1) * pos_of(gmm_object.gmm[x].pose)
                    r2gc   = flat_gc_pos - flat_robot_pos
                    rating = math.exp(rating_scale * norm(r2gc)) / (gmm_object.gmm[x].weight * object_weight)
                    color = hsva_to_rgba((1.0 - gmm_object.gmm[x].weight) * 0.65, 1, 1, 0.7)
                    self.visualizer.draw_arrow('gmm_ratings', arrow_start, flat_gc_pos + draw_offset, *color)
                    self.visualizer.draw_text('gmm_ratings', arrow_start + 0.5 * r2gc, '{:.2f}'.format(float(rating)))
                    if rating < best_rating:
                        new_obj_index = n
                        new_gmm_index = x
                        best_rating = rating
        self.visualizer.render('gmm_ratings')

        if new_obj_index == -1:
            return
        elif new_obj_index != self.goal_obj_index or new_gmm_index != self.goal_gmm_index:
            if new_obj_index != self.goal_obj_index:
                msg = StringMsg()
                msg.data = ''.join([i for i in self.search_objects.search_object_list[new_obj_index].id if not i.isdigit()])
                self.pub_pursued_object.publish(msg)

            self.goal_obj_index = new_obj_index
            self.goal_gmm_index = new_gmm_index
            self.base_ang_vels.reset()
            self.base_lin_vels.reset()
            self.obs_vels.reset()
            self.occ_vels.reset()
            self.current_subs[self.s_occlusion_weight] = 0
            self.update_object_terms()

    def find_global_pose(self, iterations=100, time_step=0.25, samples=20, spread=3.0):
        if self.goal_obj_index == -1:
            return

        trajectory_log = []
        base_subs = {s: (self.current_subs[s] if s in self.current_subs else 0.0) for s in self.global_base_controller.free_symbols}
        base_subs[self.s_area_border] = math.pi
        current_object = self.get_current_object()
        if current_object.id in self.occlusion_maps and current_object.gmm[self.goal_gmm_index].id in self.occlusion_maps[current_object.id]:
            oc_map = self.occlusion_maps[current_object.id][current_object.gmm[self.goal_gmm_index].id]
            area_width = (2 * math.pi - oc_map.get_occluded_area()[0])
            center_angle = wrap_to_circle(oc_map.min_corner[0] - 0.5 * area_width)
            base_subs[self.s_occlusion_weight] = 2
            base_subs[self.s_yaw_goal] = center_angle
            base_subs[self.s_area_border] = 0.5 * (area_width * 0.6)

        self.global_base_controller.current_subs = base_subs
        #print('\n  '.join(['{}: {}'.format(str(s), v) for s, v in base_subs.items()]))
        for cpq in self.bc_cpq:
            cpq.reset_subs_dict(base_subs)

        good = False
        self.context.log('Doing initial search...')
        self.visualizer.begin_draw_cycle('runner_step_1')
        for x in range(iterations):
            next_cmd = self.global_base_controller.get_cmd()
            self.base_integrator.integrate(base_subs, next_cmd, time_step)
            #print('\n  '.join(['{}: {}'.format(str(s), v) for s, v in base_subs.items()]))
            #positions = {j: (JointState(base_subs[s], 0, 0) if s in base_subs else JointState(self.current_subs[s], 0,0)) for j, s in self.base_integrator.symbol_map.items()}
            #trajectory_log.append(StampedData(rospy.Time.from_sec(x * time_step), positions))
            if self.visualizer != None:
                self.visualizer.draw_mesh('runner_step_1', frame3_rpy(0,0,base_subs[self.base_integrator.symbol_map['localization_z_ang']],
                        [base_subs[self.base_integrator.symbol_map['localization_x']],
                         base_subs[self.base_integrator.symbol_map['localization_y']], 0]), [1.0] * 3, 'package://gebsyas/meshes/nav_arrow.dae', g=1.0, b=1.0, a=1.0)

            if self.global_base_controller.qp_problem_builder.constraints_met(lbThreshold=LBA_BOUND, ubThreshold=UBA_BOUND, names=[obs_dist_constraint]):
                self.context.log('Terminated initial search prematurely after {} iterations'.format(x))
                break
        self.visualizer.render('runner_step_1')

        self.context.log('Doing collision search...')
        x = 1

        coll_subs = []
        for y in range(samples):
            c = base_subs.copy()
            c[self.base_integrator.symbol_map['localization_x']] += spread * (0.5 - random.random())
            c[self.base_integrator.symbol_map['localization_y']] += spread * (0.5 - random.random())
            coll_subs.append(c)

        constraint_counter = {c: 0 for c in self.essential_base_constraints}

        while not good:
            self.visualizer.begin_draw_cycle('runner_step_2')
            for c_base_subs in coll_subs:
                self.global_base_controller.current_subs = c_base_subs
                self.bullet_bot.set_joint_positions({j: c_base_subs[s] for j, s in self.base_integrator.symbol_map.items() if s in c_base_subs})
                quat = pb.getQuaternionFromEuler((0,0,c_base_subs[self.base_integrator.symbol_map['localization_z_ang']]))
                self.bullet_bot.set_pose(Frame((c_base_subs[self.base_integrator.symbol_map['localization_x']],
                                                c_base_subs[self.base_integrator.symbol_map['localization_y']],
                                                0), quat))
                for cpq in self.bc_cpq:
                    cpq.update_subs_dict(self.simulator, c_base_subs)#, self.visualizer)

                self.base_integrator.integrate(c_base_subs, self.global_base_controller.get_cmd(), time_step)
                for c in self.essential_base_constraints:
                    lbA, ubA = self.global_base_controller.qp_problem_builder.get_a_bounds(c)
                    if lbA > LBA_BOUND or ubA < UBA_BOUND:
                        constraint_counter[c] += 1
                # positions = {j: (JointState(c_base_subs[s], 0, 0) if s in base_subs else JointState(self.current_subs[s], 0,0)) for j, s in self.base_integrator.symbol_map.items()}
                # trajectory_log.append(StampedData(rospy.Time.from_sec(x * time_step), positions))
                # if self.visualizer != None:
                #     self.visualizer.draw_robot_pose('runner', self.robot, {j: s.position for j, s in positions.items()})

                # positions = {j: (JointState(c_base_subs[s], 0, 0) if s in base_subs else JointState(self.current_subs[s], 0,0)) for j, s in self.base_integrator.symbol_map.items()}
                # trajectory_log.append(StampedData(rospy.Time.from_sec(x * time_step), positions))
                if self.visualizer != None:
                     self.visualizer.draw_mesh('runner_step_2', frame3_rpy(0,0,c_base_subs[self.base_integrator.symbol_map['localization_z_ang']],
                                [c_base_subs[self.base_integrator.symbol_map['localization_x']],
                                 c_base_subs[self.base_integrator.symbol_map['localization_y']], 0]), [1.0] * 3, 'package://gebsyas/meshes/nav_arrow.dae', r=1.0, g=0.6, a=1.0)

                if self.global_base_controller.qp_problem_builder.constraints_met(lbThreshold=LBA_BOUND, ubThreshold=UBA_BOUND, names=self.essential_base_constraints):
                    good = True
                    base_subs = c_base_subs
                    print('Terminated collision resolve search after {} iterations'.format(x))
                    break
            if x > 50:
                coll_subs = []
                for y in range(samples):
                    c = base_subs.copy()
                    c[self.base_integrator.symbol_map['localization_x']] += spread * (0.5 - random.random())
                    c[self.base_integrator.symbol_map['localization_y']] += spread * (0.5 - random.random())
                    coll_subs.append(c)
                constraint_counter = {c: 0 for c in self.essential_base_constraints}
                x = 1

            self.visualizer.render('runner_step_2')
            self.context.log('Total Iterations: {}\nConstraint hit proportion:\n  {}'.format(x, '\n  '.join(['{}: {}'.format(k, v / float(x * y)) for v, k in sorted([(t[1], t[0]) for t in constraint_counter.items()])])))
            x += 1

        goal_x = base_subs[self.base_integrator.symbol_map['localization_x']]
        goal_y = base_subs[self.base_integrator.symbol_map['localization_y']]
        goal_theta = base_subs[self.base_integrator.symbol_map['localization_z_ang']]

        self.global_nav((goal_x, goal_y, goal_theta))
        self.visualizer.begin_draw_cycle('nav_goal')
        self.visualizer.draw_mesh('nav_goal', frame3_rpy(0,0,goal_theta, [goal_x, goal_y, 0]), [1.0] * 3, 'package://gebsyas/meshes/nav_arrow.dae', r=1.0, a=1.0)
        positions = {j: (JointState(base_subs[s], 0, 0) if s in base_subs else JointState(self.current_subs[s], 0,0)) for j, s in self.base_integrator.symbol_map.items()}
        if self.visualizer != None:
            self.visualizer.draw_robot_pose('nav_goal', self.robot, {j: s.position for j, s in positions.items()})
        self.visualizer.render('nav_goal')


    def update_object_terms(self):
        if self.goal_obj_index == -1 or self.goal_obj_index >= len(self.search_objects.search_object_list):
            return

        gmm_object = self.search_objects.search_object_list[self.goal_obj_index]
        if self.goal_gmm_index >= len(gmm_object.gmm):
            self.goal_obj_index = -1
            self.goal_gmm_index = -1
            return

        gc = gmm_object.gmm[self.goal_gmm_index]
        pose = gc.pose
        if gc.occluded:
            if gmm_object.id not in self.occlusion_maps:
                self.occlusion_maps[gmm_object.id] = {}
            if gc.id not in self.occlusion_maps[gmm_object.id]:
                oc_map = OcclusionMap()
                self.occlusion_maps[gmm_object.id][gc.id] = oc_map
            else:
                oc_map = self.occlusion_maps[gmm_object.id][gc.id]

            c_pos = self.camera_position.subs(self.current_subs)
            goal_yaw, goal_pitch = oc_map.update(c_pos, pos_of(pose))
            #print('   goal_pitch: {}\ncurrent pitch: {}'.format(goal_pitch, self.obs_dir_pitch.subs(self.current_subs)))
            if oc_map.is_closed():
                msg = NPCMsg()
                msg.object_id = int(''.join([c for c in gmm_object.id if c.isdigit()]))
                msg.component_id = gc.id
                self.current_subs[self.s_occlusion_weight] = 0
                print('Bad GMM component found!')
                self.pub_bad_component.publish(msg)
            else:
                #print('Goal yaw: {}\nYaw term: {}'.format(goal_yaw, self.yaw_goal_angular.subs(self.current_subs)))
                self.current_subs[self.s_yaw_goal] = goal_yaw
                self.current_subs[self.s_pitch_goal] = goal_pitch
                self.current_subs[self.s_occlusion_weight] = 2

                self.visualizer.begin_draw_cycle('occlusion_map')
                oc_map.draw(self.visualizer, pos_of(pose), 0.5)
                self.visualizer.render('occlusion_map')
        else:
            self.current_subs[self.s_occlusion_weight] = 0


        #print('New best gmm: {}\nAt location:\n{}\nWeight: {}'.format(self.goal_gmm_index, str(pos_of(pose)), gc.weight))
        self.current_cov    = np.array(gc.cov.tolist(), dtype=float)
        self.current_weight = gc.weight
        self.current_cov_occluded = gc.occluded
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

            motion_success, m_lf, t_log = run_observation_controller(context.agent.robot, motion_ctrl, context.agent, 0.015, 0.9)

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

        self.controller.local_nav()
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

        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()
        self.f_send_command(command)
        if self.controller.goal_obj_index > -1:
            cov = self.controller.current_cov
            c_obj = self.controller.get_current_object()
            t_var = c_obj.good_variance if hasattr(c_obj, 'good_variance') else ([self.t_variance] * 3) + [5]
            self.terminate = not self.controller.current_cov_occluded and \
                             self.controller.current_weight >= self.t_weight and \
                             sqrt(abs(cov[0,0])) <= t_var[0] and \
                             sqrt(abs(cov[1,1])) <= t_var[1] and \
                             sqrt(abs(cov[2,2])) <= t_var[2] #and \
                                  #abs(cov[3,3]) >= t_var[3]
            if self.terminate:
                print(sqrt(abs(cov[0,0])))
                print(sqrt(abs(cov[1,1])))
                print(sqrt(abs(cov[2,2])))
        self.last_update = now

def run_observation_controller(robot, controller, agent, variance=0.02, weight=0.9):
    """Comfort function for easily instantiating and running an inequality runner."""
    runner = ObservationRunner(robot, controller, agent.act, agent.add_js_callback, variance, weight)
    constraints_met, lf = runner.run()
    return constraints_met, lf, runner.trajectory_log
