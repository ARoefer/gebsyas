import rospy
import math
import traceback


from gebsyas.gaussian_observer import GaussianInspector, Camera, GaussianComponent
from gebsyas.utils             import real_quat_from_matrix

from geometry_msgs.msg    import Pose        as PoseMsg, \
                                 PoseArray   as PoseArrayMsg, \
                                 PoseStamped as PoseStampedMsg, \
                                 Twist       as TwistMsg
from gop_gebsyas_msgs.msg import SearchObjectList as SearchObjectListMsg, \
                                 ViewPose         as ViewPoseMsg, \
                                 ViewPoseList     as ViewPoseListMsg

from gop_gebsyas_msgs.srv import GetViewPoses         as GetViewPosesSrv
from gop_gebsyas_msgs.srv import GetViewPosesResponse as GetViewPosesResponseMsg

from nav_msgs.srv    import GetPlan    as GetPlanSrv
from sensor_msgs.msg import JointState as JointStateMsg

from kineverse.gradients.diff_logic         import Position
from kineverse.model.paths                  import Path
from kineverse.network.ros_conversion       import encode_pose
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

tucked_arm = {'wrist_roll_joint': 0.0, 'shoulder_pan_joint': 1.32, 'elbow_flex_joint': 1.72, 'forearm_roll_joint': 0.0, 'upperarm_roll_joint': -0.2, 'wrist_flex_joint': 1.66, 'shoulder_lift_joint': 1.4}

def str_list(l):
    return [str(x) for x in l]

class ViewPoseGenerator(object):
    def __init__(self, km, camera, sym_loc_x, sym_loc_y, sym_loc_a, service_name, collision_link_paths=[], vismodel=None):
        self.km = km

        self.visualizer = ROSBPBVisualizer('debug_vis', 'map') if rospy.get_param('~visualization', True) else None
        self.n_iterations     = rospy.get_param('~iterations', 100)
        self.n_samples        = rospy.get_param('~samples', 10)
        self.integration_step = max(0.02, min(1.0, rospy.get_param('~integration_step', 0.2)))
        self.equilibrium      = rospy.get_param('~equilibrium', 0.05)
        visualize_iterations  = rospy.get_param('~visualize_iterations', False)
        tilt_limit_min        = min(rospy.get_param('~tilt_limit_min', 0.4), math.pi * 0.5)
        tilt_limit_max        = max(tilt_limit_min, min(rospy.get_param('~tilt_limit_max', 0.9), math.pi * 0.5))
        print('Visualize final pose: {}\nVisualize iterations: {}\nMax iterations: {}\nSamples per component: {}\nIntegration step: {}\nEquilibrium threshold: {}\nTilt range: [{}, {}]\n'.format(self.visualizer is not None, visualize_iterations, self.n_iterations, self.n_samples, self.integration_step, self.equilibrium, tilt_limit_min, tilt_limit_max))

        if self.visualizer is not None and visualize_iterations:
            self.gi = GaussianInspector(km, camera, sym_loc_x, sym_loc_y, sym_loc_a, 0.2, collision_link_paths, self.visualizer, tilt_limit_min, tilt_limit_max) 
        else:
            self.gi = GaussianInspector(km, camera, sym_loc_x, sym_loc_y, sym_loc_a, 0.2, collision_link_paths, None, tilt_limit_min, tilt_limit_max) 

        self.service   = rospy.Service(service_name, GetViewPosesSrv, self.srv_generate_view_poses)
        self.srv_compute_path = rospy.ServiceProxy('/move_base/make_plan', GetPlanSrv)
        self.robot_pose       = PoseStampedMsg()
        self.robot_pose.header.frame_id = 'map'
        self.robot_pose.pose.orientation.w = 1

        self.goal_pose = PoseStampedMsg()
        self.goal_pose.header.frame_id = 'map'
        self.pub_pose_debug = rospy.Publisher('/debug_generated_view_poses', PoseArrayMsg, queue_size=1, tcp_nodelay=True)
        self.pub_initial_pose_debug = rospy.Publisher('/debug_initial_view_poses', PoseArrayMsg, queue_size=1, tcp_nodelay=True)
        

        print('Camera free_symbols:\n{}'.format('\n'.join(str_list(camera.pose.free_symbols))))
        joint_prefix = Path(next(iter(camera.pose.free_symbols)))[:-1]

        if self.visualizer:
            draw_filter = {str(p) for p in collision_link_paths}
            self.visualizer.begin_draw_cycle('world')
            for name, obj in self.km._collision_objects.items():
                if name not in draw_filter:
                    self.visualizer.draw_collision_object('world', obj)
            self.visualizer.render('world')
            
            if vismodel is not None:
                self.robot_subworld = vismodel.get_active_geometry(camera.pose.free_symbols, static_state={Position((joint_prefix + (k,)).to_symbol()): v for k, v in tucked_arm.items()}, include_static=False) if vismodel is not None else None
                
    # goal as x, y, a in map
    def compute_path_length(self, goal):
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pose.pose.position.x = goal[0]
        self.goal_pose.pose.position.y = goal[1]
        self.goal_pose.pose.orientation.z = math.sin(goal[2] * 0.5)
        self.goal_pose.pose.orientation.w = math.cos(goal[2] * 0.5)
        self.robot_pose.header.stamp = self.goal_pose.header.stamp
        path = self.srv_compute_path(self.robot_pose, self.goal_pose, 0.1).plan
        if len(path.poses) == 0:
            print('Planer returned empty path. Retrying...')
            path = self.srv_compute_path(self.robot_pose, self.goal_pose, 0.1).plan
            if len(path.poses) == 0:
                print('Planner still returned empty path. Returning -1')
                return -1

        length     = 0.0
        last_point = self.robot_pose.pose.position
        for pose_stamped in path.poses:
            length += math.sqrt((pose_stamped.pose.position.x - last_point.x)**2 +
                                (pose_stamped.pose.position.y - last_point.y)**2)
            last_point = pose_stamped.pose.position

        return length

    def srv_generate_view_poses(self, req):
        res = GetViewPosesResponseMsg()

        self.robot_pose.pose = req.robot_pose
        try:
            pose_array_msg = PoseArrayMsg()
            pose_array_msg.header.frame_id = 'map'
            init_array_msg = PoseArrayMsg()
            init_array_msg.header.frame_id = 'map'
            if self.visualizer is not None:
                self.visualizer.begin_draw_cycle('final_states', 'opt_traj')
            for obj in req.objects:
                view_list = ViewPoseListMsg()
                self.gi.set_observation_distance(obj.min_observation_distance,
                                                 obj.max_observation_distance)
                view_poses = []
                opt_trajectories = []

                for gmm_msg in obj.object_pose_gmm:
                    gc = GaussianComponent(obj.id, 
                                           gmm_msg.id, 
                                           [gmm_msg.cov_pose.pose.position.x, 
                                            gmm_msg.cov_pose.pose.position.y, 
                                            gmm_msg.cov_pose.pose.position.z], 
                                            gmm_msg.cov_pose.covariance, 0)

                    self.gi.set_gaussian_component(gc, 0.3)
                    result = self.gi.get_view_poses(self.n_iterations, self.integration_step, self.n_samples, opt_trajectories, equilibrium=self.equilibrium)
                    view_poses.extend(zip([gmm_msg.id] * len(result), result, opt_trajectories))

                for x, (gmm_id, (rating, pose, js, nav_pose), traj) in enumerate(sorted(view_poses)):
                    msg = ViewPoseMsg()
                    msg.obj_id      = obj.id
                    msg.gaussian_id = gmm_id
                    msg.nav_distance = self.compute_path_length(nav_pose)
                    if msg.nav_distance == -1:
                        continue
                    msg.pose.position.x = pose[0,3]
                    msg.pose.position.y = pose[1,3]
                    msg.pose.position.z = pose[2,3]
                    qx, qy, qz, qw  = real_quat_from_matrix(pose)
                    msg.pose.orientation.x = qx
                    msg.pose.orientation.y = qy
                    msg.pose.orientation.z = qz
                    msg.pose.orientation.w = qw
                    view_list.views.append(msg)
                    pose_array_msg.poses.append(msg.pose)

                    if self.robot_subworld is not None:
                        traj.update({s: [0.0] * len(traj.values()[0]) for s in self.robot_subworld.free_symbols.difference(traj.keys())})
                        for x in range(len(traj.values()[0])):
                            self.robot_subworld.update_world({s: v[x] for s, v in traj.items()})
                            self.visualizer.draw_world('opt_traj', self.robot_subworld)
                    #print('Final js:\n  {}'.format('\n  '.join(['{}: {}'.format(k, v) for k, v in js.items()])))
                    msg.joint_state.name, msg.joint_state.position = zip(*js.items())
                    msg.base_position.linear.x  = nav_pose[0]
                    msg.base_position.linear.y  = nav_pose[1]
                    msg.base_position.angular.z = nav_pose[2]
                res.views.append(view_list)

            if self.visualizer is not None:
                self.visualizer.render('final_states', 'opt_traj')
            pose_array_msg.header.stamp = rospy.Time.now()
            self.pub_pose_debug.publish(pose_array_msg)
        except Exception as e:
            traceback.print_exc()
            print(e)
        return res


