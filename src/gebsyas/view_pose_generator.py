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

from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer


class ViewPoseGenerator(object):
    def __init__(self, km, camera, sym_loc_x, sym_loc_y, sym_loc_a, service_name, collision_link_paths=[]):
        self.km = km

        self.visualizer = ROSBPBVisualizer('debug_vis', 'map')

        self.gi = GaussianInspector(km, camera, sym_loc_x, sym_loc_y, sym_loc_a, self.compute_path_length, 0.2, collision_link_paths, self.visualizer) 

        self.service   = rospy.Service(service_name, GetViewPosesSrv, self.srv_generate_view_poses)
        self.srv_compute_path = rospy.ServiceProxy('/move_base/make_plan', GetPlanSrv)
        self.robot_pose       = PoseStampedMsg()
        self.robot_pose.header.frame_id = 'map'
        self.robot_pose.pose.orientation.w = 1

        self.goal_pose = PoseStampedMsg()
        self.goal_pose.header.frame_id = 'map'
        self.pub_pose_debug = rospy.Publisher('/debug_generated_view_poses', PoseArrayMsg, queue_size=1, tcp_nodelay=True)
        
        self.visualizer.begin_draw_cycle('world')
        self.visualizer.draw_world('world', self.km.kw)
        self.visualizer.render('world')

    # goal as x, y, a in map
    def compute_path_length(self, goal):
        self.goal_pose.header.stamp = rospy.Time.now()
        self.goal_pose.pose.position.x = goal[0]
        self.goal_pose.pose.position.y = goal[1]
        self.goal_pose.pose.orientation.z = math.sin(goal[2] * 0.5)
        self.goal_pose.pose.orientation.w = math.cos(goal[2] * 0.5)
        self.robot_pose.header.stamp = self.goal_pose.header.stamp
        path = self.srv_compute_path(self.robot_pose, self.goal_pose, 0.1).plan

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
            self.visualizer.begin_draw_cycle('final_states')
            for obj in req.objects:
                view_list = ViewPoseListMsg()
                self.gi.set_observation_distance(obj.min_observation_distance,
                                                 obj.max_observation_distance)
                for gmm_msg in obj.object_pose_gmm:
                    gc = GaussianComponent(obj.id, 
                                           gmm_msg.id, 
                                           [gmm_msg.cov_pose.pose.position.x, 
                                            gmm_msg.cov_pose.pose.position.y, 
                                            gmm_msg.cov_pose.pose.position.z], 
                                            gmm_msg.cov_pose.covariance, 0)

                    self.gi.set_gaussian_component(gc, 0.3)

                    lowest_rating = -1
                    for rating, pose, js, nav_pose in self.gi.get_view_poses(100, 0.25, 5):
                        msg = ViewPoseMsg()
                        msg.obj_id      = obj.id
                        msg.gaussian_id = gmm_msg.id
                        msg.pose.position.x = pose[0,3]
                        msg.pose.position.y = pose[1,3]
                        msg.pose.position.z = pose[2,3]
                        qx, qy, qz, qw  = real_quat_from_matrix(pose)
                        msg.pose.orientation.x = qx
                        msg.pose.orientation.y = qy
                        msg.pose.orientation.z = qz
                        msg.pose.orientation.w = qw
                        view_list.views.append(msg)
                        if lowest_rating == -1:
                            pose_array_msg.poses.append(msg.pose)
                        #    lowest_rating = rating
                        elif rating < lowest_rating:
                            pose_array_msg.poses[-1] = msg.pose
                        #print('Final js:\n  {}'.format('\n  '.join(['{}: {}'.format(k, v) for k, v in js.items()])))
                        msg.joint_state.name, msg.joint_state.position = zip(*js.items())
                        msg.base_position.linear.x  = nav_pose[0]
                        msg.base_position.linear.y  = nav_pose[1]
                        msg.base_position.angular.z = nav_pose[2]
                res.views.append(view_list)

            self.visualizer.render('final_states')
            pose_array_msg.header.stamp = rospy.Time.now()
            self.pub_pose_debug.publish(pose_array_msg)
        except Exception as e:
            traceback.print_exc()
            print(e)
        return res


