import rospy

from gebsyas.gaussian_observer import GaussianInspector, Camera, GaussianComponent
from gebsyas.utils             import real_quat_from_matrix

from gop_gebsyas_msgs.msg import SearchObjectList as SearchObjectListMsg, \
                                 ViewPose         as ViewPoseMsg, \
                                 ViewPoseList     as ViewPoseListMsg
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer


class ViewPoseGenerator(object):
    def __init__(self, km, camera, sym_loc_x, sym_loc_y, sym_loc_a, sub_topic, pub_topic):
        self.km = km

        self.gi = GaussianInspector(km, camera, sym_loc_x, sym_loc_y, sym_loc_a, 0.2, 0.8, 0.2, ROSBPBVisualizer('debug_vis', 'odom')) 

        self.pub_poses = rospy.Publisher(pub_topic, ViewPoseListMsg, queue_size=1)
        self.sub_objs  = rospy.Subscriber(sub_topic, SearchObjectListMsg, callback=self.cb_objects, queue_size=1)

    def cb_objects(self, object_list_msg):
        out = ViewPoseList()

        for obj in object_list_msg.search_object_list:
            for gmm_msg in obj.object_pose_gmm:
                gc = GaussianComponent(obj.id, 
                                       gmm_msg.id, 
                                       [gmm_msg.cov_pose.pose.position.x, 
                                        gmm_msg.cov_pose.pose.position.y, 
                                        gmm_msg.cov_pose.pose.position.z], 
                                       gmm_msg.cov_pose.covariance, 0)

                self.gi.set_gaussian_component(gc, 0.3)

                for rating, pose in self.gi.get_view_poses():
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
                    out.poses.append(msg)

        self.pub_poses.publish(out)


