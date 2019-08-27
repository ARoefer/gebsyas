#!/usr/bin/env python
import rospy
import numpy as np

from kineverse.gradients.gradient_math import spw, point3

from geometry_msgs.msg    import Pose             as PoseMsg, \
                                 PoseStamped      as PoseStampedMsg
from gop_gebsyas_msgs.msg import SearchObject     as SearchObjectMsg, \
                                 SearchObjectList as SearchObjectListMsg, \
                                 ObjectPoseGaussianComponent as OPGCMsg

from gop_gebsyas_msgs.srv import GetViewPoses    as GetViewPosesSrv
from nav_msgs.srv         import GetPlan         as GetPlanSrv, \
                                 GetPlanResponse as GetPlanResponseMsg

def check_missing(d, *args):
    return [a for a in args if a not in d]


def convert_to_obj_msgs(object_data_list):
    out = []
    for x, d in enumerate(object_data_list):
        missing = check_missing(d, 'name', 'min_observation_distance', 'max_observation_distance')
        if len(missing) > 0:
            raise Exception('Missing attributes ({}) in object at index {}'.format(', '.join(missing), x))
        msg = SearchObjectMsg()
        msg.name = d['name']
        msg.id   = x
        msg.min_observation_distance = d['min_observation_distance']
        msg.max_observation_distance = d['max_observation_distance']
        out.append(msg)
    return out

def srv_fake_nav_plan(req):
    res = GetPlanResponseMsg()
    res.plan.header.stamp    = rospy.Time.now()
    res.plan.header.frame_id = 'map'
    res.plan.poses = [req.start, req.goal]
    return res


if __name__ == '__main__':
    rospy.init_node('mock_search_object_generator')

    view_pose_srv_name = '/get_view_poses'

    pub_debug = rospy.Publisher('/generated_objects', SearchObjectListMsg, queue_size=1, tcp_nodelay=True)
    srv_generate_view_poses = rospy.ServiceProxy(view_pose_srv_name, GetViewPosesSrv)
    srv_get_fake_plan = rospy.Service('/move_base/make_plan', GetPlanSrv, srv_fake_nav_plan)

    if not rospy.has_param('object_data'):
        print('Please provide object data via the parameter server as "object_data". The data should be a list of dictionaries.')
        exit()

    if not rospy.has_param('location_data'):
        print('Please provide location data via the parameter server as "location_data". The data should be a list of lists with three floats each.')
        exit()


    object_msgs = convert_to_obj_msgs(rospy.get_param('object_data'))
    locations   = [np.array(x) for x in rospy.get_param('location_data')] 

    for obj_msg in object_msgs:
        location_threshold = max(0.1, min(1.0, np.random.normal(0.5, 0.2)))
        total_weight = 0.0
        for l in locations:
            if np.random.random() <= location_threshold:
                weight        = max(0.05, np.random.random())
                total_weight += weight
                median        = l + np.hstack((np.random.normal(0, 0.3, 2), [0]))
                covariance    = np.diag(np.hstack((np.random.normal(0, 0.2, 2), np.random.normal(0, 0.03), [0]*3)))

                opgc_msg = OPGCMsg()
                opgc_msg.weight = weight
                opgc_msg.id = len(obj_msg.object_pose_gmm)
                opgc_msg.cov_pose.pose.orientation.w = 1
                opgc_msg.cov_pose.pose.position.x = l[0]
                opgc_msg.cov_pose.pose.position.y = l[1]
                opgc_msg.cov_pose.pose.position.z = l[2]
                opgc_msg.cov_pose.covariance = covariance.flatten().tolist()
                obj_msg.object_pose_gmm.append(opgc_msg)

        # Normalize weights
        for opgc_msg in obj_msg.object_pose_gmm:
            opgc_msg.weight /= total_weight

    print('Generated test data for {} objects'.format(len(object_msgs)))

    objs_msg = SearchObjectListMsg()
    objs_msg.search_object_list = object_msgs

    rospy.sleep(0.6) # Sleep, because ROS is STUPID
    pub_debug.publish(objs_msg)

    # Sample a robot location
    angle  = (np.random.random() - 0.5) * np.pi
    radius = np.random.normal(6.0, 1.0)

    robot_pose = PoseMsg()
    robot_pose.position.x = np.cos(angle) * radius
    robot_pose.position.y = np.sin(angle) * radius
    # Make the robot face towards the center
    robot_pose.orientation.z = np.sin((angle - np.pi) * 0.5)
    robot_pose.orientation.w = np.cos((angle - np.pi) * 0.5)

    print('Waiting for service')
    rospy.wait_for_service(view_pose_srv_name)

    result = srv_generate_view_poses(robot_pose, object_msgs)
    print('Called service and got {} sets of views'.format(len(result.views)))

