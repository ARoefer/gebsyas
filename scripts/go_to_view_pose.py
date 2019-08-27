#!/usr/bin/env python
import rospy
import actionlib
import math

from actionlib_msgs.msg   import GoalStatus as GoalStatusMsg
from sensor_msgs.msg      import JointState as JointStateMsg
from move_base_msgs.msg   import MoveBaseAction, MoveBaseGoal

from gop_gebsyas_msgs.srv import GoToViewPose         as GoToViewPoseSrv, \
                                 GoToViewPoseResponse as GoToViewPoseResponseMsg

nav_client    = None
pub_joint_cmd = None

def srv_go_to_pose(req):
    view = req.view_pose

    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = view.base_position.linear.x
    goal.target_pose.pose.position.y = view.base_position.linear.y
    goal.target_pose.pose.orientation.z = math.sin(0.5 * view.base_position.angular.z)
    goal.target_pose.pose.orientation.w = math.cos(0.5 * view.base_position.angular.z)
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp    = rospy.Time.now()

    nav_client.send_goal(goal)

    js_cmd = view.joint_state
    js_cmd.header.stamp = rospy.Time.now()
    pub_joint_cmd.publish(js_cmd)

    nav_client.wait_for_result()
    result = nav_client.get_result()

    res = GoToViewPoseResponseMsg()
    res.success = True #result.status.status == GoalStatusMsg.SUCCEEDED
    return res

if __name__ == '__main__':
    rospy.init_node('view_pose_observer')

    pub_joint_cmd = rospy.Publisher('/joint_position_controller/commands', JointStateMsg, queue_size=1, tcp_nodelay=True)
    nav_client  = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    nav_client.wait_for_server()

    srv_handler = rospy.Service('/go_to_view_pose', GoToViewPoseSrv, srv_go_to_pose)

    print('Ready to move the robot!')
    while not rospy.is_shutdown():
        try:
            rospy.sleep(1000)
        except:
            pass
