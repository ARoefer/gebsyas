#!/usr/bin/env python
import rospy
from gebsyas.msg import LocalizedPoseStamped as LPSMsg
from navigation_msgs.msg import RobotState as RobotStateMsg


class Converter(object):
    def __init__(self):
        self.pub = rospy.Publisher('/fetch/localization', LPSMsg, queue_size=1, tcp_nodelay=True)
        self.sub = rospy.Subscriber('/vulcan_robot_state', RobotStateMsg, self.process_state, queue_size=1)

    def process_state(self, state_msg):
        msg = LPSMsg()
        msg.header.stamp = state_msg.header.stamp
        msg.pose.linear.x = state_msg.pose.x
        msg.pose.linear.y = state_msg.pose.y
        msg.pose.angular.z = state_msg.pose.theta
        msg.local_velocity.linear.x  = state_msg.linear_velocity.data
        msg.local_velocity.angular.z = state_msg.angular_velocity.data
        self.pub.publish(msg)

    def shutdown(self):
        self.sub.unregister()
        self.pub.unregister()


if __name__ == '__main__':
    rospy.init_node('localization_converter')

    node = Converter()

    while not rospy.is_shutdown():
        pass

    node.shutdown()