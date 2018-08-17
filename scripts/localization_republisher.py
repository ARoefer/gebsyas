#!/usr/bin/env python
import rospy
from gebsyas.msg import Pose2DStamped as Pose2DStampedMsg
from navigation_msgs.msg import RobotState as RobotStateMsg


class Converter(object):
    def __init__(self):
        self.pub = rospy.Publisher('/fetch/localization', Pose2DStampedMsg, queue_size=1, tcp_nodelay=True)
        self.sub = rospy.Subscriber('/vulcan_robot_state', RobotStateMsg, self.process_state, queue_size=1)

    def process_state(self, state_msg):
        msg = Pose2DStampedMsg()
        msg.header.stamp = state_msg.header.stamp
        msg.pose.x = state_msg.pose.x
        msg.pose.y = state_msg.pose.y
        msg.pose.theta = state_msg.pose.theta
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