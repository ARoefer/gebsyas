#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState as JointStateMsg
from geometry_msgs.msg import Twist as TwistMsg


class Converter(object):
    def __init__(self):
        self.pub = rospy.Publisher('/fetch/commands/joint_velocities', JointStateMsg, queue_size=1, tcp_nodelay=True)
        self.sub = rospy.Subscriber('/cmd_vel', TwistMsg, self.process_twist, queue_size=1)
        self.msg_template = JointStateMsg()
        self.msg_template.name.extend(['base_linear_joint', 'base_angular_joint'])
        self.msg_template.velocity.extend([0.0, 0.0])

    def process_twist(self, twist_msg):
        self.msg_template.header.stamp = rospy.Time.now()
        self.msg_template.velocity[0] = twist_msg.linear.x
        self.msg_template.velocity[1] = twist_msg.angular.z
        self.pub.publish(self.msg_template)

    def shutdown(self):
        self.sub.unregister()
        self.pub.unregister()


if __name__ == '__main__':
    rospy.init_node('cmd_vek_converter')

    node = Converter()

    while not rospy.is_shutdown():
        pass

    node.shutdown()