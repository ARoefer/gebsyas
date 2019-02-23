#!/usr/bin/env python
import rospy
import sys

from sensor_msgs.msg import JointState as JointStateMsg

def print_js(msg):
    print('---\n{:>30}  {:>15} {:>15} {:>15}\n{}'.format('Joint', 'Position', 'Velocity', 'Effort',
        '\n'.join(['{:>30}: {: 15.5f} {: 15.5f} {: 15.5f}'.format(msg.name[x], msg.position[x], msg.velocity[x], msg.effort[x]) for x in range(len(msg.name))])))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Provide a joint states topic to subscribe to.')
        exit(0)

    rospy.init_node('pretty_joint_printer')

    sub = rospy.Subscriber(sys.argv[1], JointStateMsg, print_js, queue_size=1)

    while not rospy.is_shutdown():
        pass

    sub.unregister()