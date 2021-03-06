#!/usr/bin/env python
import sys
import rospy
from gebsyas.grasp_any_controller import GraspAnyController
from gebsyas.msg import ProbabilisticObject as POMsg
from gebsyas.msg import StringArray
from fetch_giskard.fetch_robot import Fetch
from sensor_msgs.msg import JointState
from pprint import pprint

def cmdDictToJointState(command):
    """Comfort function. Converts a dictionary {joint: velocity} to a ROS-JointState."""
    js = JointState()
    js.header.stamp = rospy.Time.now()
    for joint, vel in command.iteritems():
        js.name.append(str(joint))
        js.position.append(0)
        js.velocity.append(vel)
        js.effort.append(0)

    return js


class AffordanceCommandNode(object):
    """
    @brief      A simple ROS-node demonstrating grapsing based on giskard constraints. It's probably broken.
    """
    def __init__(self, fetch_urdf, cmd_topic='/simulator/commands'):
        self.objects = {}
        self.robot = Fetch(fetch_urdf)
        self.controller = None
        self.bInit = False
        self.object_sub = rospy.Subscriber('objects', POMsg, self.refresh_object, queue_size=10)
        self.grasp_any_sub = rospy.Subscriber('grasp_any', StringArray, self.grasp_any, queue_size=1)
        self.js_sub = rospy.Subscriber('/joint_states', JointState, self.js_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher(cmd_topic, JointState, queue_size=1)
        self.state_len = 0
        print('Init finished. Waiting for commands.')

    def refresh_object(self, object_msg):
        if object_msg.id not in self.objects:
            print("Received new object '{}'".format(object_msg.id))

        self.objects[object_msg.id] = object_msg
        if self.controller is not None:
            self.controller.update_object(object_msg)

    def grasp_any(self, command):
        print('Received command to grasp any of {}'.format(command.strings))
        self.bInit = False
        candidates = []
        for name in command.strings:
            if name in self.objects:
                candidates.append(self.objects[name])

        print('Assembled candidates. Moving on to constructing controller...')
        start = rospy.Time.now()
        self.controller = GraspAnyController(self.robot, [self.robot.gripper], candidates)
        print('Construction finished in {:4f} seconds.'.format((rospy.Time.now() - start).to_sec()))


    def js_callback(self, joint_state):
        self.robot.set_joint_state(joint_state)
        if self.controller is None:
            return

        if not self.bInit:
            print('Compiling controller...')
            start = rospy.Time.now()
        else:
            start = rospy.Time.now()

        command = self.controller.get_next_command()
        #print('\b' * self.state_len)
        #self.controller.qp_problem_builder.print_jacobian(self.bInit)
        state_str = '\n'.join(['{}: {}'.format(j, v) for j, v in command.items()])
        self.state_len = len(state_str)
        #print(state_str)

        if not self.bInit:
            print('Controller compiled in {:4f} seconds'.format((rospy.Time.now() - start).to_sec()))
            self.bInit = True
        else:
            print('Controller update took {:4f} seconds'.format((rospy.Time.now() - start).to_sec()))

        cmdMsg = cmdDictToJointState(command)
        self.cmd_pub.publish(cmdMsg)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: <FETCH URDF PATH> [<COMMAND TOPIC>]')
        exit(0)

    rospy.init_node('grasp_any_node')

    if len(sys.argv) == 2:
        Instance = AffordanceCommandNode(sys.argv[1])
    else:
        Instance = AffordanceCommandNode(sys.argv[1], sys.argv[2])

    rospy.spin()