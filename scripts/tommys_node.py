#!/usr/bin/env python
import rospy
import sys
from geometry_msgs.msg import Point as PointMsg
from gebsyas.utils import symbol_formatter, res_pkg_path, cmdDictToJointState
from giskardpy import print_wrapper
from giskardpy.symengine_controller import *
from giskardpy.symengine_wrappers import *
from giskardpy.input_system import Point3Input
from giskardpy.qp_problem_builder import SoftConstraint as SC
from fetch_giskard.fetch_robot import Fetch
from navigation_msgs.msg import RobotState as RobotStateMsg 

from sensor_msgs.msg import JointState as JointStateMsg

class ControllerWrapper(SymEngineController):
    def __init__(self, robot, print_fn=print_wrapper):
        self.path_to_functions = res_pkg_path('package://gebsyas/.controllers/')
        self.controlled_joints = []
        self.hard_constraints = {}
        self.joint_constraints = {}
        self.qp_problem_builder = None
        self.robot = robot
        self.current_subs = {}
        self.print_fn = print_fn


class HeadPointer(object):
    """docstring for HeadPointer"""
    def __init__(self, urdf_path='Fetch.urdf'):
        super(HeadPointer, self).__init__()

        self.robot = Fetch(urdf_path)
        self.controller = ControllerWrapper(self.robot)
        self.controller.set_controlled_joints(['head_tilt_joint', 'head_pan_joint'])

        self.point_input = Point3Input.prefix(symbol_formatter, 'point')

        self.subs = {s: 0.0 for j, s in self.robot.joint_states_input.joint_map.items()}
        self.subs.update({s: 0.0 for s in self.point_input.get_expression().free_symbols})

        goal_dir   = self.point_input.get_expression() - pos_of(self.robot.camera.pose)
        gaze_angle = acos(dot(x_of(self.robot.camera.pose), goal_dir) / norm(goal_dir))

        self.controller.init({'gaze_control': SC(-gaze_angle, -gaze_angle, 1, gaze_angle)}, self.subs.keys(), print_wrapper)

        self.pub_cmd = rospy.Publisher('/velocity_controller/joint_velocity_controller/joint_velocity_commands', JointStateMsg, queue_size=1, tcp_nodelay=True)
        self.sub_localization = rospy.Subscriber('/vulcan_robot_state', RobotStateMsg, self.process_localization, queue_size=1)
        self.sub_point = rospy.Subscriber('/give_me_point', PointMsg, self.process_poi, queue_size=1)
        self.sub_js    = rospy.Subscriber('/joint_states', JointStateMsg, self.process_js, queue_size=1)


    def process_js(self, msg):
        for x in range(len(msg.name)):
            if msg.name[x] in self.robot.joint_states_input.joint_map:
                self.subs[self.robot.joint_states_input.joint_map[msg.name[x]]] = msg.position[x]

        cmd = self.controller.get_cmd({str(s): p for s, p in self.subs.items()}, None)
        self.pub_cmd.publish(cmdDictToJointState(cmd))
        print('\n'.join(['{}: {}'.format(k, v) for k, v in self.subs.items() if 'joint' not in str(k)]))


    def process_localization(self, msg):
        self.subs[self.robot.joint_states_input.joint_map['localization_x']] = msg.pose.x
        self.subs[self.robot.joint_states_input.joint_map['localization_y']] = msg.pose.y
        self.subs[self.robot.joint_states_input.joint_map['localization_z_ang']] = msg.pose.theta

    def process_poi(self, msg):
        self.subs[self.point_input.x] = msg.x
        self.subs[self.point_input.y] = msg.y
        self.subs[self.point_input.z] = msg.z


if __name__ == '__main__':
    rospy.init_node('tommys_head_pointer')

    hp = HeadPointer(sys.argv[1])

    while not rospy.is_shutdown():
        pass