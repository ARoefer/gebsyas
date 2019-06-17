#!/usr/bin/env python
import math
import os
import rospy
import subprocess
import tf

from fetch_giskard.fetch_robot import Fetch

from gebsyas.test.demo_builder import cube, cylinder, sphere, compartment
from gebsyas.kinematics.kinematic_rule import *
from gebsyas.utils import bb, visualize_obj, res_pkg_path
from gebsyas.core.subs_ds import ks_from_obj
from gebsyas.ros_visualizer import *
from gebsyas.plotting import draw_recorders, ValueRecorder, SymbolicRecorder


from giskardpy.symengine_controller import position_conv
from giskardpy.qp_problem_builder   import QProblemBuilder, JointConstraint, SoftConstraint, HardConstraint

from sensor_msgs.msg import JointState as JointStateMsg
#from iai_bullet_sim.basic_simulator import BasicSimulator

def pos_goal(goal, current, weight=1, gain=1, ns=''):
    d = norm(goal - current)
    return {'align {} position'.format(ns): SoftConstraint(-d * gain, -d * gain, weight, d)}

class Scenario(object):
    def __init__(self, visualizer):
        self.vis = visualizer
        self.state = {}

    def update(self, dt):
        raise NotImplemented


class CubesExample(Scenario):
    def __init__(self, visualizer):
        super(CubesExample, self).__init__(visualizer)

        self.co  = cube(('co',),  frame3_rpy(0,0,0, point3(-1,-1,-0.5)), 0.2, 0.2, 0.2, state=self.state)
        self.uco = cube(('uco',), frame3_rpy(0,0,0, point3(-1,-1,-0.5)), 0.2, 0.2, 0.2, state=self.state)
        self.do  = cube(('do',),  frame3_rpy(0,0,0, point3(-1,-1,-0.8)), 0.2, 0.2, 0.2, state=self.state)

        self.str_sym_map = {str(s): s for s in self.state.keys()}

        self.pos_joint_symbols = [s for s in list(self.co.free_symbols) 
                                           + list(self.uco.free_symbols) 
                                           + list(self.do.free_symbols) 
                                           if str(s)[-3:]]# in {'__x', '__y', '__z'}]

        joint_constraints = {str(s): JointConstraint(-0.5, 0.5, 0.001) for s in self.pos_joint_symbols}

        print('\n'.join(['{:>15}: {} {} @ {}'.format(k, v.lower, v.upper, v.weight) for k, v in joint_constraints.items()]))

        self.goal     = point3(0,0,1)
        self.axis     = vector3(1,1,0.5)
        self.support  = point3(0,0,0)
        self.offset   = vector3(0,0,-0.3)
        self.hook_pos = self.co.pose * point3(*[0.1]*3)


        soft_constraints =      pos_goal(self.goal, pos_of(self.co.pose),  ns='co')
        soft_constraints.update(pos_goal(self.goal, pos_of(self.uco.pose), ns='uco'))

        print('\n'.join(['{}\n  l: {}\n  u: {}\n  e: {}\n'.format(k, s.lower, s.upper, s.expression) for k, s in soft_constraints.items()]))

        kinematic_hc = {str('kc_{}'.format(x)): c for x, c in enumerate(
                prismatic_joint(pos_of(self.co.pose), self.support, self.axis) + 
                fixed_position(pos_of(self.do.pose),  pos_of(self.co.pose) + self.offset) +
                leash_joint(self.hook_pos , pos_of(self.uco.pose), 0.5))}

        self.qp = QProblemBuilder(joint_constraints, kinematic_hc, soft_constraints, self.pos_joint_symbols)


    def update(self, dt):
        self.vis.begin_draw_cycle('static', 'objects')
        self.vis.draw_sphere('static', self.goal, 0.02)
        self.vis.draw_vector('static', self.support - self.axis, 2*self.axis, r=0, g=0.6)

        #print('State:\n  {}'.format('\n  '.join(['{:>15}: {:>20}'.format(s, self.state[s]) for s in self.pos_joint_symbols])))

        str_state    = {str(s): v for s, v in self.state.items()}
        #missing_vars = [x for x in self.qp.cython_big_ass_M.str_params if x not in str_state]
        #print('Missing vars: {}'.format(missing_vars))
        cmd = self.qp.get_cmd(str_state)

        #print('Command:\n  {}'.format('\n  '.join(['{:>15}: {:>20}'.format(j, v) for j, v in cmd.items()])))

        for k, v in cmd.items():
            self.state[self.str_sym_map[k]] = self.state[self.str_sym_map[k]] + v * dt
        
        co = self.co.subs(self.state)
        print('{}\n{}'.format('{:>20} '*3, '{:>20} '*3).format('||CO_x||', 
                                                               '||CO_y||', 
                                                               '||CO_z||', 
                                                               norm(x_of(co.pose)), 
                                                               norm(y_of(co.pose)), 
                                                               norm(z_of(co.pose))))

        visualize_obj(co,  self.vis, co.pose,  'objects', (  0,   0,   1, 1))
        visualize_obj(self.uco.subs(self.state), self.vis, self.uco.pose.subs(self.state), 'objects', (  0,   1, 0.3, 1))
        visualize_obj(self.do.subs(self.state),  self.vis, self.do.pose.subs(self.state),  'objects', (0.9, 0.7,   0, 1))

        self.vis.draw_sphere('objects', self.hook_pos.subs(self.state), 0.02)

        self.vis.render()


class DoorExample(Scenario):
    def __init__(self, visualizer):
        super(DoorExample, self).__init__(visualizer)    

        self.door        = cube(('door',),        rotation3_rpy(0,0,0), 0.02, 0.3, 0.4, state=self.state)
        self.handle      = cylinder(('handle',),  self.door.pose * translation3(-0.03,-0.12,-0.08), 0.01, 0.1, state=self.state)
        self.manipulator = cube(('manipulator',), rotation3_rpy(0,0,0), 0.2, 0.2, 0.2, state=self.state)

        self.str_sym_map = {str(s): s for s in self.state.keys()}

        self.pos_joint_symbols = [s for s in list(self.door.free_symbols) 
                                           + list(self.handle.free_symbols) 
                                           + list(self.manipulator.free_symbols) 
                                           if str(s)[-3:]]# in {'__x', '__y', '__z'}]

        joint_constraints = {str(s): JointConstraint(-0.5, 0.5, 0.001) for s in self.pos_joint_symbols}

        print('\n'.join(['{:>15}: {} {} @ {}'.format(k, v.lower, v.upper, v.weight) for k, v in joint_constraints.items()]))

        self.goal = dot(unitX, self.door.pose * unitY)

        soft_constraints = {'orientation_goal': SoftConstraint(1 - self.goal, 1 - self.goal, 1, self.goal)}

        print('\n'.join(['{}\n  l: {}\n  u: {}\n  e: {}\n'.format(k, s.lower, s.upper, s.expression) for k, s in soft_constraints.items()]))

        kinematic_hc = {str('kc_{}'.format(x)): c for x, c in enumerate(fixed_position(pos_of(self.manipulator.pose),  pos_of(self.handle.pose)))}

        self.qp = QProblemBuilder(joint_constraints, kinematic_hc, soft_constraints, self.pos_joint_symbols)

    def update(self, dt):
        self.vis.begin_draw_cycle('static', 'objects')
        self.vis.draw_vector('static', point3(0,0,0), unitX, r=0, g=0.6)

        print('State:\n  {}'.format('\n  '.join(['{:>25}: {:<20}'.format(s, self.state[s]) for s in self.pos_joint_symbols])))

        cmd = self.qp.get_cmd({str(s): v for s, v in self.state.items()})

        #print('Command:\n  {}'.format('\n  '.join(['{:>15}: {:>20}'.format(j, v) for j, v in cmd.items()])))

        for k, v in cmd.items():
            self.state[self.str_sym_map[k]] = self.state[self.str_sym_map[k]] + v * dt
        
        visualize_obj(self.door.subs(self.state), self.vis, self.door.pose.subs(self.state), 'objects', (0, 0, 1, 1))
        visualize_obj(self.handle.subs(self.state), self.vis, self.handle.pose.subs(self.state), 'objects', (0, 1, 0.3, 1))
        visualize_obj(self.manipulator.subs(self.state), self.vis, self.manipulator.pose.subs(self.state), 'objects', (0.9, 0.7, 0, 1))

        self.vis.draw_sphere('objects', self.hook_pos.subs(self.state), 0.02)

        self.vis.render()


class RobotOpensThings(Scenario):
    def __init__(self, visualizer):
        super(RobotOpensThings, self).__init__(visualizer)

        res_path   = res_pkg_path('package://fetch_giskard/robots/fetch.urdf')
        self.robot = Fetch(res_path)
        self.eef   = self.robot.eef
        self._setup_robot_rendering(res_path)

        print('\n'.join(['{:>20}: {:>10} {:>10}'.format(jn, j.lower, j.upper) for jn, j in self.robot._joints.items()]))

        # Setting up a drawer
        #self.ds = compartment('drawer', frame3_rpy(0,0, 0, point3(1.2, 0.1, 0.7)), 
        #                      0.6, 0.5, 0.2, 0.2, ('center', 'center'), 'horizontal')
        self.ds = compartment('drawer', frame3_rpy(0,0, 0, point3(1.2, 0.1, 0.7)), 
                              0.6, 0.5, 0.8, 0.2, ('right', 'center'), 'horizontal')

        self.handle = self.ds['drawer_handle'].subs(self.ds.value_table)
        self.panel  = self.ds['drawer_door'].subs(self.ds.value_table)
        self.comp   = self.ds['drawer_compartment'].subs(self.ds.value_table)

        self.s_drawer   =  Symbol('j_drawer')
        self.state[self.s_drawer] = 0.0
        self.str_sym_map = {str(s): s for s in self.state.keys()}

        #self.panel.pose  = self.panel.pose * translation3(self.s_drawer, 0, 0)
        hinge_pose       = self.comp.pose * translation3(self.comp.length * -0.5, self.comp.width * 0.5, 0)
        self.panel.pose  = hinge_pose * rotation3_axis_angle(unitZ, self.s_drawer) * inverse_frame(hinge_pose) * self.panel.pose
        self.handle.pose = self.panel.pose * (inverse_frame(self.panel.pose.subs(self.state)) * self.handle.pose)

        #print('Panel pose:\n{}\nHandle pose:\n{}'.format(self.panel.pose, self.handle.pose))

        # Setting up constraints
        x_align = dot(x_of(self.eef), x_of(self.handle.pose))
        y_align = fake_Abs(dot(y_of(self.eef), z_of(self.handle.pose)))
        dist    = norm(pos_of(self.eef) - pos_of(self.handle.pose))
        alignment = 0.02 - dist + x_align + y_align - 2
        in_handle_contact = sigmoid(alignment)

        open_measure = fake_Abs(dot(pos_of(self.panel.pose) - pos_of(self.comp.pose), x_of(self.comp.pose))) - self.comp.length * 0.5
        p_gain = 2
        soft_constraints = {'sc_d-a': SoftConstraint(-dist, -dist, 1, dist),
                            'sc_x-a': SoftConstraint(p_gain * (1 - x_align), p_gain * (1 - x_align), 1, x_align),
                            'sc_y-a': SoftConstraint(p_gain * (1 - y_align), p_gain * (1 - y_align), 1, y_align),
                            'sc_open': SoftConstraint(in_handle_contact * (0.3 - open_measure), 
                                  in_handle_contact * (0.6 - open_measure), 1000, open_measure)}

        # Setting up the Q problem
        self.s_bl = self.robot.joint_states_input.joint_map['base_linear_joint']
        self.s_ba = self.robot.joint_states_input.joint_map['base_angular_joint']
        self.s_la = self.robot.joint_states_input.joint_map['localization_z_ang']
        self.s_lx = self.robot.joint_states_input.joint_map['localization_x']
        self.s_ly = self.robot.joint_states_input.joint_map['localization_y']
        self.s_lz = self.robot.joint_states_input.joint_map['localization_z']
        self.s_wr = self.robot.joint_states_input.joint_map['wrist_roll_joint']
        uncontrolled_symbols = {self.s_la, self.s_lx, self.s_ly, self.s_lz, self.s_bl, self.s_ba}

        controlled_symbols   = list(self.eef.free_symbols.union({self.s_drawer}).difference(uncontrolled_symbols))
        #print('Controlled symbols: {}'.format(controlled_symbols))

        joint_constraints = {str(self.s_drawer): JointConstraint(0, 0.5, 0.001)}
        joint_constraints.update({j: c for j, c in self.robot.joint_constraints.items() if self.robot.joint_states_input.joint_map[j] in controlled_symbols})
        hard_constraints = {j: c for j, c in self.robot.hard_constraints.items() if self.robot.joint_states_input.joint_map[j] in controlled_symbols}

        #print('\n'.join(['{}: {}'.format(k, v) for k, v in self.state.items()]))

        self.qp = QProblemBuilder(joint_constraints, hard_constraints, soft_constraints, controlled_symbols)
        
        # Data recorders
        self.r_controls = SymbolicRecorder('Control Terms', x_align=x_align, y_align=y_align, 
                                           dist=dist, alignment=alignment, ihc=in_handle_contact, y_align_diff=y_align.diff(self.s_wr))
        self.r_commands = ValueRecorder('Commands', *[str(x) for x in controlled_symbols])


    def update(self, dt):
        self.vis.begin_draw_cycle('static', 'objects')

        now = rospy.Time.now()

        self.r_controls.log_symbols(self.state)
        cmd = self.qp.get_cmd({str(s): v for s, v in self.state.items()})
        #cmd = {}

        #print('Command:\n  {}'.format('\n  '.join(['{:>15}: {:>20}'.format(j, v) for j, v in cmd.items()])))

        for k, v in cmd.items():
            s = self.str_sym_map[k]
            if s == self.s_bl:
                self.state[self.s_lx] += cos(self.state[self.s_la]) * v * dt
                self.state[self.s_ly] += sin(self.state[self.s_la]) * v * dt
            elif s == self.s_ba:
                #self.r_commands.log_data(k, v)
                self.state[self.s_la] += v * dt
            elif s == self.s_lx or s == self.s_ly:
                print('WTF')
            else:
                if s == self.s_wr:
                    self.r_commands.log_data(k, v)
                self.state[s] += v * dt

        #print('\n'.join(['{:>20}: {}'.format(k, v) for k, v in cmd.items()]))


        self.tf_broadcaster.sendTransform((self.state[self.s_lx], self.state[self.s_ly], 0), 
                                          tf.transformations.quaternion_from_euler(0, 0, self.state[self.s_la]), 
                                          now, self._base_frame_name, 'map')
        self._js_msg.header.stamp = now
        self._js_msg.position = [self.state[s] for s in self._joint_symbol_cache]
        self.pub_js.publish(self._js_msg)

        visualize_obj(self.comp,   self.vis, self.comp.pose,  'static')
        visualize_obj(self.panel,  self.vis, self.panel.pose.subs(self.state), 'objects')
        visualize_obj(self.handle, self.vis, self.handle.pose.subs(self.state),  'objects')

        self.vis.render()


    def kill(self):
        self.sp_p.terminate()
        self.sp_p.wait()
        print('Stopped publisher process')


    def _setup_robot_rendering(self, urdf_path):
        publisher_path = '/opt/ros/{}/lib/robot_state_publisher/robot_state_publisher'.format(os.environ['ROS_DISTRO'])
        param_path = '/{}/robot_description'.format(self.robot.get_name())
        rospy.set_param(param_path, open(urdf_path, 'r').read())

        self.sp_p   = subprocess.Popen([publisher_path,
            '__name:={}_state_publisher'.format(self.robot.get_name()),
            'robot_description:={}'.format(param_path),
            '_tf_prefix:={}'.format(self.robot.get_name()),
            'joint_states:=/{}/joint_states'.format(self.robot.get_name())])

        self.tf_broadcaster   = tf.TransformBroadcaster()
        self._base_frame_name = '{}/{}'.format(self.robot.get_name(), self.robot._urdf_robot.get_root())
        print('Base frame: {}'.format(self._base_frame_name))
        self.pub_js = rospy.Publisher('/{}/joint_states'.format(self.robot.get_name()), JointStateMsg, queue_size=1)
        print(self.robot.get_joint_names())
        self._joint_symbol_cache = [self.robot.get_joint_symbol_map().joint_map[jn] for jn in self.robot.get_joint_names()]
        self.state.update({s: 0.0 for s in self.robot.get_joint_symbol_map().joint_map.values()})
        self.state[self.robot.joint_states_input.joint_map['torso_lift_joint']]    = 0.3
        self.state[self.robot.joint_states_input.joint_map['shoulder_pan_joint']]  = math.pi * 0.45
        self.state[self.robot.joint_states_input.joint_map['shoulder_lift_joint']] = math.pi * 0.4
        self.state[self.robot.joint_states_input.joint_map['elbow_flex_joint']]    = math.pi * 0.6
        self.state[self.robot.joint_states_input.joint_map['forearm_roll_joint']]  = math.pi * -0.5
        self.state[self.robot.joint_states_input.joint_map['wrist_flex_joint']]    = math.pi * 0.5
        self.state[self.robot.joint_states_input.joint_map['gripper_joint']]       = 0
        self.state[self.robot.joint_states_input.joint_map['l_gripper_finger_joint']] = 0
        self.state[self.robot.joint_states_input.joint_map['r_gripper_finger_joint']] = 0
        self._js_msg = JointStateMsg()
        self._js_msg.name     = self.robot.get_joint_names()
        self._js_msg.position = [0.0]*len(self._js_msg.name)
        self._js_msg.velocity = [0.0]*len(self._js_msg.name)
        self._js_msg.effort   = [0.0]*len(self._js_msg.name)



if __name__ == '__main__':
    rospy.init_node('kso_test')

    vis = ROSVisualizer('kso_visuals', 'map')

    scenario = RobotOpensThings(vis)

    secs = rospy.get_time()    
    dt   = 0.0
    #for x in range(20):
    while not rospy.is_shutdown():

        scenario.update(dt)
        
        n_secs = rospy.get_time()
        dt     = n_secs - secs
        secs   = n_secs


    fig = draw_recorders([scenario.r_controls, scenario.r_commands], plot_width=8, plot_height=4)
    fig.savefig('execution_robot_does_things.png')

    #sim = BasicSimulator()
