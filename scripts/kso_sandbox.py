#!/usr/bin/env python
import rospy

from gebsyas.kinematics.kinematic_rule import *
from gebsyas.utils import bb, visualize_obj
from gebsyas.core.subs_ds import ks_from_obj
from gebsyas.ros_visualizer import *

from giskardpy.symengine_controller import position_conv
from giskardpy.qp_problem_builder   import QProblemBuilder, JointConstraint,SoftConstraint
#from iai_bullet_sim.basic_simulator import BasicSimulator

def pos_goal(goal, current, weight=1, gain=1, ns=''):
    d = norm(goal - current)
    return {'align {} position'.format(ns): SoftConstraint(-d * gain, -d * gain, weight, d)}

def cube(name, pose, length, width, height, state={}):
    return ks_from_obj(bb(pose=pose, 
                          length=length, 
                          width=width, 
                          height=height), name, state)

def sphere(name, pose, radius, state={}):
    return ks_from_obj(bb(pose=pose, radius=radius), name, state)

def cylinder(name, pose, radius, height, state={}):
    return ks_from_obj(bb(pose=pose, radius=radius, height=height), name, state)


class Scenario(object):
    def __init__(self, visualizer):
        self.vis = visualizer
        self.state = {}

    def update(self, dt):
        raise NotImplemented


class CubesExample(Scenario):
    def __init__(self, visualizer):
        super(CubesExample, self).__init__(visualizer)

        self.co  = cube(('co',),  frame3_rpy(0,0,0, point3(-1,-1,-0.5)), 0.2, 0.2, 0.2, self.state)
        self.uco = cube(('uco',), frame3_rpy(0,0,0, point3(-1,-1,-0.5)), 0.2, 0.2, 0.2, self.state)
        self.do  = cube(('do',),  frame3_rpy(0,0,0, point3(-1,-1,-0.8)), 0.2, 0.2, 0.2, self.state)

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

        cmd = self.qp.get_cmd({str(s): v for s, v in self.state.items()})

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

        self.door        = cube(('door',),        rotation3_rpy(0,0,0), 0.02, 0.3, 0.4, self.state)
        self.handle      = cylinder(('handle',),  self.door.pose * translation3(-0.03,-0.12,-0.08), 0.01, 0.1, self.state)
        self.manipulator = cube(('manipulator',), rotation3_rpy(0,0,0), 0.2, 0.2, 0.2, self.state)

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


if __name__ == '__main__':
    rospy.init_node('kso_test')

    vis = ROSVisualizer('kso_visuals', 'map')

    scenario = CubesExample(vis)

    secs = rospy.get_time()    
    dt   = 0.0
    while not rospy.is_shutdown():

        scenario.update(dt)
        
        n_secs = rospy.get_time()
        dt     = n_secs - secs
        secs   = n_secs
        


    #sim = BasicSimulator()
