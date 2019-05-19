#!/usr/bin/env python
import rospy
import matplotlib.pyplot as plt
import numpy as np

from giskardpy.symengine_wrappers import *

from gebsyas.core.data_state import DataState
from gebsyas.kinematics.kinematic_state import KinematicState
from gebsyas.kinematics.kinematic_rule import *
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.utils import bb

from math import pi


def transform_dist(a, b):
    return pos_of(b) - pos_of(a), rotation_distance(rot_of(a), rot_of(b))

if __name__ == '__main__':
    rospy.init_node('debug_vis_data_state')
    vis = ROSVisualizer('kso_visuals', 'map')
    vis.begin_draw_cycle('lps')
    
    ks = KinematicState(DataState())

    # Whatever parameters
    axis_rot  = rotation3_rpy(0.0, 0.2, 0)
    true_axis = axis_rot * unitZ
    r_error   = rotation3_rpy(0.0, 0.1, 0.0)
    p_error   = vector3(0.01, 0.0, 0.02)
    d_error   = 1.0

    c_space = (-pi, pi)
    steps   = 20

    error   = np.zeros((4, steps + 1))

    obj_t = bb(id='obj', pose=translation3(1,0,0))
    obj_o = bb(id='obj', pose=translation3(1,0,0))
    obj_o2 = bb(id='obj', pose=translation3(1,0,0))

    hp_true = HingePair(obj_t, point3(0,0,0), true_axis)
    hp_obs  = ScrewPair(obj_o, p_error, r_error * true_axis, 0.2, 0, 0.5)
    
    hp_true.apply(ks)
    hp_obs.apply(ks)

    hp_true.render(ks, vis, 'lps', (0,0.7,0,1))
    hp_obs.render(ks, vis, 'lps', (0.7,0,0,1))

    r_poses = []
    o_poses = []

    for s in range(steps + 1):
        q = (c_space[1] - c_space[0]) / steps * s
        ks.data_state.value_table[hp_true.sym_alpha] = q
        real  = obj_t.pose.subs({hp_true.sym_alpha: q})
        obs   = obj_o.pose.subs({hp_obs.sym_alpha: q})
        r_poses.append(real)
        o_poses.append(obs)

        ep = hp_true.inlier_function(ks, obs)
        error[:, s] = (q, ep[0], ep[1], ep[2])

    vis.draw_poses('lps', sp.eye(4), 0.1, 0.01, r_poses, r=0.3, b=0.4)
    vis.draw_poses('lps', sp.eye(4), 0.1, 0.01, o_poses, g=0.3, b=0.4)
    vis.render()

    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(error[0], error[1], 'r', label='Planar')
    ax.plot(error[0], error[2], 'g', label='Radial')
    #ax.plot(error[0], error[3], 'b', label='Z')
    ax.plot(error[0], error[3], 'm', label='Angular')
    ax.legend(loc='upper left')
    ax.set_title('Metric Error')
    fig.tight_layout()
    fig.savefig('errorplot_{}_{}.png'.format(hp_true, hp_obs))

    rospy.sleep(0.3)
    
    #print(error)


