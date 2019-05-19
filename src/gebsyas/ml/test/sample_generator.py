import sys
import math

import numpy as np
import numpy.random as rnd

from random import random
from gebsyas.core.dl_types   import DLPhysicalThing
from gebsyas.core.data_state import DataState
from gebsyas.utils           import visualize_obj, yaml
from gebsyas.ros_visualizer  import ROSVisualizer
from gebsyas.test.demo_builder import compartment, cube, cylinder
from giskardpy.symengine_wrappers import frame3_rpy, point3

def n_dist(a, b):
    o = rnd.normal(a, b)
    if hasattr(o, 'item'):
        return o.item()
    return o

def choice(o, p):
    o = rnd.choice(o, p=p)
    if hasattr(o, 'item'):
        return o.item()
    return o    


def generate_shelf_samples(n_samples=20, vis=None):
    K = ['none', 'prismatic', 'hinge']

    if vis:
        vis.begin_draw_cycle('scene')
    
    configs = []
    h_fac = {'left': 1, 'center': 0, 'right': -1}
    v_fac = {'bottom': -1, 'center': 0, 'top': 1}
    o_fac = {'horizontal': 0, 'vertical': 1, 'knob': 0}

    x_acc = 0
    y_acc = 0
    r_height = 0

    min_handle_width = 0.08

    out = []

    r_length = int(np.sqrt(n_samples))

    for x in range(n_samples):
        ds = DataState()

        if x % r_length == 0:
            y_acc   += r_height + 0.05
            r_height = 0
            x_acc    = 0

        k = rnd.choice(K, p=[0.4, 0.3, 0.3])
        #k = rnd.choice(K, p=[0.5, 0.5])
        s_out = {'k': k}

        if k == 'none':
            t_type   = choice(['countertop', 'table', 'stove'], [1.0/3.0]*3)
            t_width  = n_dist(1, 0.4)
            t_depth  = n_dist(0.6, 0.15)
            t_height = n_dist(0.04, 0.02)
            t_pose   = frame3_rpy(0,0, math.pi * 2 * random(), point3(0, x_acc - 0.5 * t_width, y_acc))

            table = cube((t_type,), t_pose, t_depth, t_width, t_height, (1,1,1,1), ds.value_table, sem_type=t_type, geom_type='cube')

            c_type   = choice(['cup', 'mug', 'bottle', 'thing'], [0.25]*4)
            c_radius = n_dist(0.05, 0.02)
            c_height = n_dist(0.15, 0.08)
            f_laying = choice([1, 0], [0.1, 0.9]) 
            c_pose   = t_pose * frame3_rpy(0, f_laying * math.pi * 0.5, math.pi * 2 * random(), 
                                           point3((0.5 - random()) * t_depth, 
                                                  (0.5 - random()) * t_width,
                                                   0.5 * t_height + f_laying * c_radius + (1 - f_laying) * 0.5 * c_height))
            cup  = cylinder((c_type,), c_pose, c_radius, c_height, (1,1,1,1), ds.value_table, sem_type=c_type, geom_type='cylinder')

            x_acc -= t_width + 0.05

            ds.insert_data(table, t_type)
            ds.insert_data(cup,   c_type)

            s_out['d']  = cup.subs(ds.value_table)
            s_out['s']  = table.subs(ds.value_table)

            if vis:
                for Id, o in ds.dl_data_iterator(DLPhysicalThing):
                    o = o.subs(ds.value_table)
                    visualize_obj(o, vis, o.pose, 'scene')

        elif k == 'prismatic':
            hl = choice(['left',   'center', 'right'],      p=[0.05, 0.9, 0.05])
            vl = choice(['bottom', 'center', 'top'],        p=[0.05, 0.8, 0.15])
            ol = choice(['horizontal', 'vertical', 'knob'], p=[0.7, 0.05, 0.25])

            hf = h_fac[hl]
            vf = v_fac[vl]
            of = o_fac[ol]

            p_width = n_dist(0.01, 0.02)

            c_width  = n_dist(0.2 + (1 - of) * 0.2, 0.1)
            c_height = n_dist(      0.2 + of * 0.2, 0.25 * (0.2 + of * 0.2))
            c_depth  = n_dist( 0.4, 0.2)
            margin   = n_dist(0.06, 0.04)
            handle_depth  = n_dist(0.04, 0.01)
            handle_height = choice([n_dist(0.12, 0.04), of * (c_height - margin) + (1 - of) * (c_width - margin)], p=[0.78, 0.22])

            compartment('a', frame3_rpy(0,0,0, point3(0, x_acc - c_width * 0.5, y_acc + c_height * 0.5)), 
                c_depth, c_width, c_height, handle_height, (hl,vl), ol, ds, handle_depth, handle_margin=margin)

            x_acc -= c_width + 0.05
            r_height = max(r_height, c_height)

            comp   = ds['a_compartment'].subs(ds.value_table)
            door   = ds['a_door'].subs(ds.value_table)
            handle = ds['a_handle'].subs(ds.value_table)

            s_out['d']  = door
            s_out['s']  = comp
            s_out['k0'] = 'static'
            s_out['c0'] = handle

            if vis:
                for Id, o in ds.dl_data_iterator(DLPhysicalThing):
                    o = o.subs(ds.value_table)
                    visualize_obj(o, vis, o.pose, 'scene')

        elif k == 'hinge':
            hl = choice(['left',   'center', 'right'], p=[0.475, 0.05, 0.475])
            vl = choice(['bottom', 'center', 'top'],   p=[0.4,  0.2, 0.4])
            ol = choice(['horizontal', 'vertical', 'knob'], p=[0.6, 0.2, 0.2])

            hf = h_fac[hl]
            vf = v_fac[vl]
            of = o_fac[ol]

            p_width = n_dist(0.01, 0.02)

            c_width  = n_dist(0.4, 0.1)
            c_height = n_dist(0.2 + of * 0.4, 0.25 * (0.2 + of * 0.2))
            c_depth  = n_dist(0.4, 0.2)
            margin   = n_dist(0.06, 0.04)
            handle_depth  = n_dist(0.04, 0.01)
            handle_height = choice([n_dist(0.12, 0.04), of * (c_height - margin) + (1 - of) * (c_width - margin)], p=[0.78, 0.22])

            compartment('a', frame3_rpy(0,0,0, point3(0, x_acc - c_width * 0.5, y_acc + c_height * 0.5)), 
                c_depth, c_width, c_height, handle_height, (hl,vl), ol, ds, handle_depth, handle_margin=margin)

            x_acc -= c_width + 0.05
            r_height = max(r_height, c_height)

            comp   = ds['a_compartment'].subs(ds.value_table)
            door   = ds['a_door'].subs(ds.value_table)
            handle = ds['a_handle'].subs(ds.value_table)

            s_out['d']  = door
            s_out['s']  = comp
            s_out['k0'] = 'static'
            s_out['c0'] = handle

            if vis:
                for Id, o in ds.dl_data_iterator(DLPhysicalThing):
                    o = o.subs(ds.value_table)
                    visualize_obj(o, vis, o.pose, 'scene')
        out.append(s_out)

    if vis:
        vis.render()

    return out
