import math

from gebsyas.core.data_state import DataState
from gebsyas.core.subs_ds import ks_from_obj
from gebsyas.utils import bb

from giskardpy.symengine_wrappers import translation3, rotation3_rpy, eye

def cube(name, pose, length, width, height, color=(1,1,1,1), state={}, **kwargs):
    return ks_from_obj(bb(pose=pose, 
                          length=length, 
                          width=width, 
                          height=height,
                          color=color, **kwargs), name, state)[0]

def sphere(name, pose, radius, color=(1,1,1,1), state={}, **kwargs):
    return ks_from_obj(bb(pose=pose, radius=radius, color=color, **kwargs), name, state)[0]

def cylinder(name, pose, radius, height, color=(1,1,1,1), state={}, **kwargs):
    return ks_from_obj(bb(pose=pose, radius=radius, height=height, color=color, **kwargs), name, state)[0]


def compartment(name, pose, length, width, height, handle_height, handle_location=('left', 'top'), handle_orientation='vertical', data_state=None, handle_depth=0.04, handle_width=0.01, panel_thickness=0.01, handle_margin=0.1):
    out = data_state if data_state != None else DataState()

    c_pose   = pose
    c_length = length
    c_width  = width
    c_height = height
    c_name   = '{}_compartment'.format(name)
    compartment = cube((c_name,), c_pose, 
                            c_length, c_width, c_height,
                            (0.3, 0.2, 0.2, 1), 
                            out.value_table, sem_type='corpus', geom_type='box')

    d_pose    = c_pose * translation3((-c_length - panel_thickness) * 0.5, 0, 0)
    d_width   = c_width
    d_length  = panel_thickness
    d_height  = c_height
    d_name    = '{}_door'.format(name)
    door      = cube((d_name,), d_pose, 
                     d_length, d_width, d_height,
                     (0.8, 0.7, 0.6, 1), 
                     out.value_table, sem_type='panel', geom_type='box')

    h_fac    = {'left': 1, 'right': -1, 'center': 0}[handle_location[0]]
    v_fac    = {'top': 1, 'bottom': -1, 'center': 0}[handle_location[1]]
    h_length, h_width, h_height = {'vertical': (handle_depth, handle_width, handle_height), 
                                   'horizontal': (handle_depth, handle_height, handle_width),
                                   'knob': (handle_depth, handle_width, handle_width)}[handle_orientation]

    h_pose = d_pose * translation3(-0.5 * handle_depth, 
                                    0.5 * (d_width - handle_margin - h_width) * h_fac, 
                                    0.5 * (d_height - handle_margin - h_height) * v_fac) * (rotation3_rpy(0, 0.5*math.pi, 0) if handle_orientation == 'knob' else eye(4))

    h_name = '{}_handle'.format(name)
    if handle_orientation != 'knob':
        handle = cube((h_name,), h_pose,
                           h_length, h_width, h_height,
                           (0.6, 0.6, 0.6, 1), 
                           out.value_table, sem_type='handle', geom_type='box')
    else:
        handle = cylinder((h_name,), h_pose,
                           h_width, h_length,
                           (0.6, 0.6, 0.6, 1), 
                           out.value_table, sem_type='handle', geom_type='cylinder')

    out.insert_data(compartment, c_name)
    out.insert_data(door, d_name)
    out.insert_data(handle, h_name)
    return out
