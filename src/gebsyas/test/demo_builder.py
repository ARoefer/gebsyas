from gebsyas.core.data_state import DataState
from gebsyas.core.subs_ds import ks_from_obj
from gebsyas.utils import bb

from giskardpy.symengine_wrappers import translation3

def cube(name, pose, length, width, height, color=(1,1,1,1), state={}):
    return ks_from_obj(bb(pose=pose, 
                          length=length, 
                          width=width, 
                          height=height,
                          color=color), name, state)[0]

def sphere(name, pose, radius, state={}):
    return ks_from_obj(bb(pose=pose, radius=radius), name, state)[0]

def cylinder(name, pose, radius, height, state={}):
    return ks_from_obj(bb(pose=pose, radius=radius, height=height), name, state)[0]


def compartment(name, pose, length, width, height, handle_height, handle_location=('left', 'top'), handle_orientation='vertical', data_state=None):
    out = data_state if data_state != None else DataState()

    c_pose   = pose
    c_length = length
    c_width  = width
    c_height = height
    c_name   = '{}_compartment'.format(name)
    compartment = cube((c_name,), c_pose, 
                            c_length, c_width, c_height,
                            (0.3, 0.2, 0.2, 1), 
                            out.value_table)

    d_pose    = c_pose * translation3(-c_length * 0.5 - 0.005, 0, 0)
    d_width   = c_width
    d_length  = 0.01
    d_height  = c_height
    d_name    = '{}_door'.format(name)
    door      = cube((d_name,), d_pose, 
                     d_length, d_width, d_height,
                     (0.8, 0.7, 0.6, 1), 
                     out.value_table)

    h_fac    = {'left': 1, 'right': -1, 'center': 0}[handle_location[0]]
    v_fac    = {'top': 1, 'bottom': -1, 'center': 0}[handle_location[1]]
    h_length, h_width, h_height = {'vertical': (0.04, 0.01, handle_height), 
                                   'horizontal': (0.04, handle_height, 0.01)}[handle_orientation]

    h_pose = d_pose * translation3(-0.02, 
                                    0.5 * (d_width - 0.1 - h_width) * h_fac, 
                                    0.5 * (d_height - 0.1 - h_height) * v_fac)

    h_name = '{}_handle'.format(name)
    handle = cube((h_name,), h_pose,
                       h_length, h_width, h_height,
                       (0.6, 0.6, 0.6, 1), 
                       out.value_table)

    out.insert_data(compartment, c_name)
    out.insert_data(door, d_name)
    out.insert_data(handle, h_name)
    return out
