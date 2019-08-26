import os
import sys

import xml.etree.ElementTree as ET
from kineverse.model.paths import Path
from kineverse.gradients.gradient_math import frame3_quaternion, translation3, rotation3_quaternion, frame3_rpy, spw, vector3
from kineverse.operations.urdf_operations import KinematicLink, URDFRobot, Geometry

gazebo_paths = os.environ['GAZEBO_MODEL_PATH'].split(':') if 'GAZEBO_MODEL_PATH' in os.environ else ['']

def print_return(f):
    def out(*args, **kwargs):
        result = f(*args, **kwargs)
        print('{}({},{})\n  -> {}'.format(f.func_name, ', '.join([str(x) for x in args]), ', '.join(['{}={}'.format(k, v) for k, v in kwargs.items()]), str(result)))
        return result
    return out

@print_return
def res_uri(uri):
    parts = uri.split('://')
    if len(parts) == 2:
        for p in gazebo_paths:
            r = '{}/{}'.format(p, parts[1])
            if os.path.isfile(r):
                return r
    return parts[0]

@print_return
def load_xml(path):
    return ET.parse(res_uri(path)).getroot()

@print_return
def load_world_xml(path):
    return load_xml(path).find('world')

@print_return
def load_model_xml(path):
    return load_xml(path).find('model')

@print_return
def load_world_from_xml(model, path):
    for world_node in load_xml(path).findall('world'):
        world_loader(model, Path([]), world_node, spw.eye(4))

@print_return
def load_model_from_xml(model, path):
    for model_node in load_xml(path).findall('model'):
        model_loader(model, Path([]), model_node, spw.eye(4))    

@print_return
def handle_pose_tag(pose_tag, path, frames={}):
    if pose_tag is None:
        return spw.eye(4)

    args = [float(x) for x in pose_tag.text.split(' ') if len(x) > 0]
    if len(args) != 6:
        raise Exception('A pose tag should contain six values separated by spaces. This one contains {}. Path: {}'.format(len(args), path))
    out = frame3_rpy(args[3], args[4], args[5], args[:3])
    if pose_tag.get('frame') is not None and pose_tag.get('frame') != '':
        return frames[pose_tag.get('frame')] * out
    return out

@print_return
def handle_include_tag(model, prefix, include_node):
    uri  = include_node.find('uri')
    if uri is None:
        raise Exception('uri tag is missing in include tag. Path: {}'.format(prefix))
    uri  = uri.text
    name = include_node.find('name')
    pose = handle_include_tag(include_node.find('pose'), prefix)
    if name is not None:
        if uri[:8] == 'model://':
            model_loader(model, prefix + (name.text,), load_model_xml(uri[8:]), pose, False)
        elif uri[:8] == 'world://':
            world_loader(model, prefix + (name.text,), load_world_xml(uri[8:]), pose, False)
    else:
        if uri[:8] == 'model://':
            model_loader(model, prefix, load_model_xml(uri[8:]), pose)
        elif uri[:8] == 'world://':
            world_loader(model, prefix, load_world_xml(uri[8:]))

@print_return
def handle_geometry_node(part_path, geometry_node):
    if geometry_node.find('empty') is not None:
        return None

    sphere_node   = geometry_node.find('sphere')
    cylinder_node = geometry_node.find('cylinder')
    mesh_node     = geometry_node.find('mesh')
    box_node      = geometry_node.find('box')
    if sphere_node is not None:
        return Geometry(part_path, spw.eye(4), 'sphere', [float(sphere_node.find('radius').text)] * 3)
    elif cylinder_node is not None:
        diameter = float(cylinder_node.find('radius').text) * 2
        length   = float(cylinder_node.find('length').text)
        return Geometry(part_path, spw.eye(4), 'cylinder', vector3(diameter, diameter, length))
    elif mesh_node is not None:
        resolved_path = res_uri(mesh_node.find('uri').text)
        scale =  vector3(*[float(x) for x in mesh_node.find('scale').text.split(' ') if len(x) > 0]) if mesh_node.find('scale') is not None else vector3(1,1,1)
        return Geometry(part_path, spw.eye(4), 'mesh', scale, resolved_path)
    elif box_node is not None:
        size =  vector3(*[float(x) for x in box_node.find('size').text.split(' ') if len(x) > 0])
        return Geometry(part_path, spw.eye(4), 'box', size)
    else:
        print('Currently only empty, sphere, cylinder, mesh, box are supported.')

@print_return
def model_loader(model, prefix, model_node, pose, use_name=True):
    if use_name:
        prefix = prefix + (model_node.get('name'), )

    print('{} children:\n  {}'.format(prefix, '\n  '.join([str(x) for x in model_node])))

    static = False
    static_node = model_node.find('static')
    if static_node is not None:
        static_text = static_node.text.strip()
        print('Static text: "{}"'.format(static_text))
        static = static_text.lower() == 'true' or static_text == '1' 

    if not static:
        return

    frames = {}
    for frame_node in model_node.findall('frame'):
        frames[frame_node.get('name')] = handle_pose_tag(frame_node.find('pose'), prefix, frames)


    urdf_container = URDFRobot(str(prefix))
    for link_node in model_node.findall('link'):
        name = link_node.get('name')
        pose = handle_pose_tag(link_node.find('pose'), prefix, frames)
        collision = None
        if link_node.find('collision') is not None:
            collision_node = link_node.find('collision')
            if collision_node is not None:
                collision = handle_geometry_node(str(prefix), collision_node.find('geometry'))
        urdf_container.links[name] = KinematicLink('map', pose, geometry=None, collision=collision)

    for joint_node in model_node.findall('joint'):
        if joint_node.get('type') == 'fixed':
            print('Skipped over a fixed joint')
    
    model.set_data(prefix, urdf_container)

@print_return
def world_loader(model, prefix, world_node, pose, use_name=True):
    if use_name:
        prefix = prefix + (world_node.get('name'), )

    model.set_data(prefix, {})

    for include_node in world_node.findall('include'):
        handle_include_tag(model, prefix, include_node)

    for model_node in world_node.findall('model'):
        model_loader(model, prefix, model_node, pose)

    state_node = world_node.find('state')
    for model_node in state_node.findall('model'):
        path = prefix + (model_node.get('name'), )
        if model.has_data(path):
            m = model.get_data(path)
            for link_node in model_node.findall('link'):
                link = m.links[link_node.get('name')]
                link.pose = handle_pose_tag(link_node.find('pose'), path, {})

