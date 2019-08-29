import os
import sys

import xml.etree.ElementTree as ET
from kineverse.model.paths import Path
from kineverse.gradients.gradient_math import frame3_quaternion, translation3, rotation3_quaternion, frame3_rpy, spw, vector3
from kineverse.operations.urdf_operations import KinematicLink, URDFRobot, Geometry

def print_return(f):
    def out(*args, **kwargs):
        result = f(*args, **kwargs)
        print('{}({},{})\n  -> {}'.format(f.func_name, ', '.join([str(x) for x in args]), ', '.join(['{}={}'.format(k, v) for k, v in kwargs.items()]), str(result)))
        return result
    return out


search_paths = (['.'] + os.environ['GAZEBO_MODEL_PATH'].split(':')) if 'GAZEBO_MODEL_PATH' in os.environ else ['.'] 

@print_return
def res_sdf_path(rpath):
    """Resolves a ROS package relative path to a global path.
    :param rpath: Potential ROS URI to resolve.
    :type rpath: str
    :return: Local file system path
    :rtype: str
    """
    if rpath[:8] == 'model://':

        rpath = rpath[8:]
        file  = rpath[rpath.rfind('/') + 1:]
        if '.' not in file or file[-4] != '.':
            file += '.sdf'

        for rpp in search_paths:
            if os.path.isfile('{}/{}'.format(rpp, rpath)):
                return os.path.abspath('{}/{}'.format(rpp, rpath))
        #raise Exception('Path "{}" can not be found in GAZEBO_MODEL_PATH or locally!'.format(rpath))
    return rpath

#@print_return
def load_xml(path):
    return ET.parse(res_sdf_path(path)).getroot()

def parse_bool_node(node, default=False):
    if node is not None:
        return node.text.lower() == 'true' or node.text == '1' 
    return default

def parse_float_node(node, default=0.0):
    if node is not None:
        return float(node.text.lower().strip())
    return default

def parse_int_node(node, default=0):
    if node is not None:
        return int(node.text.lower().strip())
    return default

def parse_int_array(node, num, default):
    if node is not None:
        args = [int(x) for x in node.text.lower().strip().split(' ') if len(x) > 0]
        if len(args) != num:
            raise Exception('Ill-formed tag. Should contain {} ints but contains {}.'.format(num, len(args)))
        return args
    return default

def parse_float_array(node, num, default):
    if node is not None:
        args = [float(x) for x in node.text.lower().strip().split(' ') if len(x) > 0]
        if len(args) != num:
            raise Exception('Ill-formed tag. Should contain {} floats but contains {}.'.format(num, len(args)))
        return args
    return default


class SDFPose(object):
    def __init__(self, transform, frame):
        self.transform = transform
        self.frame = frame

    @classmethod
    def from_xml(cls, pose_node):
        if pose_node is None:
            return cls(spw.eye(4), None)
        else:
            args = parse_float_array(pose_node, 6, [0.0]*6)
            transform = frame3_rpy(args[3], args[4], args[5], args[:3])
            frame = pose_node.get('frame')
            if frame == '':
                frame = None
            return cls(transform, frame)

    def apply_scale(self, scale):
        transform = self.transform * 1
        transform[:3,3] = transform[:3,3].mul_matrix(scale[:3,:])
        return SDFPose(transform, self.frame)


class SDFFrame(object):
    def __init__(self, name, pose):
        self.name = name
        self.pose = pose

    @classmethod
    def from_xml(cls, frame_node):
        if frame_node is None:
            raise Exception('{} has no default initialization'.format(type(self)))
        return cls(frame_node.get('name'), SDFPose.from_xml(frame_node.find('pose')))

    def apply_scale(self, scale):
        return SDFFrame(self.name, self.pose.apply_scale(scale))


class SDFInertial(object):
    def __init__(self, inertial_node):
        self.mass = 1.0
        self.tensor = spw.eye(3)
        self.pose   = SDFPose.from_xml(None)
        self.frames = {}

        if inertial_node is not None:
            self.mass = parse_float_node(inertial_node.find('mass'), 1.0)
            if inertial_node.find('inertia') is not None:
                m_node    = inertial_node.find('inertia')
                self.pose = SDFPose.from_xml(m_node.find('pose'))
                self.frames = {n.get('name'): SDFFrame.from_xml(n) for n in m_node.findall('frame')}
                xx, xy, xz, yy, yz, zz = [parse_float_node(m_node.get(x)) for x in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']]
                self.tensor = spw.Matrix([[xx, xy, xz], 
                                          [xy, yy, yz], 
                                          [xz, yz, zz]])


class SDFGeometry(object):
    def __init__(self, type, size=vector3(0,0,0), radius=1.0, length=1.0, mesh=None, scale=vector3(1,1,1)):
        self.type   = type
        self.size   = size
        self.radius = radius
        self.length = length
        self.mesh   = mesh
        self.scale  = scale

    @classmethod
    def from_xml(cls, geometry_node):
        if geometry_node is None:
            raise Exception('{} has no default initialization'.format(cls))

        if geometry_node.find('empty') is not None:
            type = 'empty'
            return cls(type)
        elif geometry_node.find('box') is not None:
            type = 'box'
            s_node = geometry_node.find('box')
            size = spw.Matrix(parse_float_array(s_node.find('size'), 3, [1]*3))
            return cls(type, size=size)
        elif geometry_node.find('cylinder') is not None:
            type   = 'cylinder'
            s_node      = geometry_node.find('cylinder')        
            radius = parse_float_node(s_node.find('radius'), 1.0)
            length = parse_float_node(s_node.find('length'), 1.0)
            return cls(type, radius=radius, length=length)
        elif geometry_node.find('mesh') is not None:
            type  = 'mesh'
            s_node     = geometry_node.find('mesh')
            mesh  = res_sdf_path(s_node.find('uri').text)
            scale = spw.Matrix(parse_float_array(s_node.find('scale'), 3, [1.0]*3))
            return cls(type, scale=scale, mesh=mesh)
        elif geometry_node.find('sphere') is not None:
            type   = 'sphere'
            s_node      = geometry_node.find('sphere')
            radius = parse_float_node(s_node.find('radius'), 1.0)
            return cls(type, radius=radius)
        else:
            print('Unsupported geometry type. Skipping it.')
            return None

    def apply_scale(self, scale):
        scaled_size  = spw.Matrix([self.size[0]  * scale[0], self.size[1]  * scale[1], self.size[2]  * scale[2]])
        scaled_scale = spw.Matrix([self.scale[0] * scale[0], self.scale[1] * scale[1], self.scale[2] * scale[2]])
        return SDFGeometry(self.type, 
                           scaled_size, 
                           self.radius * scale[0], 
                           self.length * scale[2], 
                           self.mesh, 
                           scaled_scale)


class SDFCollision(object):
    def __init__(self, name, laser_retro, max_contacts, frames, pose, geometry):
        self.name         = name
        self.laser_retro  = laser_retro
        self.max_contacts = max_contacts
        self.frames       = frames
        self.pose         = pose
        self.geometry     = geometry

    @classmethod
    def from_xml(cls, collision_node):
        if collision_node is None:
            raise Exception('{} has no default initialization'.format(cls))
        return cls(name=collision_node.get('name'),
                  laser_retro=parse_float_node(collision_node.find('laser_retro'), 0.0),
                  max_contacts=parse_int_node(collision_node.find('max_contacts'), 10),
                  frames={n.get('name'): SDFFrame.from_xml(n) for n in collision_node.findall('frame')}, 
                  pose=SDFPose.from_xml(collision_node.find('pose')), 
                  geometry=SDFGeometry.from_xml(collision_node.find('geometry')))

    def apply_scale(self, scale):
        return SDFCollision(self.name,
                            self.laser_retro,
                            self.max_contacts,
                            {n: f.apply_scale(scale) for n, f in self.frames.items()},
                            self.pose.apply_scale(scale),
                            self.geometry.apply_scale(scale) if self.geometry is not None else None)

class SDFVisual(object):
    def __init__(self, name, cast_shadows, transparency, frames, pose, geometry):
        self.name = name
        self.cast_shadows = cast_shadows
        self.transparency = transparency
        self.frames = frames
        self.pose = pose
        self.geometry = geometry

    @classmethod
    def from_xml(cls, visual_node):
        if visual_node is None:
            raise Exception('{} has no default initialization'.format(cls))

        return cls(name=visual_node.get('name'),
                   cast_shadows=parse_bool_node(visual_node.find('cast_shadow'), True),
                   transparency=parse_float_node(visual_node.find('transparency'), 0.0),
                   frames={n.get('name'): SDFFrame.from_xml(n) for n in visual_node.findall('frame')},
                   pose=SDFPose.from_xml(visual_node.find('pose')),
                   geometry=SDFGeometry.from_xml(visual_node.find('geometry')))

    def apply_scale(self, scale=vector3(1,1,1)):
        return SDFVisual(self.name,
                         self.cast_shadows,
                         self.transparency,
                         self.frames,
                         self.pose,
                         self.geometry.apply_scale(scale) if self.geometry is not None else None)


class SDFLink(object):
    def __init__(self, name, gravity, enable_wind, self_collide, kinematic_only, must_be_base, frames, pose, inertial, collisions, visuals, sensors):
        self.name           = name
        self.gravity        = gravity
        self.enable_wind    = enable_wind
        self.self_collide   = self_collide
        self.kinematic_only = kinematic_only
        self.must_be_base   = must_be_base
        self.frames         = frames
        self.pose           = pose
        self.inertial       = inertial
        self.collisions     = collisions
        self.visuals        = visuals
        self.sensors        = sensors

    @classmethod
    def from_xml(cls, link_node):
        if link_node is None:
            raise Exception('{} has no default initialization'.format(cls))
        return cls(name=link_node.get('name'),
                    gravity=parse_bool_node(link_node.find('gravity'), True),
                    enable_wind=parse_bool_node(link_node.find('enable_wind')),
                    self_collide=parse_bool_node(link_node.find('self_collide')),
                    kinematic_only=parse_bool_node(link_node.find('kinematic')),
                    must_be_base=parse_bool_node(link_node.find('must_be_base_link')),
                    frames={n.get('name'): SDFFrame.from_xml(n) for n in link_node.findall('frame')},
                    pose=SDFPose.from_xml(link_node.find('pose')),
                    inertial=SDFInertial(link_node.find('inertial')),
                    collisions={n.get('name'): SDFCollision.from_xml(n) for n in link_node.findall('collision')},
                    visuals={n.get('name'): SDFVisual.from_xml(n) for n in link_node.findall('visual')},
                    sensors={})#{n.get('name'): SDFSensor(n) for n in link_node.findall('sensor')})

    def apply_state(self, state, scale=vector3(1,1,1)):
        return SDFLink(self.name, 
                       self.gravity, 
                       self.enable_wind, 
                       self.self_collide, 
                       self.kinematic_only,
                       self.must_be_base,
                       {n: f.apply_scale(scale) if n not in state.frames else state.frames[n] for n, f in self.frames.items()},
                       self.pose.apply_scale(scale) if state.pose is None else state.pose,
                       self.inertial,
                       {n: c.apply_scale(scale) for n, c in self.collisions.items()},
                       {n: v.apply_scale(scale) for n, v in self.visuals.items()},
                       self.sensors)

    def apply_scale(self, scale):
        return SDFLink(self.name, 
                       self.gravity, 
                       self.enable_wind, 
                       self.self_collide, 
                       self.kinematic_only,
                       self.must_be_base,
                       {n: f.apply_scale(scale) for n, f in self.frames.items()},
                       self.pose.apply_scale(scale),
                       self.inertial,
                       {n: c.apply_scale(scale) for n, c in self.collisions.items()},
                       {n: v.apply_scale(scale) for n, v in self.visuals.items()},
                       self.sensors)


class SDFLinkState(object):
    def __init__(self, ls_node):
        if ls_node is None:
            raise Exception('{} has no default initialization'.format(type(self)))

        self.name         = ls_node.get('name')
        self.velocity     = parse_float_array(ls_node.find('velocity'), 6, [0.0]*6)
        self.acceleration = parse_float_array(ls_node.find('acceleration'), 6, [0.0]*6)
        self.wrench       = parse_float_array(ls_node.find('wrench'), 6, [0.0]*6)
        self.pose         = None if ls_node.find('pose') is None else SDFPose.from_xml(ls_node.find('pose'))
        self.frames       = {n.get('name'): SDFFrame.from_xml(n) for n in ls_node.findall('frame')}

class SDFAxis(object):
    def __init__(self, axis, initial_position, use_parent_model_frame, damping, friction, spring_reference, spring_stiffness, limit_lower, limit_upper, effort, velocity, stiffness, dissipation):
        self.axis                   = axis
        self.initial_position       = initial_position
        self.use_parent_model_frame = use_parent_model_frame
        self.damping                = damping
        self.friction               = friction
        self.spring_reference       = spring_reference
        self.spring_stiffness       = spring_stiffness
        self.limit_lower            = limit_lower
        self.limit_upper            = limit_upper
        self.effort                 = effort
        self.velocity               = velocity
        self.stiffness              = stiffness
        self.dissipation            = dissipation


    @classmethod
    def from_xml(cls, axis_node):
        if axis_node is None:
            return cls(axis=vector3(0,0,1),
                       initial_position=0.0,
                       use_parent_model_frame=False,
                       damping=0.0,
                       friction=0.0,
                       spring_reference=0.0,
                       spring_stiffness=0.0,
                       limit_lower=-1e16,
                       limit_upper= 1e16,
                       effort=-1,
                       velocity=-1 ,
                       stiffness=1e8,
                       dissipation=1)
        else:
            d_node = axis_node.find('dynamics')
            l_node = axis_node.find('limit')
            return cls(axis=vector3(*parse_float_array(axis_node.find('xyz'), 3, [0, 0, 1.0])),
                       initial_position=parse_float_node(axis_node.find('initial_position'), 0.0),
                       use_parent_model_frame=parse_bool_node(axis_node.find(',use_parent_model_frame')),
                       damping=0.0 if d_node is None else d_node.get('damping'),
                       friction=0.0 if d_node is None else d_node.get('friction'),
                       spring_reference=0.0 if d_node is None else d_node.get('spring_reference'),
                       spring_stiffness=0.0 if d_node is None else d_node.get('spring_stiffness'),
                       limit_lower=parse_float_node(l_node.find('lower'), -1e16),
                       limit_upper=parse_float_node(l_node.find('upper'),  1e16),
                       effort=parse_float_node(l_node.find('effort'), -1),
                       velocity=parse_float_node(l_node.find('velocity'), -1 ),
                       stiffness=parse_float_node(l_node.find('stiffness'), 1e8),
                       dissipation=parse_float_node(l_node.find('dissipation'), 1))



class SDFJoint(object):
    def __init__(self, joint_node):
        if joint_node is None:
            raise Exception('{} has no default initialization'.format(type(self)))        
        self.name = joint_node.get('name')
        self.type = joint_node.get('type')
        self.parent = joint_node.find('parent').text
        self.child  = joint_node.find('child').text
        self.frames = {n.get('name'): SDFFrame(n) for n in joint_node.findall('frame')}
        self.pose   = SDFPose.from_xml(joint_node.find('pose'))
        if self.type == 'gearbox':
            self.gearbox_ratio = parse_float_node(joint_node.find('gearbox_ratio'), 1.0)
            self.gearbox_reference_body = joint_node.find('gearbox_reference_body').text
        elif self.type == 'screw':
            self.thread_pitch = parse_float_node(joint_node.find('thread_pitch'), 1.0)
        elif self.type in {'revolute', 'revolute2', 'ball'}:
            self.axis = SDFAxis(joint_node.find('axis'))
            if self.type == 'revolute2' or self.type == 'ball':
                self.axis2 = SDFAxis(joint_node.find('axis2'))


class SDFJointState(object):
    def __init__(self, js_node):
        if js_node is None:
            raise Exception('{} has no default initialization'.format(type(self)))
        
        self.name  = js_node.get('name')
        self.angle = [x for _, x in sorted([(int(n.get('axis')), parse_float_node(n, 0.0)) for n in js_node.findall('angle')])]


class SDFModel(object):
    def __init__(self, name, pose, frames, links, joints, static, sub_models, enable_wind):
        self.name = name
        self.pose = pose
        self.frames = frames
        self.links = links
        self.joints = joints
        self.static = static
        self.sub_models = sub_models
        self.enable_wind = enable_wind

    @classmethod
    def from_xml(cls, model_node):
        if model_node is None:
            raise Exception('{} has no default initialization'.format(cls))

        name   = model_node.get('name')
        pose   = SDFPose.from_xml(model_node.find('pose'))
        frames = {n.get('name'): SDFFrame.from_xml(n) for n in model_node.findall('frame')}
        links  = {n.get('name'): SDFLink.from_xml(n)  for n in model_node.findall('link')}
        joints = {n.get('name'): SDFJoint(n) for n in model_node.findall('joint')}
        static = parse_bool_node(model_node.find('static'))
        sub_models = {n.get('name'): SDFModel.from_xml(n) for n in model_node.findall('model')}
        enable_wind = parse_bool_node(model_node.find('enable_wind'))
        for include_node in model_node.findall('include'):
            fp  = res_sdf_path(include_node.find('uri').text)
            sdf = SDF(load_xml(fp))
            name, sdf_model = sdf.models.items()[0]
            if include_node.find('name'):
                name = include_node.find('name').text
                sdf_model.name = name
            if include_node.find('static'):
                sdf_model.static = parse_bool_node(include_node.find('static'))
            if include_node.find('pose'):
                sdf_model.apply_pose(SDFPose(include_node.find('pose')))
            if name in sub_models:
                raise Exception('Name "{}" of included sub model in "{}" is not unique!'.format(name, name))
            sub_models[name] = sdf_model
        return cls(name, pose, frames, links, joints, static, sub_models, enable_wind)

    def apply_pose(self, pose):
        self.pose.apply_pose(pose)

    def apply_state(self, state):
        return SDFModel(self.name,
                        self.pose if state.pose is None else state.pose,
                        {n: f if n not in state.frames else state.frame[n] for n, f in self.frames.items()},
                        {n: l if n not in state.links else l.apply_state(state.links[n], state.scale) for n, l in self.links.items()},
                        self.joints, 
                        self.static, 
                        {n: m if n not in state.models else m.apply_state(state.models[n]) for n, m in self.sub_models.items()}, 
                        self.enable_wind)


class SDFModelState(object):
    def __init__(self, ms_node):
        if ms_node is None:
            raise Exception('{} has no default initialization'.format(type(self)))

        self.name   = ms_node.get('name')
        self.joints = {n.get('name'): SDFJointState(n) for n in ms_node.findall('joint')}
        self.links  = {n.get('name'): SDFLinkState(n)  for n in ms_node.findall('link')}
        self.models = {n.get('name'): SDFModelState(n) for n in ms_node.findall('model')}
        self.frames = {n.get('name'): SDFFrame(n)      for n in ms_node.findall('frame')}
        self.pose   = None if ms_node.find('pose') is None else SDFPose.from_xml(ms_node.find('pose'))
        self.scale  = vector3(*parse_float_array(ms_node.find('scale'), 3, [1.0]*3))


class SDFWorldState(object):
    def __init__(self, state_node):
        if state_node is None:
            raise Exception('{} has no default initialization'.format(type(self)))

        self.world_name = state_node.get('world_name')
        self.sim_time   = 0.0 # parse_float_node(state_node.find('sim_time'), 0.0)
        self.wall_time  = 0.0 # parse_float_node(state_node.find('wall_time'), 0.0)
        self.real_time  = 0.0 # parse_float_node(state_node.find('real_time'), 0.0)
        self.iterations = parse_int_node(state_node.find('iterations'), 0.0)
        self.inserted_models = {} if state_node.find('insertions') is None else {n.get('name'): SDFModel.from_xml(n) for n in state_node.find('insertions').findall('model')}
        self.deletions = set() if state_node.find('deletions') is None else {n.text for n in state_node.find('deletions').findall('name')}
        self.model_state = {n.get('name'): SDFModelState(n) for n in state_node.findall('model')}


class SDFWorld(object):
    def __init__(self, world_node):
        if world_node is None:
            raise Exception('{} has no default initialization'.format(type(self)))

        self.name   = world_node.get('name')
        self.models = {n.get('name'): SDFModel.from_xml(n) for n in world_node.findall('model')}
        self.state  = None if world_node.find('state') is None else SDFWorldState(world_node.find('state'))


class SDF(object):
    def __init__(self, sdf_node):
        self.version = sdf_node.get('version')
        self.worlds  = {n.get('name'): SDFWorld(n) for n in sdf_node.findall('world')}
        self.models  = {n.get('name'): SDFModel.from_xml(n) for n in sdf_node.findall('model')}

    def get_models(self):
        out = self.models.copy()
        for w in self.worlds.values():
            out.update(w.get_models())
        return out


def resolve_frames(unresolved_frames, root_pose=spw.eye(4)):
    if len(unresolved_frames) == 0:
        return unresolved_frames
    deps = {}
    for n, f in unresolved_frames.items():
        if f.pose.frame not in deps:
            deps[f.pose.frame] = {}
        deps[f.pose.frame][n] = f

    if None not in deps:
        raise Exception('No frames without dependencies in set. Frame -> Dep:\n  {}'.format('\n  '.join(['{} -> {}'.format(n, f.pose.frame) for n, f in unresolved_frames.items()])))
    next_set   = {n: SDFFrame(n, SDFPose(root_pose * f.pose.transform, None)) for n, f in deps[None].items()}
    new_frames = deps[None]
    while len(next_set) > 0:
        temp_set = {}
        for n, f in next_set.items():
            if n in deps:
                for dn, df in deps[n].items():
                    res_frame = SDFFrame(dn, SDFPose(f.pose.transform * df.pose.transform, n))
                    new_frames[dn] = res_frame
                    temp_set[dn]   = res_frame
        next_set = temp_set
    return new_frames


def collision_sdf_to_geometry(sdf_collisions):
    out = {}
    for n, c in sdf_collisions.items():
        if c.geometry is not None and c.geometry.type != 'empty':
            if c.geometry.type == 'box':
                out[n] = Geometry('map', c.pose.transform, c.geometry.type, scale=c.geometry.size, mesh=None)
            elif c.geometry.type == 'cylinder':
                out[n] = Geometry('map', c.pose.transform, c.geometry.type, scale=vector3(c.geometry.radius * 2, c.geometry.radius * 2, c.geometry.length), mesh=None)
            elif c.geometry.type == 'sphere':
                out[n] = Geometry('map', c.pose.transform, c.geometry.type, scale=vector3(c.geometry.radius * 2, 
                            c.geometry.radius * 2, 
                            c.geometry.radius * 2), mesh=None)
            elif c.geometry.type == 'mesh':
                out[n] = Geometry('map', c.pose.transform, 
                                           c.geometry.type, 
                                           scale=c.geometry.scale, 
                                           mesh=c.geometry.mesh)
    return out


def link_to_static_link(sdf_link, root_pose=spw.eye(4), frames={}):
    link_pose = root_pose * sdf_link.pose.transform
    new_frames = frames.copy()
    new_frames.update(resolve_frames(sdf_link.frames))

    return KinematicLink('map', link_pose, geometry=None, collision=collision_sdf_to_geometry(sdf_link.collisions), inertial=None)


def model_to_static_links(sdf_model, prefix, root_pose=spw.eye(4), frames={}):
    prefix = prefix + (sdf_model.name, )
    model_pose = root_pose * sdf_model.pose.transform
    out = {}
    new_frames = frames.copy()
    new_frames.update(resolve_frames(frames, model_pose))
    for n, m in sdf_model.sub_models.items():
        out.update(model_to_static_links(m, prefix + (n,), root_pose, new_frames))

    for n, l in sdf_model.links.items():
        out[prefix + (n, )] = link_to_static_link(l, root_pose, new_frames)

    return out


def world_to_static_links(sdf_world, prefix):
    prefix = prefix + (sdf_world.name, )

    out = {}
    for n, m in sdf_world.models.items():
        if n in sdf_world.state.model_state:
            m = m.apply_state(sdf_world.state.model_state[n])
        out.update(model_to_static_links(m, prefix))
    return out


def load_static_sdf_to_model(km, prefix, sdf_path):
    sdf = SDF(load_xml(res_sdf_path(sdf_path)))

    for p, l in world_to_static_links(sdf.worlds.values()[0], prefix).items():
        if len([True for x in p if 'ground' in x]) == 0:
            for x in range(1, len(p)):
                if not km.has_data(p[:x]):
                    km.set_data(p[:x], {})
            km.set_data(p, l)
            print('Added link: {}\n  {}'.format(p, link_to_str(l)))

def geom_to_str(geom):
    if geom.type == 'sphere':
        return 'Sphere: radius = {}'.format(geom.scale[0] * 0.5)
    elif geom.type == 'cylinder':
        return 'Cylinder: radius = {}, length = {}'.format(geom.scale[0] * 0.5, geom.scale[2])
    elif geom.type == 'box':
        return 'Box: extents [{}, {}, {}]'.format(*geom.scale[:3])
    elif geom.type == 'mesh':
        return 'Mesh: path = "{}", scale = [{}, {}, {}]'.format(geom.mesh, *geom.scale[:3])
    return ''

def link_to_str(link):
    if link.collision != None:
        return 'Link:\nParent: {}\nPose: \n{}\nCollision:\n{}'.format(link.parent, link.pose, '\n'.join(['{}:\n{}'.format(k, geom_to_str(c)) for k,c in link.collision.items()]))
    else:
        return 'Link:\nParent: {}'.format(link.parent)