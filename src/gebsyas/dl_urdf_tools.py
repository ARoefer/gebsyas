from gebsyas.dl_reasoning import DLCompoundObject, DLRigidObject, DLShape, DLCube, DLSphere, DLCylinder, DLIded
from gebsyas.utils import rot3_to_rpy
from gebsyas.simulator import vec3_to_list
from giskardpy.symengine_wrappers import *
from urdf_parser_py.urdf import Link, Joint, Collision, Visual, Pose, JointDynamics, JointLimit, Inertial, Sphere, Box, Cylinder, Inertia

def add_dl_object_to_urdf(urdf_model, attachment_link, dl_object, transform):
	"""Generates a new URDF in which the given dl-object is attached to the specified link of the robot."""
	if not DLRigidObject.is_a(dl_object):
		raise Exception('Can only add rigid bodies to urdf')

	if attachment_link not in urdf_model.link_map:
		raise Exception('Can not attach object to link "{}" as it does not exist in the urdf model.'.format(attachment_link))

	zero_pose = Pose([0,0,0], [0,0,0])
	if DLCylinder.is_a(dl_object):
		geometry  = Cylinder(dl_object.radius, dl_object.height)
	elif DLSphere.is_a(dl_object):
		geometry  = Sphere(dl_object.radius)
	elif DLCube.is_a(dl_object):
		geometry  = Box([dl_object.depth, dl_object.length, dl_object.height])
	else:
		raise Exception('Unsupported object shape')

	pos = pos_of(transform)

	pose = Pose(vec3_to_list(pos), list(rot3_to_rpy(transform, True)))
	print(pose)

	m = dl_object.mass
	visual    = Visual(geometry, None, zero_pose)
	collision = Collision(geometry, zero_pose)
	inertia   = Inertia(m * (pos[1]**2 + pos[2]**2),       - m * pos[0] * pos[1],        -m * pos[0] * pos[2],
													 m * (pos[0]**2 + pos[2]**2),         m * pos[1] * pos[2],
													 							  m * (pos[0]**2 + pos[1]**2))
	inertial  = Inertial(dl_object.mass, inertia, zero_pose)
	name      = dl_object.id if DLIded.is_a(dl_object) else 'link_{}'.format(len(urdf_model.links))
	link      = Link(name, visual, inertial, collision, zero_pose)
	joint     = Joint('{}_joint'.format(name.replace('_link', '')), attachment_link, name, 'fixed', [0,0,0], pose)

	urdf_model.add_link(link)
	urdf_model.add_joint(joint)

