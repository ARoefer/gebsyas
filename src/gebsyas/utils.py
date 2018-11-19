import os
import symengine as sp
import numpy as np
import rospy
import yaml

from collections import namedtuple
from gebsyas.data_structures import StampedData, JointState, GaussianPoseComponent
from giskardpy.symengine_wrappers import *
from sensor_msgs.msg import JointState as JointStateMsg
from gop_gebsyas_msgs.msg import ProbObject as PObject
from gop_gebsyas_msgs.msg import ObjectPoseGaussianComponent as OPGMsg
from gop_gebsyas_msgs.msg import SearchObject as SearchObjectMsg
from iai_bullet_sim.utils import Frame, Vector3, Point3
from visualization_msgs.msg import Marker as MarkerMsg
from copy import deepcopy

pi = 3.14159265359
rad2deg = 57.2957795131
deg2rad = 1.0 / rad2deg

def symbol_formatter(symbol_name):
	if '__' in symbol_name:
		raise Exception('Illegal character sequence in symbol name "{}"! Double underscore "__" is a separator sequence.'.format(symbol_name))
	return sp.Symbol(symbol_name.replace('/', '__'))

def fake_heaviside(expr):
	return 0 if expr <= 0 else 1

def res_pkg_path(rpath):
    """Resolves a ROS package relative path to a global path.
    :param rpath: Potential ROS URI to resolve.
    :type rpath: str
    :return: Local file system path
    :rtype: str
    """
    if rpath[:10] == 'package://':
        paths = os.environ['ROS_PACKAGE_PATH'].split(':')

        rpath = rpath[10:]
        pkg = rpath[:rpath.find('/')]

        for rpp in paths:
            if rpp[rpp.rfind('/') + 1:] == pkg:
                return '{}/{}'.format(rpp[:rpp.rfind('/')], rpath)
            if os.path.isdir('{}/{}'.format(rpp, pkg)):
                return '{}/{}'.format(rpp, rpath)
        raise Exception('Package "{}" can not be found in ROS_PACKAGE_PATH!'.format(pkg))
    return rpath


def import_class(class_path):
    """Imports a class using a type string.
    :param class_path: Type string of the class.
    :type  class_path: str
    :rtype: type
    """
    components = class_path.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def nested_list_to_sym(l):
	if type(l) == list:
		return [nested_list_to_sym(x) for x in l]
	else:
		return sympify(l)

def nested_symlist_to_yaml(l):
	if type(l) == list:
		return [nested_symlist_to_yaml(x) for x in l]
	else:
		return '{}'.format(str(l))

def yaml_sym_representer(dumper, data):
	return dumper.represent_scalar('!SymExpr', str(data))

def yaml_sym_constructor(loader, node):
	return sympify(str(loader.construct_scalar(node)))

def yaml_matrix_representer(dumper, matrix):
	return dumper.represent_sequence('!SymMatrix', nested_symlist_to_yaml(matrix.tolist()))

def yaml_matrix_constructor(loader, node):
	return Matrix(nested_list_to_sym(loader.construct_sequence(node, deep=True)))

yaml.add_representer(sp.Basic, yaml_sym_representer)
yaml.add_representer(sp.Number, yaml_sym_representer)
yaml.add_representer(sp.Expr, yaml_sym_representer)
yaml.add_representer(sp.Add, yaml_sym_representer)
yaml.add_representer(sp.Mul, yaml_sym_representer)
yaml.add_representer(sp.Min, yaml_sym_representer)
yaml.add_representer(sp.Max, yaml_sym_representer)
yaml.add_representer(sp.Matrix, yaml_matrix_representer)

yaml.add_constructor('!SymExpr', yaml_sym_constructor)
yaml.add_constructor('!SymMatrix', yaml_matrix_constructor)

YAML = yaml



def saturate(x, low=0, high=1):
	breadth_scale = 6 / high - low
	return 1 / (1 + sp.exp(-2* ( x * breadth_scale + low - 3)))


def tip_at_one(x):
    return -4*x**2 + 8 * x - 3


def abs(x):
	return sp.sqrt(x**2)


def cmdDictToJointState(command):
	js = JointStateMsg()
	js.header.stamp = rospy.Time.now()
	for joint, vel in command.items():
		js.name.append(str(joint))
		js.position.append(0)
		js.velocity.append(vel)
		js.effort.append(0)

	return js

def subs_if_sym(var, subs_dict):
	t = type(var)
	if t == int or t == float or t == str:
		return var
	else:
		return var.subs(subs_dict)

# Returns [x,y,z,w]
QuatT = namedtuple('QuatT', ['x', 'y', 'z', 'w'])
RPY = namedtuple('RPY', ['r', 'p', 'y'])

def rot3_to_quat(rot3):
	w  = sp.sqrt(1 + rot3[0,0] + rot3[1,1] + rot3[2,2]) * 0.5
	w4 = 4 * w
	x  = (rot3[2,1] - rot3[1,2]) / w4
	y  = (rot3[0,2] - rot3[2,0]) / w4
	z  = (rot3[1,0] - rot3[0,1]) / w4
	return QuatT(x,y,z,w)


def rot3_to_rpy(rot3, evaluate=False):
	sy = sp.sqrt(rot3[0,0] * rot3[0,0] + rot3[2,2] * rot3[2,2])

	if sy >= 1e-6:
		if evaluate:
			return RPY(sp.atan2(rot3[2,1], rot3[2,2]).evalf(real=True), sp.atan2(-rot3[2,0], sy).evalf(real=True), sp.atan2(rot3[1,0], rot3[0,0]).evalf(real=True))
		else:
			return RPY(sp.atan2(rot3[2,1], rot3[2,2]), sp.atan2(-rot3[2,0], sy), sp.atan2(rot3[1,0], rot3[0,0]))
	else:
		if evaluate:
			return RPY(sp.atan2(-rot3[1,2], rot3[1,1]).evalf(real=True), sp.atan2(-rot3[2,0], sy).evalf(real=True), 0)
		else:
			return RPY(sp.atan2(-rot3[1,2], rot3[1,1]), sp.atan2(-rot3[2,0], sy), 0)


def real_quat_from_matrix(frame):
	tr = frame[0,0] + frame[1,1] + frame[2,2]

	if tr > 0:
		S = sqrt(tr+1.0) * 2 # S=4*qw
		qw = 0.25 * S
		qx = (frame[2,1] - frame[1,2]) / S
		qy = (frame[0,2] - frame[2,0]) / S
		qz = (frame[1,0] - frame[0,1]) / S
	elif frame[0,0] > frame[1,1] and frame[0,0] > frame[2,2]:
		S  = sqrt(1.0 + frame[0,0] - frame[1,1] - frame[2,2]) * 2 # S=4*qx
		qw = (frame[2,1] - frame[1,2]) / S
		qx = 0.25 * S
		qy = (frame[0,1] + frame[1,0]) / S
		qz = (frame[0,2] + frame[2,0]) / S
	elif frame[1,1] > frame[2,2]:
		S  = sqrt(1.0 + frame[1,1] - frame[0,0] - frame[2,2]) * 2 # S=4*qy
		qw = (frame[0,2] - frame[2,0]) / S
		qx = (frame[0,1] + frame[1,0]) / S
		qy = 0.25 * S
		qz = (frame[1,2] + frame[2,1]) / S
	else:
		S  = sqrt(1.0 + frame[2,2] - frame[0,0] - frame[1,1]) * 2 # S=4*qz
		qw = (frame[1,0] - frame[0,1]) / S
		qx = (frame[0,2] + frame[2,0]) / S
		qy = (frame[1,2] + frame[2,1]) / S
		qz = 0.25 * S
	return (qx, qy, qz, qw)


from std_msgs.msg import Header, String, Float64, Bool, Int32
from geometry_msgs.msg import Pose    as PoseMsg
from geometry_msgs.msg import Point   as PointMsg
from geometry_msgs.msg import Vector3 as Vector3Msg
from geometry_msgs.msg import Quaternion as QuaternionMsg
from geometry_msgs.msg import PoseStamped as PoseStampedMsg

def jsDictToJSMsg(js_dict):
	js = JointStateMsg()
	js.header.stamp = rospy.Time.now()
	for joint, state in js_dict.items():
		js.name.append(joint)
		js.position.append(state.position)
		js.velocity.append(state.velocity)
		js.effort.append(state.effort)

	return js

def ros_msg_to_expr(ros_msg):
	t_msg = type(ros_msg)
	if t_msg == int or t_msg == float or t_msg == bool or t_msg == str:
		return ros_msg
	elif t_msg == PObject:
		return pobj_to_expr(ros_msg)
	elif t_msg == SearchObjectMsg:
		return gmm_obj_to_expr(ros_msg)
	elif t_msg == PoseMsg:
		return frame3_quaternion(ros_msg.position.x,
								 ros_msg.position.y,
								 ros_msg.position.z,
								 ros_msg.orientation.x,
								 ros_msg.orientation.y,
								 ros_msg.orientation.z,
								 ros_msg.orientation.w)
	elif t_msg == PointMsg:
		return point3(ros_msg.x, ros_msg.y, ros_msg.z)
	elif t_msg == Vector3Msg:
		return vector3(ros_msg.x, ros_msg.y, ros_msg.z)
	elif t_msg == QuaternionMsg:
		return rotation3_quaternion(ros_msg.x, ros_msg.y, ros_msg.z, ros_msg.w)
	elif t_msg == PoseStampedMsg:
		return StampedData(ros_msg.header.stamp, ros_msg_to_expr(t_msg.pose))
	elif t_msg == list or t_msg == tuple:
		if len(ros_msg) == 36 and type(ros_msg[0]) == float:
			return sp.Matrix([ros_msg[x * 6: x*6 + 6] for x in range(6)]) # Construct covariance matrix
		return [ros_msg_to_expr(x) for x in ros_msg]
	elif t_msg == JointStateMsg:
		return {ros_msg.name[x]: JointState(ros_msg.position[x], ros_msg.velocity[x], ros_msg.effort[x]) for x in range(len(ros_msg.name))}
	elif t_msg == OPGMsg:
		return GaussianPoseComponent(ros_msg.id,
									 ros_msg.weight,
									 ros_msg_to_expr(ros_msg.cov_pose.pose),
									 ros_msg_to_expr(ros_msg.cov_pose.covariance),
									 ros_msg.occluded)
	else:
		for field in dir(ros_msg):
			if field[0] == '_':
				continue
			attr = getattr(ros_msg, field)
			if not callable(attr) and type(attr) != Header:
				setattr(ros_msg, field, ros_msg_to_expr(getattr(ros_msg, field)))
		return ros_msg


def expr_to_rosmsg(expr):
	t = type(expr)

	#try:
	if t == str:
		out = String()
		out.data = expr
	elif t == int:
		out = Int32()
		out.data = expr
	elif t == float:
		out = Float64()
		out.data = expr
	elif t == bool:
		out = Bool()
		out.data = expr
	elif t == list or t == tuple:
		if len(expr) >= 3 and type(expr[0]) == float or type(expr[0]) == int or type(expr[0]) == sp.RealDouble:
			if len(expr) == 3 or (len(expr) == 4 and expr[3] == 0):
				out = Vector3Msg()
				out.x = expr[0]
				out.y = expr[1]
				out.z = expr[2]
			elif len(expr) == 4 and expr[3] == 1:
				out = PointMsg()
				out.x = expr[0]
				out.y = expr[1]
				out.z = expr[2]
		else:
			raise Exception('Can only convert lists of length 3 or 4 to rosmsg, with the contents being numbers! Given list:\n   {}\n   Inner types: {}'.format(str(expr), ', '.join([str(type(x)) for x in expr])))
	elif t is sp.Matrix and expr.ncols() == 1 and (expr[expr.nrows() - 1] == 0 or expr[expr.nrows() - 1] == 0.0):
		out = Vector3Msg()
		out.x = expr[0]
		out.y = expr[1]
		out.z = expr[2]
	elif t is sp.Matrix and expr.ncols() == 1 and (expr[expr.nrows() - 1] == 1 or expr[expr.nrows() - 1] == 1.0):
		out = PointMsg()
		out.x = expr[0]
		out.y = expr[1]
		out.z = expr[2]
	elif t is np.ndarray and (expr.shape == (3,) or (expr.shape == (4,) and expr[3] == 0.0)):
		out = Vector3Msg()
		out.x = expr[0]
		out.y = expr[1]
		out.z = expr[2]
	# elif DLRotation().is_a(expr):
	# 	out = QuaternionMsg()
	# 	out.w = sqrt(1 + expr[0,0] + expr[1,1] + expr[2,2]) * 0.5
	# 	w4 = 4 * out.w
	# 	out.x = (expr[2,1] - expr[1,2]) / w4
	# 	out.y = (expr[0,2] - expr[2,0]) / w4
	# 	out.z = (expr[1,0] - expr[0,1]) / w4
	elif t is sp.Matrix and expr.ncols() == 4 and expr.nrows() == 4:
		out = PoseMsg()
		quat = real_quat_from_matrix(expr)
		out.orientation.w = quat[3]
		out.orientation.x = quat[0]
		out.orientation.y = quat[1]
		out.orientation.z = quat[2]
		out.position.x = expr[0, 3]
		out.position.y = expr[1, 3]
		out.position.z = expr[2, 3]
	elif t == Vector3:
		out = Vector3Msg()
		out.x = expr[0]
		out.y = expr[1]
		out.z = expr[2]
	elif t == Point3:
		out = PointMsg()
		out.x = expr[0]
		out.y = expr[1]
		out.z = expr[2]
	elif t == Frame:
		out = PoseMsg()
		out.orientation.x = expr.quaternion[0]
		out.orientation.y = expr.quaternion[1]
		out.orientation.z = expr.quaternion[2]
		out.orientation.w = expr.quaternion[3]
		out.position.x = expr.position[0]
		out.position.y = expr.position[1]
		out.position.z = expr.position[2]
	else:
		raise Exception('Can not convert {} of type {} to ROS message'.format(str(expr), t))
	# except ROSSerializationException as e:
	# 	if t == list or t == tuple or t == sp.Matrix:
	# 		type_str = 'Outer type {}\n Inner types: [{}]'.format(str(t), ', '.join([str(type(x)) for x in expr]))
	# 	else:
	# 		type_str = 'Type: {}'.format(str(t))
	# 	raise Exception('Conversion failure: {}\n{}'.format(str(e), type_str))

	return out



class Blank:
	def __str__(self):
		return '\n'.join(['{}: {}'.format(field, str(getattr(self, field))) for field in dir(self) if field[0] != '_' and not callable(getattr(self, field))])

	def __deepcopy__(self, memo):
		out = Blank()
		for attrn in [x  for x in dir(self) if x[0] != '_']:
			attr = getattr(self, attrn)
			if isinstance(attr, sp.Basic) or isinstance(attr, sp.Number) or isinstance(attr, sp.Expr) or isinstance(attr, sp.Add) or isinstance(attr, sp.Mul) or isinstance(attr, sp.Min) or isinstance(attr, sp.Max) or isinstance(attr, sp.Matrix):
				setattr(out, attrn, attr)
			else:
				setattr(out, attrn, deepcopy(attr, memo))
		memo[id(self)] = out
		return out


def bb(**kwargs):
	out = Blank()
	for k, v in kwargs.items():
		setattr(out, k, v)
	return out

def decode_obj_shape(name, out):
	if name == 'coke' or name == 'sprite':
		out.radius = 0.034
		out.height = 0.126
		out.presurized = True
		out.mass   = 0.4
	elif name == 'pringle':
		out.radius = 0.037
		out.height = 0.248
		out.mass   = 0.3
	elif name == 'blue_box':
		out.length = 0.079
		out.width  = 0.079
		out.height = 0.038
		out.mass   = 0.3
	elif name == 'blueberry_box':
		out.length = 0.133
		out.width  = 0.133
		out.height = 0.056
		out.mass   = 0.2
	elif name == 'clorox':
		body = bb(name = 'body', radius = 0.37, height=0.131, pose=frame3_rpy(0,0,0, point3(0,0,-0.04)))
		cap  = bb(name = 'cap', radius=0.026, height=0.04, pose=frame3_rpy(0,0,0, point3(0,0, 0.065)))
		out.subObject =[body, cap]
		out.mass   = 1.2
	elif name == 'delight':
		out.length = 0.152
		out.width  = 0.114
		out.height = 0.070
		out.mass   = 1.0
	elif name == 'table':
		out.width  = 1.15
		out.height = 0.84
		out.length = 0.54
		out.mass   = 100.0
	elif name == 'floor':
		out.width  = 20.0
		out.height = 1.0
		out.length = 20.0
	elif name == 'ball':
		out.radius = 0.04
		out.mass   = 0.5
	elif name == 'flower_table':
		out.radius = 0.915 * 0.5
		out.height = 0.725
		out.mass   = 30.0
		out.good_variance = [0.1, 0.1, 0.1, 10]
	elif name == 'green_table':
		out.length = 0.59
		out.width  = 1.17
		out.height = 0.79
		out.mass   = 10.0
		out.good_variance = [0.15, 0.15, 0.15, 10]
	elif name == 'brown_table':
		out.length = 0.61
		out.width  = 1.22
		out.height = 0.585
		out.mass   = 10.0
		out.good_variance = [0.1, 0.1, 0.1, 10]
	elif name == 'grey_box':
		out.length = 0.27
		out.width  = 0.39
		out.height = 0.265
		out.mass   = 0.5
		out.container = True
	elif name == 'sugar':
		out.radius = 0.076 * 0.5
		out.height = 0.171
		out.mass   = 1.0
	elif name == 'green_tea':
		out.length = 0.075
		out.width  = 0.131
		out.height = 0.065
		out.mass   = 0.1
	elif name == 'chai_tea':
		out.length = 0.079
		out.width  = 0.134
		out.height = 0.069
		out.mass   = 0.1
	elif name == 'remote':
		out.length = 0.21
		out.width  = 0.054
		out.height = 0.025
		out.mass   = 0.3
	elif name == 'popcorn':
		out.length = 0.068
		out.width  = 0.15
		out.height = 0.153
		out.mass   = 1.0
	elif name == 'grey_cup' or name == 'blue_cup' or name == 'dark_cup':
		out.radius = 0.089 * 0.5
		out.height = 0.167
		out.mass   = 0.1
	elif name == 'the_post_dvd':
		out.length = 0.19
		out.width  = 0.136
		out.height = 0.015
		out.mass   = 0.2
	elif name == 'cereal':
		out.length = 0.055
		out.width  = 0.19
		out.height = 0.245
		out.mass   = 0.8
	else:
		pass


def pobj_to_expr(msg):
	out = Blank()

	msg.name = ''.join([i for i in msg.name if not i.isdigit()])

	decode_obj_shape(msg.name, out)

	#raise Exception("Unrecognized semantic class '{}'".format(msg.name))

	out.id = '{}{}'.format(msg.name, msg.id)
	out.pose = ros_msg_to_expr(msg.cov_pose.pose)
	out.pose_cov = ros_msg_to_expr(msg.cov_pose.covariance)

	out.concepts = {msg.name}

	return out


def gmm_obj_to_expr(msg):
	out = Blank()

	msg.name = ''.join([i for i in msg.name if not i.isdigit()])

	decode_obj_shape(msg.name, out)

	out.id = '{}{}'.format(msg.name, msg.id)
	out.gmm = ros_msg_to_expr(msg.object_pose_gmm)

	out.concepts = {msg.name}

	return out


def vis_obj_to_expr(msg):
	out = Blank()

	out.id = msg.ns
	out.concepts = {''.join([i for i in msg.ns if not i.isdigit()])}
	out.pose = ros_msg_to_expr(msg.pose)
	out.mass = 0.0
	if msg.type == MarkerMsg.CUBE:
		out.length = msg.scale.x
		out.width  = msg.scale.y
		out.height = msg.scale.z
	elif msg.type == MarkerMsg.SPHERE:
		out.radius = msg.scale.x
	elif msg.type == MarkerMsg.CYLINDER:
		out.radius = msg.scale.x * 0.5
		out.height = msg.scale.z
	else:
		raise Exception('Can not convert visual marker of type {}'.format(msg.type))
	return out