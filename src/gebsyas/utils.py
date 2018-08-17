import os
import symengine as sp
import rospy
import yaml

from collections import namedtuple
from gebsyas.data_structures import StampedData, JointState
from giskardpy.symengine_wrappers import *
from sensor_msgs.msg import JointState as JointStateMsg
from gop_gebsyas_msgs.msg import ProbObject as PObject
from iai_bullet_sim.utils import Frame, Vector3
from copy import deepcopy

pi = 3.14159265359

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
		return vec3(ros_msg.x, ros_msg.y, ros_msg.z)
	elif t_msg == QuaternionMsg:
		return rotation3_quaternion(ros_msg.x, ros_msg.y, ros_msg.z, ros_msg.w)
	elif t_msg == PoseStampedMsg:
		return StampedData(ros_msg.header.stamp, ros_msg_to_expr(t_msg.pose))
	elif t_msg == list:
		if len(ros_msg) == 36 and type(ros_msg[0]) == float:
			return sp.Matrix([t_msg[x * 6: x*6 + 6] for x in range(6)]) # Construct covariance matrix
		return [ros_msg_to_expr(x) for x in ros_msg]
	elif t_msg == JointStateMsg:
		return {ros_msg.name[x]: JointState(ros_msg.position[x], ros_msg.velocity[x], ros_msg.effort[x]) for x in range(len(ros_msg.name))}
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
	# elif DLRotation().is_a(expr):
	# 	out = QuaternionMsg()
	# 	out.w = sqrt(1 + expr[0,0] + expr[1,1] + expr[2,2]) * 0.5
	# 	w4 = 4 * out.w
	# 	out.x = (expr[2,1] - expr[1,2]) / w4
	# 	out.y = (expr[0,2] - expr[2,0]) / w4
	# 	out.z = (expr[1,0] - expr[0,1]) / w4
	elif t is sp.Matrix and expr.ncols() == 4 and expr.nrows() == 4:
		out = PoseMsg()
		quat = quaternion_from_matrix(expr)
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

def pobj_to_expr(msg):
	out = Blank()
	
	msg.name = ''.join([i for i in msg.name if not i.isdigit()])

	if msg.name == 'coke':
		out.radius = 0.034
		out.height = 0.126
		out.presurized = True
		out.mass   = 0.4
	elif msg.name == 'pringle':
		out.radius = 0.039
		out.height = 0.248
		out.mass   = 0.3
	elif msg.name == 'blue_box':
		out.length = 0.079
		out.width  = 0.079
		out.height = 0.038
		out.mass   = 0.3
	elif msg.name == 'blue_cup':
		out.radius = 0.045
		out.height = 0.146
		out.subObject = bb(name='rim', radius = 0.049, rotation_axis = vec3(0,0,1), pose = frame3_rpy(0,0,0, point3(0,0, -0.073)))
		out.mass   = 0.1
	elif msg.name == 'blueberry_box':
		out.length = 0.133
		out.width  = 0.133
		out.height = 0.056
		out.mass   = 0.2
	elif msg.name == 'clorox':
		body = bb(name = 'body', radius = 0.37, height=0.131, pose=frame3_rpy(0,0,0, point3(0,0,-0.04)))
		cap  = bb(name = 'cap', radius=0.026, height=0.04, pose=frame3_rpy(0,0,0, point3(0,0, 0.065)))
		out.subObject =[body, cap]
		out.mass   = 1.2
	elif msg.name == 'delight':
		out.length = 0.152
		out.width  = 0.114
		out.height = 0.070
		out.mass   = 1.0
	elif msg.name == 'table':
		out.width  = 1.15
		out.height = 0.84
		out.length = 0.54
		out.mass   = 100.0
	elif msg.name == 'floor':
		out.width  = 20.0
		out.height = 1.0
		out.length = 20.0
	elif msg.name == 'ball':
		out.radius = 0.04
		out.mass   = 0.5
	else:
		pass	
	#raise Exception("Unrecognized semantic class '{}'".format(msg.name))

	out.id = '{}{}'.format(msg.name, msg.id)
	out.pose = ros_msg_to_expr(msg.cov_pose.pose)
	out.pose_cov = ros_msg_to_expr(msg.cov_pose.covariance)
	out.concepts = {msg.name}

	return out
