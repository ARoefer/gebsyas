from collections import namedtuple
from giskardpy.symengine_wrappers import *
from sensor_msgs.msg import JointState as JointStateMsg
from giskard_affordances.msg import ProbabilisticObject as PObject
import symengine as sp
import rospy

StampedData = namedtuple('StampedData', ['stamp', 'data'])

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

from std_msgs.msg import Header, String, Float64, Bool, Int32
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseStamped

JointState = namedtuple('JointState', ['position', 'velocity', 'effort'])

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
	elif t_msg == Pose:
		return frame3_quaternion(ros_msg.orientation.x,
								 ros_msg.orientation.y,
								 ros_msg.orientation.z,
								 ros_msg.orientation.w,
								 point3(ros_msg.position.x, ros_msg.position.y, ros_msg.position.z))
	elif t_msg == Point:
		return point3(ros_msg.x, ros_msg.y, ros_msg.z)
	elif t_msg == Vector3:
		return vec3(ros_msg.x, ros_msg.y, ros_msg.z)
	elif t_msg == Quaternion:
		return rotation3_quaternion(ros_msg.x, ros_msg.y, ros_msg.z, ros_msg.w)
	elif t_msg == PoseStamped:
		return StampedData(ros_msg.header.stamp, ros_msg_to_expr(t_msg.pose))
	elif t_msg == list:
		return [ros_msg_to_expr(x) for x in ros_msg]
	elif t_msg == JointStateMsg:
		return dict([(ros_msg.name[x], JointState(ros_msg.position[x], ros_msg.velocity[x], ros_msg.effort[x])) for x in range(len(ros_msg.name))])
	else:
		for field in dir(ros_msg):
			attr = getattr(ros_msg, field)
			if field[0] != '_' and not callable(attr) and type(attr) != Header:
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
		if len(expr) >= 3 and type(expr[0]) == float or type(expr[0]) == int:
			if len(expr) == 3 or (len(expr) == 4 and expr[3] == 0):
				out = Vector3()
				out.x = expr[0]
				out.y = expr[1]
				out.z = expr[2]
			elif len(expr) == 4 and expr[3] == 1:
				out = Point()
				out.x = expr[0]
				out.y = expr[1]
				out.z = expr[2]
	elif t is sp.Matrix and expr.ncols() == 1 and expr[expr.nrows() - 1] == 0:
		out = Vector3()
		out.x = expr[0]
		out.y = expr[1]
		out.z = expr[2]
	elif t is sp.Matrix and expr.ncols() == 1 and expr[expr.nrows() - 1] == 1:
		out = Point()
		out.x = expr[0]
		out.y = expr[1]
		out.z = expr[2]
	# elif DLRotation().is_a(expr):
	# 	out = Quaternion()
	# 	out.w = sqrt(1 + expr[0,0] + expr[1,1] + expr[2,2]) * 0.5
	# 	w4 = 4 * out.w
	# 	out.x = (expr[2,1] - expr[1,2]) / w4
	# 	out.y = (expr[0,2] - expr[2,0]) / w4
	# 	out.z = (expr[1,0] - expr[0,1]) / w4
	elif t is sp.Matrix and expr.ncols() == 4 and expr.nrows() == 4:
		out = Pose()
		out.orientation.w = sp.sqrt(1 + expr[0,0] + expr[1,1] + expr[2,2]) * 0.5
		w4 = 4 * out.orientation.w
		out.orientation.x = (expr[2,1] - expr[1,2]) / w4
		out.orientation.y = (expr[0,2] - expr[2,0]) / w4
		out.orientation.z = (expr[1,0] - expr[0,1]) / w4
		out.position.x = expr[0, 3]
		out.position.y = expr[1, 3]
		out.position.z = expr[2, 3]
	else:
		raise Exception('Can not convert {} of type {} to ROS message'.format(str(expr), t))
	return out



class Blank:
	def __str__(self):
		return '\n'.join(['{}: {}'.format(field, str(getattr(self, field))) for field in dir(self) if field[0] != '_' and not callable(getattr(self, field))])

def bb(**kwargs):
	out = Blank()
	for k, v in kwars.items():
		setattr(out, k, v)
	return out

def pobj_to_expr(msg):
	out = Blank()
	if msg.semantic_class == 'cube':
		out.length = msg.dimensions.x
		out.width  = msg.dimensions.y
		out.height = msg.dimensions.z
	elif msg.semantic_class == 'sphere':
		out.radius = msg.dimensions.x * 0.5
	elif msg.semantic_class == 'cylinder':
		out.radius = msg.dimensions.x * 0.5
		out.height = msg.dimensions.z
	elif msg.semantic_class == 'coke':
		out.radius = 0.034
		out.height = 0.126
		out.presurized = True
	elif msg.semantic_class == 'pringles':
		out.radius = 0.078
		out.height = 0.248
	elif msg.semantic_class == 'blue_box':
		out.length = 0.079
		out.width  = 0.079
		out.height = 0.038
	elif msg.semantic_class == 'blue_cup':
		out.radius = 0.045
		out.height = 0.146
		out.subObject = bb(name='rim', radius = 0.049, rotation_axis = vec3(0,0,1), pose = frame3_rpy(0,0,0, point3(0,0, -0.073)))
	elif msg.semantic_class == 'blueberry_box':
		out.length = 0.133
		out.width  = 0.133
		out.height = 0.056
	elif msg.semantic_class == 'clorox':
		body = bb(name = 'body', radius = 0.37, height=0.131, pose=frame3_rpy(0,0,0, point3(0,0,-0.04)))
		cap  = bb(name = 'cap', radius=0.026, height=0.04, pose=frame3_rpy(0,0,0, point3(0,0, 0.065)))
		out.subObject =[body, cap]
	elif msg.semantic_class == 'delight':
		out.length = 0.152
		out.width  = 0.114
		out.height = 0.070
	else:
		raise Exception("Unrecognized semantic class '{}'".format(msg.semantic_class))

	out.id = msg.id
	out.pose = ros_msg_to_expr(msg.pose)
	out.semantic_class = msg.semantic_class
	out.probability_class = msg.probability_class
	out.probability_position = msg.probability_position
	out.probability_rotation = msg.probability_rotation

	return out
