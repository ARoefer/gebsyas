import numpy as np

from iai_bullet_sim.utils import Vector3, Point3, Quaternion, Frame

from gebsyas.core.subs_ds import ks_from_obj
from giskardpy.symengine_wrappers import sp, \
                                         point3, \
                                         vector3, \
                                         rotation3_quaternion, \
                                         frame3_quaternion

from gebsyas.utils import real_quat_from_matrix

from gebsyas.data_structures import StampedData, JointState, GaussianPoseComponent

from std_msgs.msg      import Header, String, Float64, Bool, Int32
from geometry_msgs.msg import Pose    as PoseMsg
from geometry_msgs.msg import Point   as PointMsg
from geometry_msgs.msg import Vector3 as Vector3Msg
from geometry_msgs.msg import Quaternion as QuaternionMsg
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from sensor_msgs.msg   import JointState as JointStateMsg


def ks_from_ros_msg(ros_msg, name, state={}):
    """
    :param name: Name of the object currently being converted
    :type  name: tuple
    :param state: Table of all current variable assignments
    :type  state: dict
    :rtype: Structure, ListStructure, SymEngine Objects, bool, str
    """
    t_msg = type(ros_msg)
    if t_msg == PoseMsg:
        syms = []
        for x in ['x', 'y', 'z']:
            p = to_sym(name + ('position', x))
            state[p] = getattr(ros_msg.position, x)
            syms.append(p)
        for x in ['x', 'y', 'z', 'w']:
            p = to_sym(name + ('orientation', x))
            state[p] = getattr(ros_msg.orientation, x)
            syms.append(p)

        def rf(o, state):
            state[syms[0]] = o.position.x
            state[syms[1]] = o.position.y
            state[syms[2]] = o.position.z
            state[syms[3]] = o.orientation.x
            state[syms[4]] = o.orientation.y
            state[syms[5]] = o.orientation.z
            state[syms[6]] = o.orientation.w
        return frame3_quaternion(*syms), rf
    elif t_msg == PointMsg:
        syms = []
        for x in ['x', 'y', 'z']:
            p = to_sym(name + (x,))
            state[p] = getattr(ros_msg, x)
            syms.append(p)

        def rf(o, state):
            state[syms[0]] = o.x
            state[syms[1]] = o.y
            state[syms[2]] = o.z

        return point3(*syms), rf
    elif t_msg == Vector3Msg:
        syms = []
        for x in ['x', 'y', 'z']:
            p = to_sym(name + (x,))
            state[p] = getattr(ros_msg, x)
            syms.append(p)

        def rf(o, state):
            state[syms[0]] = o.x
            state[syms[1]] = o.y
            state[syms[2]] = o.z

        return vector3(*syms), rf
    elif t_msg == QuaternionMsg:
        syms = []
        for x in ['x', 'y', 'z', 'w']:
            p = to_sym(name + (x,))
            state[p] = getattr(ros_msg, x)
            syms.append(p)

        def rf(o, state):
            state[syms[0]] = o.x
            state[syms[1]] = o.y
            state[syms[2]] = o.z
            state[syms[3]] = o.w

        return rotation3_quaternion(*syms), rf
    else:
        return ks_from_obj(obj, name, state)


def ks_to_rosmsg(expr):
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
    #   out = QuaternionMsg()
    #   out.w = sqrt(1 + expr[0,0] + expr[1,1] + expr[2,2]) * 0.5
    #   w4 = 4 * out.w
    #   out.x = (expr[2,1] - expr[1,2]) / w4
    #   out.y = (expr[0,2] - expr[2,0]) / w4
    #   out.z = (expr[1,0] - expr[0,1]) / w4
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
    #   if t == list or t == tuple or t == sp.Matrix:
    #       type_str = 'Outer type {}\n Inner types: [{}]'.format(str(t), ', '.join([str(type(x)) for x in expr]))
    #   else:
    #       type_str = 'Type: {}'.format(str(t))
    #   raise Exception('Conversion failure: {}\n{}'.format(str(e), type_str))

    return out