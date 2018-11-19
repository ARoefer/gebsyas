from giskardpy.symengine_wrappers import sp, point3, vector3, frame3_quaternion,

def to_sym(tuple_name):
    return sp.Symbol('__'.join(tuple_name))

class ListStructure(list):
    def __init__(self, name, *args):
        super(ListStructure, self).__init__(args)
        self.name = name
        self.free_symbols = set()
        for x in self.__list:
            self.free_symbols = self.free_symbols.union(x.free_symbols) if hasattr(v, 'free_symbols') else self.free_symbols

    def subs(self, sdict):
        return ListStructure(self.name, *[x.subs(sdict) for x in self.__list if hasattr(x, 'subs') else x])


class Structure(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.free_symbols = set()
        self.edges = set(kwargs.keys())
        for n, v in kwargs:
            setattr(self, n, v)
            self.free_symbols = self.free_symbols.union(v.free_symbols) if hasattr(v, 'free_symbols') else self.free_symbols

    def __len__(self):
        return len(self.edges)

    def __iter__(self):
        return iter({n: getattr(self, n) for n in self.edges})

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

    def subs(self, sdict):
        return Structure(self.name, **{n: (v.subs(sdict) if hasattr(v, 'subs') else v) for n, v in [(k, getattr(self, k)) for k in self.edges]})


    def refresh(self, numeric, state):


from std_msgs.msg import Header, String, Float64, Bool, Int32
from geometry_msgs.msg import Pose    as PoseMsg
from geometry_msgs.msg import Point   as PointMsg
from geometry_msgs.msg import Vector3 as Vector3Msg
from geometry_msgs.msg import Quaternion as QuaternionMsg
from geometry_msgs.msg import PoseStamped as PoseStampedMsg



def ros_msg_to_ks(ros_msg, name, state={}):
    """
    :param name: Name of the object currently being converted
    :type  name: tuple
    :param state: Table of all current variable assignments
    :type  state: dict
    :rtype: Structure, ListStructure, SymEngine Objects, bool, str
    """
    t_msg = type(ros_msg)
    if t_msg == int or t_msg == float:
        sym = to_sym(name)
        state[sym] = t_msg
        return sym
    elif t_msg == bool or t_msg == str:
        return ros_msg
    elif t_msg == PoseMsg:
        syms = []
        for x in ['x', 'y', 'z']:
            p = to_sym(name + ('position', x))
            state[p] = getattr(ros_msg.position, x)
            syms.append(p)
        for x in ['x', 'y', 'z', 'w']:
            p = to_sym(name + ('orientation', x))
            state[p] = getattr(ros_msg.orientation, x)
            syms.append(p)
        return frame3_quaternion(*syms)
    elif t_msg == PointMsg:
        syms = []
        for x in ['x', 'y', 'z']:
            p = to_sym(name + (x,))
            state[p] = getattr(ros_msg, x)
            syms.append(p)
        return point3(*syms)
    elif t_msg == Vector3Msg:
        syms = []
        for x in ['x', 'y', 'z']:
            p = to_sym(name + (x,))
            state[p] = getattr(ros_msg, x)
            syms.append(p)
        return vector3(*syms)
    elif t_msg == QuaternionMsg:
        syms = []
        for x in ['x', 'y', 'z', 'w']:
            p = to_sym(name + (x,))
            state[p] = getattr(ros_msg, x)
            syms.append(p)
        return rotation3_quaternion(*syms)
    elif t_msg == list or t_msg == tuple:
        return ListStructure(name, *[ros_msg_to_ks(ros_msg[x], name + (str(x),), state) for x in range(len(ros_msg))])
    else:
        fields = {}
        for field in dir(ros_msg):
            if field[0] == '_':
                continue
            attr = getattr(ros_msg, field)
            if not callable(attr) and type(attr) != Header:
                fields[field] = ros_msg_to_ks(getattr(ros_msg, field), name + (field,), state)
        return Structure(name, **fields)