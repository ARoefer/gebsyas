from giskardpy.symengine_wrappers import sp, axis_angle_from_matrix, frame3_axis_angle, point3, vector3, norm
from gebsyas.core.dl_types import DLVector, \
                                  DLPoint, \
                                  DLRotation, \
                                  DLTranslation, \
                                  DLTransform
import numpy as np 

def to_sym(tuple_name):
    if len(tuple_name) == 0:
        raise Exception('Can not convert empty name tuples to symbol.')
    return sp.Symbol('__'.join(tuple_name))

class ListStructure(list):
    def __init__(self, name, rf, *args):
        super(ListStructure, self).__init__(args)
        self.name = name
        self.free_symbols = set()
        self.rf = rf
        for x in self:
            self.free_symbols = self.free_symbols.union(x.free_symbols) if hasattr(x, 'free_symbols') else self.free_symbols

    def subs(self, sdict):
        return ListStructure(self.name, self.rf, *[x.subs(sdict) if hasattr(x, 'subs') else x for x in self])


class Structure(object):
    def __init__(self, name, rf, **kwargs):
        self.name = name
        self.free_symbols = set()
        self.edges = set(kwargs.keys())
        self.rf = rf
        for n, v in kwargs.items():
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
        return Structure(self.name, self.rf, **{n: (v.subs(sdict) if hasattr(v, 'subs') else v) for n, v in [(k, getattr(self, k)) for k in self.edges]})

    def to_flat_dict(self, prefix=''):
        out = {}
        for e in self.edges:
            o = getattr(self, e)
            if type(o) == Structure:
                out.update(o.to_flat_dict('{}{}/'.format(prefix, e)))
            else:
                out['{}{}'.format(prefix, e)] = o
        return out

def f_blank(o, state):
    pass

def ks_from_obj(obj, name, state={}):
    """
    :param name: Name of the object currently being converted
    :type  name: tuple
    :param state: Table of all current variable assignments
    :type  state: dict
    :rtype: Structure, ListStructure, SymEngine Objects, bool, str
    """
    t_msg = type(obj)
    if t_msg == int or t_msg == float:
        sym = to_sym(name)
        state[sym] = obj
        def rf(o, state):
            state[sym] = o
        return sym, rf
    elif t_msg == bool or t_msg == str:
        return obj, f_blank
    elif t_msg is sp.Matrix:
        if DLPoint.is_a(obj) or DLVector.is_a(obj):
            syms = [to_sym(name + (x,)) for x in ['x', 'y', 'z']]
            for x in range(3):
                state[syms[x]] = obj[x]
            def rf(o, state):
                state[syms[0]] = obj[0]
                state[syms[1]] = obj[1]
                state[syms[2]] = obj[2]
            if DLVector.is_a(obj):
                return vector3(*syms)
            else:
                return point3(*syms)
        elif DLTransform.is_a(obj):
            syms = [to_sym(name + (x,)) for x in ['ax', 'ay', 'az', 'x', 'y', 'z']]
            def rf(o, state):
                axis, angle = axis_angle_from_matrix(o)
                state[syms[0]] = axis[0] * angle
                state[syms[1]] = axis[1] * angle
                state[syms[2]] = axis[2] * angle
                state[syms[3]] = o[0, 3]
                state[syms[4]] = o[1, 3]
                state[syms[5]] = o[2, 3]

            rf(obj, state)
            axis = vector3(*syms[:3])
            return frame3_axis_angle(axis / (norm(axis) + 1e-7), norm(axis), syms[3:]), rf
        else:
            raise Exception('Can not automatically convert matrix')
    elif t_msg == list or t_msg == tuple:
        objs, rfs = zip(*[ks_from_obj(obj[x], name + (str(x),), state) for x in range(len(obj))])

        def rf(o, state):
            for x in range(len(rfs)):
                rfs[x](o[x], state)

        return ListStructure(name, rf, *objs), rf
    else:
        fields = {}
        rfs = {}
        for field in dir(obj):
            if field[0] == '_':
                continue
            attr = getattr(obj, field)
            if not callable(attr):
                fields[field], rfs[field] = ks_from_obj(getattr(obj, field), name + (field,), state)

        def rf(o, state):
            for field, f in rfs.items():
                f(getattr(o, field), state)

        return Structure(name, rf, **fields), rf