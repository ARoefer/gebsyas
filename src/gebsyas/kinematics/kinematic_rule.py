from giskardpy.symengine_wrappers import *
from giskardpy.qp_problem_builder import HardConstraint as HC
from gebsyas.core.dl_types import DLIded, DLPhysicalThing
from gebsyas.core.subs_ds import to_sym
from gebsyas.kinematics.kinematic_state import LPNode
from math import pi, ceil, sin, cos, atan2, asin, acos

STEPS_PER_2PI = 32

def frame_from_axis(axis, point=point3(0,0,0)):
    axis = axis / norm(axis)
    if abs(axis[2]) > 0.998:
        return translation3(*point) 
    else:
        pitch = acos(axis[2])
        yaw   = atan2(axis[1], axis[0])
        return frame3_rpy(0, pitch, yaw, point)

def fixed_position(posA, posB=point3(0,0,0), tolerance=0.0):
    d = norm(posA - posB)
    return [HC(-d, tolerance - d, d)]

def prismatic_joint(pos, a=point3(0,0,0), axis=vector3(1,0,0), tolerance=0.0):
    d  = norm(cross(a - pos, axis))
    return [HC(-d, tolerance - d, d)]

def leash_joint(posA, posB=point3(0,0,0), length=1):
    d = norm(posA - posB)
    return [HC(-d, length - d, d)]


def find_obj_ids_from_so(so_set, dstate):
    """Takes a set of symbolic objects and retrieves the highest level of ided objects from it.
    :param so_set: Set of symbolic objects.
    :type so_set:  set
    :param dstate: Data state the symbolic objects were defined on
    :type dstate:  gebsyas.core.DataState
    :return: Dictionary mapping object paths to objects
    :rtype: dict
    """
    symbols = set()
    for e in so_set:
        symbols |= e.free_symbols

    objs = {}
    for s in symbols:
        path = tuple(str(s).split('__') for s in symbols)
        for x in range(len(path)):
            sp = path[:x]
            if sp in obj:
                break

            o  = dstate[sp]
            if DLIded.is_a(o) and DLPhysicalThing.is_a(o):
                types[sp] = o
                break

    return objs


class KinematicRule(object):
    def __init__(self, name):
        self.name = name

    def get_boundary_constraints(self):
        raise NotImplementedError

    def apply(self, kstate):
        raise NotImplementedError   

    def remove(self, kstate):
        raise NotImplementedError

    def render(self, kstate, visualizer, namespace='kr', color=(1,1,1,1)):
        pass

    def __str__(self):
        return 'KR({})'.format(self.name)


class LowerPair(KinematicRule):
    def __init__(self, name, d_obj, **kwargs):
        super(LowerPair, self).__init__(name)
        self.d_obj = d_obj
        self.__static_set = set(kwargs.values())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def apply(self, kstate):
        # res_set = find_obj_ids_from_so(self.__static_set, kstate.data_state).items()
        # if len(res_set) != 1:
        #     raise Exception('Set should only contain one object, but contains {}:\n  {}'.format(len(objs), '\n  '.join([o.id for o in objs])))

        self._apply(kstate)

        if self.d_obj.id in kstate.lp_tree: 
            for lp in kstate.lp_tree[self.d_obj.id].children.values():
                lp.apply(kstate)


    def _apply(self, kstate):
        raise NotImplementedError

    def remove(self, kstate):
        self.d_obj.pose = self.d_obj.pose.subs(kstate.data_state.value_table)
        if self.d_obj.id in kstate.lp_tree[self.d_obj.id]:
            for lp in kstate.lp_tree[self.d_obj.id].children.values():
                lp.apply(kstate)

    def inlier_function(self, kstate, observed):
        """Returns a projection into the LP's configuration state and 
        a linear and angular distance to the next pose matching the 
        configuration perfectly.
        """
        raise NotImplementedError

# Lower Pair Joints
class StaticPair(LowerPair):
    def __init__(self, obj, frame):
        super(StaticPair, self).__init__('Static', obj, frame=frame)

    def _apply(self, kstate):
        self.d_obj.pose = self.frame * (inverse_frame(self.frame.subs(kstate.data_state.value_table)) * self.d_obj.pose)

    def remove(self, kstate):
        super(StaticPair, self).remove(kstate)

    def inlier_function(self, kstate, observed):
        m_pose = self.d_obj.pose.subs(kstate.data_state.value_table)
        return sp.Matrix([norm(pos_of(m_pose) - pos_of(observed)), 
                          rotation_distance(rot_of(m_pose), rot_of(observed))])

# Lower Pair Joints
class HingePair(LowerPair):
    def __init__(self, obj, point, axis, llower=None, lupper=None):
        super(HingePair, self).__init__('Hinge', obj, point=point, axis=axis)
        self.lower = llower
        self.upper = lupper
        self.sym_alpha = to_sym((self.d_obj.id, 'hinge', 'alpha'))
        self.radius = 0

    def _apply(self, kstate):
        t_hinge = frame3_axis_angle(self.axis, self.sym_alpha, self.point)
        kstate.data_state.value_table[self.sym_alpha] = 0.0

        if self.lower != None or self.upper != None:
            lower = self.lower if self.lower != None else -10e9
            upper = self.upper if self.upper != None else  10e9
            kstate.hc_set[str(self.sym_alpha)] = HC(lower - self.sym_alpha, 
                                                    upper - self.sym_alpha, self.sym_alpha)

        self.radius = norm(cross(self.axis.subs(kstate.data_state.value_table), 
                                 pos_of(self.d_obj.pose) - self.point.subs(kstate.data_state.value_table)))
        self.d_obj.pose = t_hinge * (inverse_frame(t_hinge.subs(kstate.data_state.value_table)) * self.d_obj.pose)


    def remove(self, kstate):
        super(HingePair, self).remove(kstate)
        if self.lower != None or self.upper != None:
            del kstate.hc_set[str(self.sym_alpha)]
        del kstate.data_state.value_table[self.sym_alpha]

    def inlier_function(self, kstate, observed):
        axis   = self.axis.subs(kstate.data_state.value_table)
        point  = self.point.subs(kstate.data_state.value_table)
        m_pose = self.d_obj.pose.subs(kstate.data_state.value_table)
        
        rel_p  = pos_of(observed) - point
        p_dist = dot(axis, rel_p)
        r_dist = self.radius - norm(cross(axis, rel_p))

        return sp.Matrix([p_dist, r_dist, rotation_distance(rot_of(m_pose), rot_of(observed))])

    def render(self, kstate, visualizer, namespace='kr', color=(1,1,1,1), radius=0.5):
        axis  = self.axis.subs(kstate.data_state.value_table)
        point = self.point.subs(kstate.data_state.value_table)
        frame = frame_from_axis(axis, point)

        visualizer.draw_vector(namespace, point - 0.5 * axis, axis, color[0], color[1], color[2], color[3])

        if self.lower != None and self.upper != None:
            lower = self.lower if not hasattr(self.lower, 'free_symbols') else self.lower.subs(kstate.data_state.value_table)
            upper = self.upper if not hasattr(self.upper, 'free_symbols') else self.upper.subs(kstate.data_state.value_table)

            width = upper - lower 
            steps = int(ceil(STEPS_PER_2PI * radius * ((0.5 * width) / pi)))
            ssize = width / steps

            points = [point3(cos(x * ssize) * radius, sin(x * ssize) * radius, 0) for x in range(steps + 1)]
            visualizer.draw_strip(namespace, frame, 0.02, points, color[0], color[1], color[2], color[3])
        else:
            visualizer.draw_cylinder(namespace, frame, 0.02, 0.2, color[0], color[1], color[2], color[3])


# Lower Pair Joints
class PrismaticPair(LowerPair):
    def __init__(self, obj, point, axis, llower=None, lupper=None):
        super(PrismaticPair, self).__init__('Prismatic', obj, point=point, axis=axis)
        self.lower = llower
        self.upper = lupper
        self.sym_alpha = to_sym((self.d_obj.id, 'prismatic', 'alpha'))

    def _apply(self, kstate):
        t_prismatic = translation3(*(self.point + self.axis * self.sym_alpha))
        kstate.data_state.value_table[self.sym_alpha] = 0.0
        
        if self.lower != None or self.upper != None:
            lower = self.lower if self.lower != None else -10e9
            upper = self.upper if self.upper != None else  10e9
            kstate.hc_set[str(self.sym_alpha)] = HC(lower - self.sym_alpha, 
                                                    upper - self.sym_alpha, self.sym_alpha)

        self.d_obj.pose = t_prismatic * (inverse_frame(t_prismatic.subs(kstate.data_state.value_table)) * self.d_obj.pose)

    def remove(self, kstate):
        super(PrismaticPair, self).remove(kstate)
        if self.lower != None or self.upper != None:
            del kstate.hc_set[str(self.sym_alpha)]
        del kstate.data_state.value_table[self.sym_alpha]

    def inlier_function(self, kstate, observed):
        axis   = self.axis.subs(kstate.data_state.value_table)
        point  = self.point.subs(kstate.data_state.value_table)
        m_pose = self.d_obj.pose.subs(kstate.data_state.value_table)
        
        rel_p  = pos_of(observed) - point
        lower  = self.lower.subs(kstate.data_state.value_table) if self.lower != None else -1e9
        upper  = self.upper.subs(kstate.data_state.value_table) if self.upper != None else -1e9
        p_dist = dot(axis, rel_p)
        p_dist = max(min(p_dist - lower, 0), p_dist - upper)
        r_dist = norm(cross(axis, rel_p))

        return sp.Matrix([p_dist, r_dist, rotation_distance(rot_of(m_pose), rot_of(observed))])

    def render(self, kstate, visualizer, namespace='kr', color=(1,1,1,1), radius=0.2):
        axis  = self.axis.subs(kstate.data_state.value_table)
        point = self.point.subs(kstate.data_state.value_table)

        frame = frame_from_axis(axis, point)
        
        limits = []
        lower  = -0.5
        if self.lower != None:
            lower = self.lower if not hasattr(self.lower, 'free_symbols') else self.lower.subs(kstate.data_state.value_table)
            limits.extend([(lower * unitZ - radius * unitX), 
                           (lower * unitZ + radius * unitX)])
        upper = 0.5
        if self.upper != None:
            upper = self.upper if not hasattr(self.upper, 'free_symbols') else self.upper.subs(kstate.data_state.value_table)
            limits.extend([(upper * unitZ - radius * unitX), 
                           (upper * unitZ + radius * unitX)])

        visualizer.draw_vector(namespace, point + lower * axis, (upper - lower) * axis, color[0], color[1], color[2], color[3])
        if len(limits) > 0:
            visualizer.draw_lines(namespace, frame, 0.02, limits)


class ScrewPair(LowerPair):
    def __init__(self, obj, point, axis, mpr, llower=None, lupper=None):
        super(ScrewPair, self).__init__('Screw', obj, point=point, axis=axis, mpp=(mpr / (2 * pi)))       
        self.lower = llower / self.mpp
        self.upper = lupper / self.mpp
        self.sym_alpha = to_sym((self.d_obj.id, 'screw', 'alpha'))

    def _apply(self, kstate):
        t_screw = frame3_axis_angle(self.axis, self.sym_alpha, self.point + self.axis * self.mpp * self.sym_alpha)
        kstate.data_state.value_table[self.sym_alpha] = 0.0
        
        if self.lower != None or self.upper != None:
            lower = self.lower if self.lower != None else -10e9
            upper = self.upper if self.upper != None else  10e9
            kstate.hc_set[str(self.sym_alpha)] = HC(lower - self.sym_alpha, 
                                                    upper - self.sym_alpha, self.sym_alpha)

        self.d_obj.pose = t_screw * (inverse_frame(t_screw.subs(kstate.data_state.value_table)) * self.d_obj.pose)

    def remove(self, kstate):
        super(ScrewPair, self).remove(kstate)
        if self.lower != None or self.upper != None:
            del kstate.hc_set[str(self.sym_alpha)]
        del kstate.data_state.value_table[self.sym_alpha]

    def inlier_function(self, kstate, observed):
        axis   = self.axis.subs(kstate.data_state.value_table)
        point  = self.point.subs(kstate.data_state.value_table)
        m_pose = self.d_obj.pose.subs(kstate.data_state.value_table)
        
        rel_p  = pos_of(observed) - point
        lower  = self.lower.subs(kstate.data_state.value_table) if self.lower != None else -1e9
        upper  = self.upper.subs(kstate.data_state.value_table) if self.upper != None else -1e9
        p_dist = dot(axis, rel_p)
        p_dist = max(min(p_dist - lower, 0), p_dist - upper)
        r_dist = norm(cross(axis, rel_p))

        return sp.Matrix([p_dist, r_dist, rotation_distance(rot_of(m_pose), rot_of(observed))])

    def render(self, kstate, visualizer, namespace='kr', color=(1,1,1,1), radius=0.5):
        axis  = self.axis.subs(kstate.data_state.value_table)
        point = self.point.subs(kstate.data_state.value_table)

        frame = frame_from_axis(axis, point)
        
        limits = []
        lower  = -0.5
        if self.lower != None:
            lower = self.lower * self.mpp if not hasattr(self.lower, 'free_symbols') else self.lower.subs(kstate.data_state.value_table) * self.mpp
            limits.extend([(lower * unitZ - radius * unitX), 
                           (lower * unitZ + radius * unitX)])
        upper = 0.5
        if self.upper != None:
            upper = self.upper * self.mpp if not hasattr(self.upper, 'free_symbols') else self.upper.subs(kstate.data_state.value_table) * self.mpp
            limits.extend([(upper * unitZ - radius * unitX), 
                           (upper * unitZ + radius * unitX)])

        visualizer.draw_vector(namespace, point + lower * axis, (upper - lower) * axis, color[0], color[1], color[2], color[3])

        width  = upper / self.mpp - lower / self.mpp
        steps  = int(ceil(STEPS_PER_2PI * radius * ((0.5 * width) / pi)))
        ssize  = width / steps

        points = [point3(cos(x * ssize) * radius, sin(x * ssize) * radius, lower + x * ssize * self.mpp) for x in range(steps + 1)]
        visualizer.draw_strip(namespace, frame, 0.02, points, color[0], color[1], color[2], color[3])

        if len(limits) > 0:
            visualizer.draw_lines(namespace, frame, 0.02, limits)


class CylindricalPair(LowerPair):
    def __init__(self, obj, point, axis, t_llower=None, t_lupper=None, r_llower=None, r_lupper=None):
        super(CylindricalPair, self).__init__('Cylindrical', obj, point=point, axis=axis)
        self.t_lower = t_llower
        self.t_upper = t_lupper
        self.r_lower = r_llower
        self.r_upper = r_lupper
        self.sym_alpha = to_sym((self.d_obj.id, 'cylindrical', 'alpha'))
        self.sym_beta  = to_sym((self.d_obj.id, 'cylindrical', 'beta'))

    def _apply(self, kstate):
        t_cylindrical = frame3_axis_angle(self.axis, self.sym_beta, self.point + self.axis * self.sym_alpha)
        kstate.data_state.value_table[self.sym_alpha] = 0.0
        kstate.data_state.value_table[self.sym_beta]  = 0.0
        
        if self.t_lower != None or self.t_upper != None:
            lower = self.t_lower if self.t_lower != None else -10e9
            upper = self.t_upper if self.t_upper != None else  10e9
            kstate.hc_set[str(self.sym_alpha)] = HC(lower - self.sym_alpha, 
                                                    upper - self.sym_alpha, self.sym_alpha)
        if self.r_lower != None or self.r_upper != None:
            lower = self.r_lower if self.r_lower != None else -10e9
            upper = self.r_upper if self.r_upper != None else  10e9
            kstate.hc_set[str(self.sym_beta)] = HC(lower - self.sym_beta, 
                                                    upper - self.sym_beta, self.sym_beta)

        self.d_obj.pose = t_screw * (inverse_frame(t_screw.subs(kstate.data_state.value_table)) * self.d_obj.pose)

    def remove(self, kstate):
        super(CylindricalPair, self).remove(kstate)
        if self.t_lower != None or self.t_upper != None:
            del kstate.hc_set[str(self.sym_alpha)]
        if self.r_lower != None or self.r_upper != None:
            del kstate.hc_set[str(self.sym_beta)]
        del kstate.data_state.value_table[self.sym_alpha]
        del kstate.data_state.value_table[self.sym_beta]

# Higher Pair Joints
# class Contact(KinematicRule):
#     def __init__(self, position):
#         super(PlanarContact, self).__init__('')
#         self.name = name

#     def get_boundary_constraints(self):
#         raise NotImplementedError

#     def get_kinematic_constraints(self):
#         raise NotImplementedError