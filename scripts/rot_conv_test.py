#!/usr/bin/env python
from collections import OrderedDict
from time import sleep
from giskardpy.symengine_wrappers import *
from giskardpy.qp_problem_builder import *
from giskardpy.symengine_controller import rotation_conv
from gebsyas.utils import res_pkg_path
import pandas as pd

import sys
from blessed import Terminal

rad2deg = 57.2957795131
deg2rad = 1.0 / rad2deg

class Reprinter(object):
    def __init__(self):
        self.term = Terminal()
        self.last_lb = 0

    def begin_draw_cycle(self):
        sys.stdout.write(self.term.move(self.term.height - 2 - self.last_lb, 0))
        self.last_lb = 0

    def print_msg(self, msg):
        lines = msg.split('\n')
        for s in lines:
            sys.stdout.write('{}{}{}{}'.format(s, self.term.clear_eol(), self.term.move_down(), self.term.move_x(0)))
        self.last_lb += len(lines)
        
    def end_draw_cycle(self):
        sys.stdout.write(self.term.clear_eos())

def rotation_alignment_gen_constraints(goal_in_current, current_rotation, current_evaluated, weight=1):
    axis, angle   = axis_angle_from_matrix(goal_in_current)

    r_rot_control = axis * angle

    hack = rotation3_axis_angle([0, 0, 1], 0.0001)

    axis, angle = axis_angle_from_matrix((current_rotation.T * (current_evaluated * hack)).T)
    c_aa = (axis * angle)

    return OrderedDict([('align rotation 0', SoftConstraint(lower=r_rot_control[0],
                                               upper=r_rot_control[0],
                                               weight=weight,
                                               expression=c_aa[0])),
            ('align rotation 1', SoftConstraint(lower=r_rot_control[1],
                                               upper=r_rot_control[1],
                                               weight=weight,
                                               expression=c_aa[1])),
            ('align rotation 2', SoftConstraint(lower=r_rot_control[2],
                                               upper=r_rot_control[2],
                                               weight=weight,
                                               expression=c_aa[2]))])


def axis_align_directed(goal_axis, current_axis, axis_frame, axis_frame_evaluated, weight=1):
    axis_rot  = rot_of(axis_frame)
    goal_dot  = dot(goal_axis, current_axis)

    goal_rot_in_frame = axis_rot.T * rotation3_axis_angle(cross(current_axis, goal_axis), acos(goal_dot)) * axis_rot

    return rotation_alignment_gen_constraints(goal_rot_in_frame, axis_rot, rot_of(axis_frame_evaluated), weight)

def axis_align_non_directed(goal_axis, current_axis, axis_frame, axis_frame_evaluated, weight=1):
    axis_rot  = rot_of(axis_frame)
    goal_dot  = dot(goal_axis, current_axis)

    goal_rot_in_frame = axis_rot.T * rotation3_axis_angle(fake_sign(goal_dot) * cross(current_axis, goal_axis), acos(fake_Abs(goal_dot))) * axis_rot

    return rotation_alignment_gen_constraints(goal_rot_in_frame, axis_rot, rot_of(axis_frame_evaluated), weight)


def str_matrix(m):
    strs = [str(n) for n in m]
    col_widths = [max(*[len(strs[x * m.ncols() + c]) for x in range(m.nrows())]) for c in range(m.ncols())]
    m_str = '\n'.join(['|{}|'.format(' '.join(['{}:>{}{}'.format('{', w, '}') for w in col_widths]))] * m.nrows())
    return m_str.format(*strs)


if __name__ == '__main__':
    cpath = res_pkg_path('package://gebsyas/.controllers/')
    rr, rp, ry = tuple([Symbol(n) for n in 'rpy'])
    joint_symbols = {rr, rp, ry}

    goal_axis = vector3(1, 0, 0)
    goal_axis *= 1.0 / norm(goal_axis).evalf(real=True)


    frame = rotation3_rpy(rr, rp, ry)
    frame_z = x_of(frame)

    goal_ang = 90 * deg2rad

    goal_dot = dot(goal_axis, frame_z)
    goal_expr = goal_dot
    rot_axis = cross(frame_z, goal_axis)
    rot_axis_norm = norm(rot_axis)
    rot_expr = asin(rot_axis_norm)
    rot_ctrl = goal_ang - rot_expr

    rel_rot  = rotation3_axis_angle(rot_axis, acos(goal_dot))
    goal_rot = rotation3_rpy(0,0,0) #rel_rot * frame #rotation3_rpy(0,0,0)


    #soft_constraints = {'dot align': SoftConstraint(goal_ctrl, goal_ctrl, 1, goal_expr)}
    subs = {s: 0 for s in joint_symbols}
    subs[rp] = 0 #130.5890570239465 * deg2rad

    soft_constraints = OrderedDict([('simple axis alignment', SoftConstraint(rot_ctrl, rot_ctrl, 1, rot_expr)), ('lol', SoftConstraint(0,0,0,0))])
    #soft_constraints = rotation_conv(goal_rot, frame, frame.subs(subs))
    #soft_constraints = axis_align_directed(goal_axis, frame_z, frame, frame.subs(subs))
    
    # for s, c in soft_constraints.items():
    #     print('{}\n  lb: {}\n  ub: {}\n   e: {}'.format(s, c.lower, c.upper, c.expression))


    # print('\n'.join(['{}: lb = {:<12}  ub = {:<12}  e = {:<12}'.format(s, type(c.lower), type(c.upper), type(c.expression)) for s, c in soft_constraints.items()]))

    joint_constraints = {s: JointConstraint(-0.2, 0.2, 0.001) for s in joint_symbols}

    printer = Reprinter()
    qppb = QProblemBuilder(joint_constraints, 
                          {}, 
                          soft_constraints,
                          joint_symbols,
                          joint_symbols,
                          cpath,
                          printer.print_msg)

    integration_factor = 0.02
    iterations = 0
    last_ang = acos(goal_dot.subs(subs))


    while True:
        printer.begin_draw_cycle()
        ang = acos(goal_expr.subs(subs))
        if abs(goal_ang - ang) < 0.01:
            printer.print_msg('Axis alignment reached after {} iterations! Final angle: {} deg\nValues:\n  {}'.format(iterations, ang * rad2deg, '\n  '.join(['{}: {}'.format(str(s), v) for s, v in subs.items()])))
            printer.end_draw_cycle()
            break
        cmd = qppb.get_cmd({str(s): p for s, p in subs.items()})

        printer.print_msg('Current frame:\n{}\nTransforming rotation:\n{}\nGoal frame:\n{}\nRotation axis:\n{}\n||Rotation axis||: {}\n'.format(str_matrix(frame.subs(subs)), 
            str_matrix(goal_rot.subs(subs)), 
            str_matrix(rel_rot.subs(subs)),
            str_matrix(rot_axis.subs(subs)),
            rot_axis_norm.subs(subs)))
        printer.print_msg('[{:5d}] Converging at a speed of {} deg/s\n        Remaining difference: {} deg\n'.format(iterations, (ang - last_ang) / integration_factor * rad2deg, ang * rad2deg))
        subs = {s: p + cmd[str(s)] * integration_factor for s, p in subs.items()}
        iterations += 1
        last_ang = ang
        printer.end_draw_cycle()
        sleep(integration_factor)

       
