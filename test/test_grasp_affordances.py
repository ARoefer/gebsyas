#!/usr/bin/env python
import unittest

from urdf_parser_py.urdf import URDF
import numpy as np

from fetch_giskard.fetch_robot import Gripper
from gebsyas.grasp_affordances import BasicGraspAffordances as BGA
from giskardpy.sympy_wrappers import *
import symengine as sp
from giskardpy.input_system import *

PKG = 'gebsyas'

def dict_union(*args):
    out = {}
    for d in args:
        out.update(d)
    return out


class TestGraspAffordances(unittest.TestCase):

    def expectation(self, msg, f, *args):
        try:
            f(*args)
        except self.failureException as e:
            self.expectBacklog.append('{}'.format(msg))

    def expectEqual(self, value, expected, msg=''):
        self.expectation('{}\n    Expected: {}\n    Got:{}'.format(msg, expected, value),self.assertEqual, value, expected)

    def expectTrue(self, value, msg=''):
        self.expectation('{}\n    Expected True but got False'.format(msg),self.assertTrue, value)

    def expectFalse(self, value, msg=''):
        self.expectation('{}\n    Expected False but got True'.format(msg),self.assertFalse, value)

    def expectIsA(self, value, expected, msg=''):
        self.expectation('{}\n    Expected {} to be {}, but it\'s not.'.format(msg, value, expected), self.assertIsA, value, expected)

    def expectIsNotA(self, value, expected, msg=''):
        self.expectation('{}\n    Expected {} not to be {}, but it is.'.format(msg, value, expected), self.assertIsNotA, value, expected)

    def expectIsNone(self, value, msg=''):
        self.expectation('{}\n    Expected {} not be None, but it\'s not.'.format(msg, value), self.assertIsNone, value)

    def expectIsNotNone(self, value, msg=''):
        self.expectation('{}\n    Expected {} not to be None, but it is.'.format(msg, value), self.assertIsNotNone, value)

    def expectIsLess(self, a, b, msg=''):
        self.expectation('{}\n    Expected {} < {}\n    But it\'s not.'.format(msg, a, b), self.assertTrue, a < b)

    def expectIsGreater(self, a, b, msg=''):
        self.expectation('{}\n    Expected {} > {}\n    But it\'s not.'.format(msg, a, b), self.assertTrue, a > b)

    def expectIsLEq(self, a, b, msg=''):
        self.expectation('{}\n    Expected {} <= {}\n    But it\'s not.'.format(msg, a, b), self.assertTrue, a <= b)

    def expectIsGEq(self, a, b, msg=''):
        self.expectation('{}\n    Expected {} >= {}\n    But it\'s not.'.format(msg, a, b), self.assertTrue, a >= b)

    def tearDown(self):
        if len(self.expectBacklog):
            msg = '\n'.join(self.expectBacklog)
            self.fail(msg)


    def setUp(self):
        self.expectBacklog = []
        self.gripper_pos_input = Point3('gripper')
        self.gripper_rpy_input = ControllerInputArray(['r', 'p', 'y'], 'gripper_rotation')
        self.gripper_opening = ScalarInput('gripper_opening')

        self.large_gripper = Gripper(frame3_rpy(self.gripper_rpy_input.to_symbol('r'),
                                                self.gripper_rpy_input.to_symbol('p'),
                                                self.gripper_rpy_input.to_symbol('y'),
                                                self.gripper_pos_input.get_expression()),
                                                self.gripper_opening.get_expression(), 0.03, 0.1)

        self.small_gripper = Gripper(frame3_rpy(self.gripper_rpy_input.to_symbol('r'),
                                                self.gripper_rpy_input.to_symbol('p'),
                                                self.gripper_rpy_input.to_symbol('y'),
                                                self.gripper_pos_input.get_expression()),
                                                self.gripper_opening.get_expression(), 0.03, 0.05)

        self.point_input  = Point3('point')
        self.point = self.point_input.get_expression()
        self.normal_input = Vec3('normal')
        self.normal = self.normal_input.get_expression()
        self.axis_input   = Vec3('axis')
        self.axis = self.axis_input.get_expression()
        self.width_input  = ScalarInput('width')
        self.width = self.width_input.get_expression()
        self.length_input = ScalarInput('length')
        self.length = self.length_input.get_expression()
        self.angle_input  = ScalarInput('angle')
        self.angle = self.angle_input.get_expression()


    def get_gripper_dict(self,x,y,z,rr,rp,ry,o):
        out = self.gripper_pos_input.get_update_dict(x,y,z)
        out.update(self.gripper_rpy_input.get_update_dict(r=rr,p=rp,y=ry))
        out.update(self.gripper_opening.get_update_dict(o))
        return out

    def test_point_grasp(self):
        expression = BGA.point_grasp(self.large_gripper, self.point)
        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0)))
        result_zero = expression.subs(dict_union(self.get_gripper_dict(-1,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0)))
        self.assertEqual(result_perfect, 1, 'Affordance should be 1 at 0m distance')
        self.assertEqual(result_zero, 0, 'Affordance should be 0 at 1m distance')


    def test_sphere_grasp(self):
        expression = BGA.sphere_grasp(self.large_gripper, self.point, self.width)
        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06)))

        result_perfect_rot = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,1.56,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06)))

        result_zero = expression.subs(dict_union(self.get_gripper_dict(-1,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06)))

        result_unsolvable = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.12)))

        self.expectEqual(result_perfect, 1, 'Affordance should be 1 at 0m distance.')
        self.expectEqual(result_perfect_rot, 1, 'Affordance should be 1 at 0m distance indendant of rotation.')
        self.expectEqual(result_zero, 0, 'Affordance should be 0 at 1m distance.')
        self.expectEqual(result_unsolvable, 0, 'Affordance should be 0 at 0m distance if the maximum gripper opening is smaller than the sphere\'s width.')

    def test_partial_sphere_grasp(self):
        expression = BGA.partial_sphere_grasp(self.large_gripper, self.point, self.normal, self.width, self.angle)
        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,-sp.pi*0.5,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06),
                                         self.normal_input.get_update_dict(0,0,1),
                                         self.angle_input.get_update_dict(sp.pi * 0.5))).evalf(real=True)

        result_perfect_rot_inside = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,-sp.pi*0.5,1.56,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06),
                                         self.normal_input.get_update_dict(0,0,1),
                                         self.angle_input.get_update_dict(sp.pi * 0.5))).evalf(real=True)

        result_perfect_rot_outside = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,-sp.pi*0.2,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06),
                                         self.normal_input.get_update_dict(0,0,1),
                                         self.angle_input.get_update_dict(sp.pi * 0.5))).evalf(real=True)

        result_zero = expression.subs(dict_union(self.get_gripper_dict(-1,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06),
                                         self.normal_input.get_update_dict(0,0,1),
                                         self.angle_input.get_update_dict(sp.pi))).evalf(real=True)

        result_unsolvable = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.width_input.get_update_dict(0.06),
                                         self.normal_input.get_update_dict(0,0,1),
                                         self.angle_input.get_update_dict(sp.pi))).evalf(real=True)

        self.expectEqual(result_perfect, 1.0, 'Affordance should be 1 at 0m distance and top down rotation.')
        self.expectEqual(result_perfect_rot_inside, 1.0, 'Affordance should be 1 at 0m distance and rotation inside bounds.')
        self.expectIsLess(result_perfect_rot_outside, 1, 'Affordance should be <1 at 0m distance and rotation outside of bounds.')
        self.expectEqual(result_zero, 0.0, 'Affordance should be 0 at 1m distance indendant of rotation.')

        self.expectEqual(result_unsolvable, 0.0, 'Affordance should be 0 at 0m distance if the maximum gripper opening is smaller than the sphere\'s width.')

    def test_axis_grasp(self):
        expression = BGA.axis_grasp(self.large_gripper, self.point, self.axis)
        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1)))

        result_perfect_offset = expression.subs(dict_union(self.get_gripper_dict(0,0,1,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1)))

        result_perfect_rot = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,1.31,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1)))

        result_zero_pos = expression.subs(dict_union(self.get_gripper_dict(-1,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1)))

        result_zero_rot = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,sp.pi*0.5,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1))).evalf(real=True)

        self.expectEqual(result_perfect, 1, 'Affordance should be 1 at 0m distance and zero rotation.')
        self.expectEqual(result_perfect_offset, 1, 'Affordance should be 1 at any offset along the axis and zero rotation.')
        self.expectEqual(result_perfect_rot, 1, 'Affordance should be 1 at any offset along the axis and any rotation around the axis.')
        self.expectEqual(result_zero_pos, 0, 'Affordance should be 0 at 1m distance from the axis.')
        self.expectIsLess(result_zero_rot, 0.01, 'Affordance should be 0 at 90 degree angle alignment error to the axis.')

    def test_line_grasp(self):
        expression = BGA.line_grasp(self.large_gripper, self.point, self.axis, self.length)
        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.length_input.get_update_dict(0.1)))

        result_inside = expression.subs(dict_union(self.get_gripper_dict(0,0,-0.08,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.length_input.get_update_dict(0.1)))

        result_outside = expression.subs(dict_union(self.get_gripper_dict(0,0,0.2,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.length_input.get_update_dict(0.1)))

        self.expectEqual(result_perfect, 1, 'Affordance should be 1 at center of line.')
        self.expectEqual(result_inside, 1, 'Affordance should be 1 as long as the gripper is in the bounds of the line.')
        self.expectIsLess(result_outside, 1, 'Affordance should be less than 1 if the gripper is outside of the line\'s bounds.')

    def test_column_grasp(self):
        expression = BGA.column_grasp(self.large_gripper, self.point, self.axis, self.width)

        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,1,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.width_input.get_update_dict(0.06)))

        result_unsolvable = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.width_input.get_update_dict(0.2)))

        self.expectEqual(result_perfect, 1, 'Affordance should be 1 if the gripper is on the columns axis and the column is slim enough.')
        self.expectEqual(result_unsolvable, 0, 'Affordance should be 0 if column is wider than gripper\'s maximum opening width.')


    def test_rod_grasp(self):
        expression = BGA.rod_grasp(self.large_gripper, self.point, self.axis, self.length, self.width)

        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,-0.08,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.length_input.get_update_dict(0.1),
                                         self.width_input.get_update_dict(0.06)))

        result_unsolvable = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.length_input.get_update_dict(0.1),
                                         self.width_input.get_update_dict(0.2)))

        self.expectEqual(result_perfect, 1, 'Affordance should be 1 if the gripper is within the bounds on the rod\'s axis and the rod is slim enough.')
        self.expectEqual(result_unsolvable, 0, 'Affordance should be 0 if rod is wider than gripper\'s maximum opening width.')

    def test_edge_grasp(self):
        expression = BGA.edge_grasp(self.large_gripper, self.point, self.normal, self.axis)

        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.normal_input.get_update_dict(-1,0,0)))

        result_zero_rot = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,sp.pi*0.5,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.normal_input.get_update_dict(-1,0,0))).evalf(real=True)

        self.expectEqual(result_perfect, 1, 'Affordance should be 1 if the gripper is on the edge and aligned with the negated normal.')
        self.expectIsLess(result_zero_rot, 0.01, 'Affordance should be 0 if gripper is at a 90 degree angle to the normal.')

    def test_rim_grasp(self):
        expression = BGA.rim_grasp(self.large_gripper, self.point, self.normal, self.axis, self.width)

        result_perfect = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.normal_input.get_update_dict(-1,0,0),
                                         self.width_input.get_update_dict(0.06)))

        result_unsolvable = expression.subs(dict_union(self.get_gripper_dict(0,0,0,0,0,0,0.1),
                                         self.point_input.get_update_dict(0,0,0),
                                         self.axis_input.get_update_dict(0,0,1),
                                         self.normal_input.get_update_dict(-1,0,0),
                                         self.width_input.get_update_dict(0.2)))

        self.expectEqual(result_perfect, 1, 'Affordance should be 1 if the gripper meets all of the edges requierements and the rim is not too wide.')
        self.expectEqual(result_unsolvable, 0, 'Affordance should be 0 if the rim is too wide to fit into the gripper.')



if __name__ == '__main__':
    import rosunit

    rosunit.unitrun(package=PKG,
                    test_name='TestGraspAffordances',
                    test=TestGraspAffordances)
