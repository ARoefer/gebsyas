import symengine as spw
import copy
import rospy

from giskardpy.symengine_controller import SymEngineController
from giskardpy.symengine_robot import Joint
from giskardpy.symengine_wrappers import *
from giskardpy.qp_problem_builder import JointConstraint, HardConstraint, SoftConstraint, BIG_NUMBER
from gebsyas.grasp_affordances import BasicGraspAffordances as BGA
from gebsyas.utils import subs_if_sym, JointState, jsDictToJSMsg
from gebsyas.ros_visualizer import ROSVisualizer
from fetch_giskard.fetch_robot import Gripper
from collections import OrderedDict, namedtuple

from sensor_msgs.msg import JointState as JointStateMsg


GNP = namedtuple('GNP', ['gnp_scs', 'gripper_t0_frame', 'object_t1_frame'])

class GNPPlanner(QPController):
    def __init__(self, robot, grippers, obj, grasp_gen_function, goal_gen_function, grasp_args={}, goal_args={}):
        self.grippers = grippers
        self.object = obj
        self.grasp_gen_function = grasp_gen_function
        self.goal_gen_function = goal_gen_function
        self.grasp_args = grasp_args
        self.goal_args = goal_args
        self.robot = robot

        super(GNPPlanner, self).__init__(robot)

    def add_inputs(self, robot):
        pass

    def make_constraints(self, robot):
        self.t0_symbols = dict([(k, spw.Symbol('{}_t0'.format(k))) for k in robot._joints.keys()])
        self.t1_symbols = dict([(k, spw.Symbol('{}_t1'.format(k))) for k in robot._joints.keys()])
        self.update_observables(robot.get_state())
        self._controllable_constraints =  OrderedDict([('{}_t0'.format(k), JointConstraint(-BIG_NUMBER, BIG_NUMBER, v.weight)) for k, v in robot.joint_constraints.items()])
        self._controllable_constraints.update(OrderedDict([('{}_t1'.format(k), JointConstraint(-BIG_NUMBER, BIG_NUMBER, v.weight)) for k, v in robot.joint_constraints.items()]))

        self._hard_constraints = OrderedDict([('{}_t0'.format(k), HardConstraint(subs_if_sym(h.lower, self.t0_symbols),
                                                                                 subs_if_sym(h.upper, self.t0_symbols),
                                                                                 subs_if_sym(h.expression, self.t0_symbols))) for k, h in robot.hard_constraints.items()])
        self._hard_constraints.update(OrderedDict([('{}_t1'.format(k), HardConstraint(subs_if_sym(h.lower, self.t1_symbols),
                                                                                 subs_if_sym(h.upper, self.t1_symbols),
                                                                                 subs_if_sym(h.expression, self.t1_symbols))) for k, h in robot.hard_constraints.items()]))
        print(self._hard_constraints)
        terms = []
        self.gnps = {}
        ifile = open('gnp_matrices.txt', 'w')
        for gn, g in self.grippers.items():
            gripper_t0 = Gripper(g.name,
                                 subs_if_sym(g.pose, self.t0_symbols),
                                 subs_if_sym(g.opening, self.t0_symbols),
                                 subs_if_sym(g.height, self.t0_symbols),
                                 subs_if_sym(g.max_opening, self.t0_symbols),
                                 g.link_name)
            ifile.write('\n{}_t0.pose:\n'.format(g.name))
            ifile.write(str(gripper_t0.pose))

            t0_frame_inv = gripper_t0.pose.inv()

            ifile.write('\n{}_t0.pose^-1:\n'.format(g.name))
            ifile.write(str(t0_frame_inv))
            ifile.write('\n{}.pose:\n'.format(self.object.id))
            ifile.write(str(self.object.pose))

            obj_in_gripper = t0_frame_inv * self.object.pose

            ifile.write('\n{}_t0.pose^-1 * {}.pose:\n'.format(g.name, self.object.id))
            ifile.write(str(obj_in_gripper))

            obj_t1 = copy.copy(self.object)
            obj_t1.pose = subs_if_sym(g.pose, self.t1_symbols) * obj_in_gripper

            ifile.write('\n{}_t1.pose * {}_t0.pose^-1 * {}.pose:\n'.format(g.name, g.name, self.object.id))
            ifile.write(str(obj_t1.pose))
            #print(obj_t1.pose)

            gnp_scs = self.grasp_gen_function(gripper_t0, self.object, *self.grasp_args)
            gnp_scs.update(self.goal_gen_function(obj_t1, *self.goal_args))
            # gnp_term = self.grasp_gen_function(gripper_t0, self.object, *self.grasp_args)
            #gnp_term = self.goal_gen_function(obj_t1, *self.goal_args)
            self.gnps[g.name] = GNP(gnp_scs, gripper_t0.pose, obj_t1.pose)

        ifile.close()
        #self.sc_expression = #BGA.combine_expressions_max(True, [gnp.gnp_term for gnp in self.gnps.values()])
        #sc_ctrl = 2 - self.sc_expression
        for gnp in self.gnps.values():
            self._soft_constraints.update(gnp.gnp_scs)

    def solve(self, iterations=10, initial_state={}, visualizer=None):
        if len(initial_state) == 0:
            self.update_observables({str(s): 0.0 for s in self.t0_symbols.values()})
            self.update_observables({str(s): 0.0 for s in self.t1_symbols.values()})
        else:
            for k, v in initial_state.items():
                self.update_observables({str(self.t0_symbols[k]): v, str(self.t1_symbols[k]): v})

        fjac = open('gnp_jacobian.txt', 'w')
        nextState = self.get_state()
        fjac.write(str(self.qp_problem_builder.A))

        for x in range(iterations):
            #print('Iteration: {:d} ---------------------'.format(x))
            #for j, p in nextState.items():
            #    print('{:>30}: {:f}'.format(j, p))
            command = self.qp_problem_builder.update_observables(nextState)
            for k, v in command.items():
                nextState[k] += v * 0.5

            t0_js = {j: nextState[str(s)] for j, s in self.t0_symbols.items()}
            t1_js = {j: nextState[str(s)] for j, s in self.t1_symbols.items()}

            if visualizer != None:
                visualizer.draw_robot_pose('t0', self.robot, t0_js, None, (0.2, 1, 0.2, 1))
                visualizer.draw_robot_pose('t1', self.robot, t1_js, None, (1, 0.2, 0.2, 1))

            fjac.write(self.qp_problem_builder.str_jacobian())
            fjac.write('\n')
        fjac.close()


        best = None
        bestRating = -BIG_NUMBER
        for k, v in self.gnps.items():
            value = v.gnp_term.subs(nextState)
            if value > bestRating:
                bestRating = value
                best = k

        if visualizer != None:
            visualizer.draw_robot_pose('t0_final', self.robot, t0_js, None, (0, 1, 0, 1))
            visualizer.draw_robot_pose('t1_final', self.robot, t1_js, None, (1, 0, 0, 1))

        return (best, bestRating, self.gnps[best].gripper_t0_frame.subs(nextState), self.gnps[best].object_t1_frame.subs(nextState))





