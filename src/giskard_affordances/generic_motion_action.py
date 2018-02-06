import traceback
import string
from giskard_affordances.constants import *
from giskard_affordances.gnp_planner import GNPPlanner
from giskard_affordances.basic_controllers import InEqController, run_ineq_controller
from giskard_affordances.bullet_based_controller import InEqBulletController
from giskard_affordances.actions import Action, PActionInterface
from giskard_affordances.dl_reasoning import DLRigidObject
from giskard_affordances.predicates import *
from giskard_affordances.utils import saturate, tip_at_one
from giskard_affordances.trackers import SymbolicObjectPoseTracker

from giskardpy.symengine_wrappers import *
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint

from sensor_msgs.msg import JointState

from time import time


def robot_joints_symbol_map(robot):
	return dict([(str(j), 'joint_state/{}'.format(j)) for j in robot._joints.keys()])


class GenericMotionAction(Action):
	def __init__(self, ineq_constraints):
		super(GenericMotionAction, self).__init__('GenericMotion')
		self.ineq_constraints = ineq_constraints

	def execute(self, context):
		try:
			original_constraints = set(self.ineq_constraints.keys())
			motion_ctrl = InEqBulletController(context,
											   context.agent.get_data_state().dl_data_iterator(DLRigidObject),
											   self.ineq_constraints,
											   3,
											   None,
											   1,
											   context.log)

			motion_success, m_lf, t_log = run_ineq_controller(context.agent.robot, motion_ctrl, 45.0, 1.5, context.agent, task_constraints=original_constraints)

			context.display.draw_robot_trajectory('motion_action', context.agent.robot, t_log)

			if motion_success:
				return 1.0
			else:
				context.log('Whoops, motion failed. I don\'t know what to do now, so I\'ll just abort...')
		except Exception as e:
			context.log(traceback.format_exc())

		return 0.0


class GenericMotionInterface(PActionInterface):
	def instantiate_action(self, context, assignments):
		ineq_constraints = {}
		pred_state = context.agent.get_predicate_state()
		d_assignments =  {a: pred_state.map_to_data(s).data for a, s in assignments.items()}
		for k in d_assignments.keys():
			v = d_assignments[k]
			if type(v) == SymbolicData:
				d_assignments[k] = v.data

		for p, args in self.postcons.items():
			for at, value in args.items():
				fargs = [d_assignments[a] for a in at]
				ineq_constraints.update(p.fp(context, *fargs))

		return GenericMotionAction(ineq_constraints)


class SimpleBinaryMoveAction(GenericMotionInterface):
	def __init__(self, binary_predicate):
		super(SimpleBinaryMoveAction, self).__init__(
			'Move{}'.format(binary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(binary_predicate, ('a', 'b'), True)],
			2.0)

class SimpleTrinaryMoveAction(GenericMotionInterface):
	def __init__(self, trinary_predicate):
		super(SimpleTrinaryMoveAction, self).__init__(
			'Move{}'.format(trinary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(trinary_predicate, ('a', 'b', 'c'), True)],
			2.0)

class UprightBinaryMoveAction(GenericMotionInterface):
	def __init__(self, binary_predicate):
		super(UprightBinaryMoveAction, self).__init__(
			'Move{}'.format(binary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(Upright, ('a',), True),
			 PInstance(binary_predicate, ('a', 'b'), True)],
			2.0)

class UprightTrinaryMoveAction(GenericMotionInterface):
	def __init__(self, trinary_predicate):
		super(UprightTrinaryMoveAction, self).__init__(
			'Move{}'.format(trinary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(Upright, ('a',), True),
			 PInstance(trinary_predicate, ('a', 'b', 'c'), True)],
			2.0)

class UnconditionalMotionAction(GenericMotionInterface):
	def __init__(self, predicate):
		super(UnconditionalMotionAction, self).__init__(
			'Move{}'.format(predicate.P),
			[],
			[PInstance(predicate, tuple(string.ascii_lowercase[:len(predicate.dl_args)]), True)],
			2.0)

