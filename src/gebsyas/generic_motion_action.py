import traceback
import string
from gebsyas.constants import *
from gebsyas.basic_controllers import InEqController, run_ineq_controller
from gebsyas.bullet_based_controller import InEqBulletController
from gebsyas.actions import Action, PActionInterface, PWrapperInstance
from gebsyas.dl_reasoning import DLRigidObject
from gebsyas.predicates import *
from gebsyas.utils import saturate, tip_at_one
from gebsyas.trackers import SymbolicObjectPoseTracker

from giskardpy.symengine_wrappers import *
from giskardpy.symengine_controller import SymEngineController
from giskardpy.qp_problem_builder import SoftConstraint

from sensor_msgs.msg import JointState

from time import time
from blessed import Terminal


def robot_joints_symbol_map(robot):
	return dict([(str(j), 'joint_state/{}'.format(j)) for j in robot._joints.keys()])


class GenericMotionAction(Action):
	"""
	@brief      This action converts a set of inequality constraints directly into a controller and runs it.
	"""
	def __init__(self, ineq_constraints, allowed_collision_ids={}):
		super(GenericMotionAction, self).__init__('GenericMotion')
		self.ineq_constraints = ineq_constraints
		self.terminal = Terminal()
		self.allowed_collision_ids = allowed_collision_ids

	def execute(self, context):
		try:
			original_constraints = set(self.ineq_constraints.keys())
			motion_ctrl = InEqBulletController(context,
											   context.agent.get_data_state().dl_data_iterator(DLRigidObject),
											   self.allowed_collision_ids,
											   3,
											   self.clear_and_print) #context.log
			motion_ctrl.init(self.ineq_constraints)

			motion_success, m_lf, t_log = run_ineq_controller(context.agent.robot, motion_ctrl, 45.0, 3.5, context.agent, task_constraints=original_constraints)

			context.display.draw_robot_trajectory('motion_action', context.agent.robot, t_log)

			if motion_success:
				return 1.0
			else:
				context.log('Whoops, motion failed. I don\'t know what to do now, so I\'ll just abort...')
		except Exception as e:
			context.log(traceback.format_exc())

		return 0.0

	def clear_and_print(self, msg):
		print('{}{}'.format(self.terminal.clear(), msg))


class FreeformMotionInterface(PActionInterface):
	"""
	@brief      Symbolic motion action interface which has neither pre- nor postconditions.
	"""
 	def __init__(self, postcons):
 		super(FreeformMotionInterface, self).__init__(
 			'Move{}'.format(''.join([p.predicate.P for p in postcons])),
 			[PInstance(IsControlled, (postcons[0].args[0],), True)],
 			 #PInstance(InPosture, ('me/robot/state', 'me/memory/basic_stance'), True)],
 			postcons, 2.0)

	def instantiate_action(self, context, assignments):
		ineq_constraints = {}
		pred_state = context.agent.get_predicate_state()
		d_assignments =  {a: pred_state.map_to_data(a).data for a in self.signature.keys()}
		for k in d_assignments.keys():
			v = d_assignments[k]
			if type(v) == SymbolicData:
				d_assignments[k] = v.data

		for p, args in self.postcons.items():
			for at, value in args.items():
				fargs = [d_assignments[a] for a in at]
				ineq_constraints.update(p.fp(context, *fargs))

		return GenericMotionAction(ineq_constraints)

	def parameterize_by_postcon(self, context, postcons):
		return iter([PWrapperInstance(assignments={}, precons=self.precons.copy(), postcons=self.postcons.copy(), action_wrapper=self)])

	def parameterize_by_precon(self, context, precons):
		return iter([PWrapperInstance(assignments={}, precons=self.precons.copy(), postcons=self.postcons.copy(), action_wrapper=self)])


class GenericMotionInterface(PActionInterface):
	"""
	@brief      Symbolic motion action interface which instantiates a GenericMotionAction.
	"""
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


class GraspableMotionInterface(PActionInterface):
	def __init__(self):
		super(GraspableMotionInterface, self).__init__(
			'Move{}'.format(Graspable.P),
			[],
			[PInstance(Graspable, ('m', 'b'), True)],
			2.0)

	"""
	@brief      Symbolic motion action interface which instantiates a GenericMotionAction.
	"""
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

		return GenericMotionAction(ineq_constraints, {assignments['b']})

class SimpleBinaryMoveAction(GenericMotionInterface):
	"""
	@brief      Symbolic action interface with a binary predicate as postcondition.
	"""
	def __init__(self, binary_predicate):
		super(SimpleBinaryMoveAction, self).__init__(
			'Move{}'.format(binary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(binary_predicate, ('a', 'b'), True)],
			2.0)

class SimpleTrinaryMoveAction(GenericMotionInterface):
	"""
	@brief      Symbolic action interface with a trinary predicate as postcondition.
	"""
	def __init__(self, trinary_predicate):
		super(SimpleTrinaryMoveAction, self).__init__(
			'Move{}'.format(trinary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(trinary_predicate, ('a', 'b', 'c'), True)],
			2.0)

class UprightBinaryMoveAction(GenericMotionInterface):
	"""
	@brief      Symbolic action interface with a binary predicate as postcondition. Additionally, The Upright('a') is also always part of the postcondition.
	"""
	def __init__(self, binary_predicate):
		super(UprightBinaryMoveAction, self).__init__(
			'Move{}'.format(binary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(Upright, ('a',), True),
			 PInstance(binary_predicate, ('a', 'b'), True)],
			2.0)

class UprightTrinaryMoveAction(GenericMotionInterface):
	"""
	@brief      Symbolic action interface with a trinary predicate as postcondition. Additionally, The Upright('a') is also always part of the postcondition.
	"""
	def __init__(self, trinary_predicate):
		super(UprightTrinaryMoveAction, self).__init__(
			'Move{}'.format(trinary_predicate.P),
			[PInstance(IsControlled,('a',), True)],
			[PInstance(Upright, ('a',), True),
			 PInstance(trinary_predicate, ('a', 'b', 'c'), True)],
			2.0)

class UnconditionalMotionAction(GenericMotionInterface):
	"""
	@brief      Symbolic action interface which doesn't have a precondition.
	"""
	def __init__(self, predicate):
		super(UnconditionalMotionAction, self).__init__(
			'Move{}'.format(predicate.P),
			[],
			[PInstance(predicate, tuple(string.ascii_lowercase[:len(predicate.dl_args)]), True)],
			2.0)



# List of action interfaces defined in this file.
ACTIONS = [GraspableMotionInterface(),
		   # SimpleBinaryMoveAction(PointingAt),
		   # SimpleBinaryMoveAction(OnTop),
		   # #SimpleBinaryMoveAction(Above),
		   # SimpleTrinaryMoveAction(Below),
		   # SimpleTrinaryMoveAction(RightOf),
		   # SimpleTrinaryMoveAction(LeftOf),
		   # SimpleTrinaryMoveAction(InFrontOf),
		   # SimpleTrinaryMoveAction(Behind),
		   # UprightBinaryMoveAction(OnTop),
		   # #UprightBinaryMoveAction(Above),
		   # UprightTrinaryMoveAction(Below),
		   # UprightTrinaryMoveAction(RightOf),
		   # UprightTrinaryMoveAction(LeftOf),
		   # UprightTrinaryMoveAction(InFrontOf),
		   # UprightTrinaryMoveAction(Behind),
		   UnconditionalMotionAction(InPosture)]

