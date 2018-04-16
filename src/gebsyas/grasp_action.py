import traceback
from gebsyas.constants import *
from gebsyas.gnp_planner import GNPPlanner
from gebsyas.basic_controllers import InEqController, InEqFFBRunner
from gebsyas.actions import Action, PActionInterface
from gebsyas.dl_reasoning import DLCube
from gebsyas.predicates import Free, Graspable, IsControlled, IsGrasped, PInstance
from gebsyas.utils import saturate, tip_at_one
from gebsyas.trackers import SymbolicObjectPoseTracker

from giskardpy.symengine_wrappers import *
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint as SC

from sensor_msgs.msg import JointState

from time import time

def robot_joints_symbol_map(robot):
	return dict([(str(j), 'joint_state/{}'.format(j)) for j in robot._joints.keys()])

class GripperAction(Action):
	"""
	@brief      Action to change a gripper's opening.
	"""
	def __init__(self, gripper, goal_pos, effort=40):
		super(GripperAction, self).__init__('GripperAction: {}'.format(gripper.name))
		self.gripper = gripper
		self.goal    = goal_pos
		self.effort  = effort

	def execute(self, context):
		ineq_c = {'gripper': SC(self.goal - self.gripper.opening, self.goal - self.gripper.opening, 1, self.gripper.opening)}
		motion_ctrl = InEqController(context.agent.robot, ineq_c, None, 1, context.log)
		runner = InEqFFBRunner(context.agent.robot,
							   motion_ctrl,
							   10.0, 1.5,
							   {'gripper_joint': self.effort},
							   context.agent.act,
							   context.agent.add_js_callback)
		motion_success, m_lf = runner.run()
		if motion_success:
			context.log('Gripper "{}" is now opened {} or is blocked at {} Nm'.format(self.gripper.name, self.goal, self.effort))
			return 1.0
		else:
			return 0.0

class GraspAction(Action):
	"""
	@brief      Action to grasp an object.
	"""
	def __init__(self, robot, gripper, obj):
		super(GraspAction, self).__init__('Grasp')
		self.robot = robot
		self.gripper = gripper
		self.object = obj

	def execute(self, context):
		try:
			if self.execute_subaction(context, GripperAction(self.gripper, 0)) > 0.8:
				context.agent.add_tracker(SymbolicObjectPoseTracker(self.object.id, context.agent.get_predicate_state(), self.gripper.name))
				return 1.0
			else:
				context.log('Whoops, closing the gripper failed. I don\'t know what to do now, so I\'ll just abort...')
		except Exception as e:
			context.log(traceback.format_exc())

		return 0.0

class LetGoAction(Action):
	"""
	@brief      Action to let go of a thing.
	"""
	def __init__(self, robot, gripper, obj):
		super(LetGoAction, self).__init__('LetGo')
		self.robot = robot
		self.gripper = gripper
		self.object = obj

	def execute(self, context):
		try:
			if self.execute_subaction(context, GripperAction(self.gripper, self.gripper.max_opening)) > 0.8:
				context.agent.remove_tracker(self.object.id)
				return 1.0
			else:
				context.log('Whoops, opening the gripper failed. I don\'t know what to do now, so I\'ll just abort...')
		except Exception as e:
			context.log(traceback.format_exc())

		return 0.0


class GraspActionInterface(PActionInterface):
	"""
	@brief      Symbolic action interface for grasping objects.
	"""
	def __init__(self):
		super(GraspActionInterface, self).__init__(
			'GraspAction',
			[PInstance(Graspable, ('a', 'b'), True),
			 PInstance(Free,           ('a'), True)],

			[PInstance(IsControlled,  ('b',), True),
			 PInstance(IsGrasped, ('a', 'b'), True),
			 PInstance(Free,           ('a'), False)])

	def instantiate_action(self, context, assignments):
		return GraspAction(context.agent.robot,
						   context.agent.get_predicate_state().map_to_data(assignments['a']).data.data,
						   context.agent.get_predicate_state().map_to_data(assignments['b']).data)

class LetGoActionInterface(PActionInterface):
	"""
	@brief      Symbolic action interface for letting go of objects.
	"""
	def __init__(self):
		super(LetGoActionInterface, self).__init__(
			'LetGoAction',
			[PInstance(IsGrasped, ('a', 'b'), True)],

			[PInstance(IsControlled,  ('b',), False),
			 PInstance(IsGrasped, ('a', 'b'), False),
			 PInstance(Free,           ('a'), True)])

	def instantiate_action(self, context, assignments):
		return LetGoAction(context.agent.robot,
						   context.agent.get_predicate_state().map_to_data(assignments['a']).data.data,
						   context.agent.get_predicate_state().map_to_data(assignments['b']).data)

# Action interfaces defined in this file.
ACTIONS = [GraspActionInterface(),
		   LetGoActionInterface()]
