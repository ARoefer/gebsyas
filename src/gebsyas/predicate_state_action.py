import rospy
from gebsyas.grasp_affordances import BasicGraspAffordances as BGA
from gebsyas.actions import Action
from gebsyas.grasp_action import GraspAction
from gebsyas.basic_controllers import InEqController, run_ineq_controller
from giskardpy.symengine_wrappers import *
from gebsyas.predicates import DLSpatialPredicate, DLGraspPredicate, Free
from gebsyas.dl_reasoning import DLMultiManipulatorRobot, DLSingleManipulatorRobot, DLManipulationCapable
from gebsyas.numeric_scene_state import AssertionDrivenPredicateState
from gebsyas.planner import PlanIterator

class PSAction(Action):
	def __init__(self, goal_dict):
		""" goal = {(Predicate, args): bool} """
		super(PSAction, self).__init__('PSAction')

		self.goal = AssertionDrivenPredicateState(goal_dict)

	def execute(self, context):
		pred_state = context.agent.get_predicate_state()
		data_state = context.agent.get_data_state()

		state_diff = self.goal.difference(context, pred_state)

		if len(state_diff) > 0:
			# Profile the difference and then decide between intuition and planning
			# For now, there's only planning
			for plan in PlanIterator(context, self.goal):
				context.log('Executing next plan:\n   {}'.format(str(plan)))
				if self.execute_subaction(context, plan) > 0.8:
					if len(self.goal.difference(context, pred_state)) == 0:
						context.log('Goal configuration achieved!')
						return 1.0
				elif rospy.is_shutdown():
					return 0.0
				#else:
				#	return self.execute_subaction(context, PSAction(self.goal.predicates))
			context.log('Failed to find a plan that achieves goal configuration.')
			return 0.0
		else:
			context.log('The goal state is already achieved. Nothing to do here.')

		return 1.0