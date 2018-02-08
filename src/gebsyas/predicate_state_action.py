from gebsyas.grasp_affordances import BasicGraspAffordances as BGA
from gebsyas.actions import Action
from gebsyas.pnp_action import PNPAction
from gebsyas.grasp_action import GraspAction
from gebsyas.basic_controllers import InEqController, run_ineq_controller
from sensor_msgs.msg import JointState
from giskardpy.symengine_wrappers import *
from gebsyas.predicates import DLSpatialPredicate, DLGraspPredicate, Free
from gebsyas.dl_reasoning import DLMultiManipulatorRobot, DLSingleManipulatorRobot, DLManipulationCapable, SymbolicData
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
					else:
						return 0.0 #return self.execute_subaction(context, PSAction(self.goal.predicates))
					return 1.0
			context.log('Failed to find a plan that achieves goal configuration.')
			return 0.0
		else:
			context.log('The goal state is already achieved. Nothing to do here.')

		return 1.0


		# for pred_inst, val in self.goal.items():
		# 	pred, args = pred_inst
		# 	if pred_state.evaluate(context, pred, args) != val:

		# 		if DLSpatialPredicate.is_a(pred):
		# 			res_args = []
		# 			for thing in [pred_state.map_to_data(x) for x in args]:
		# 				if type(thing.data) == SymbolicData:
		# 					res_args.append(thing.data.data)
		# 				else:
		# 					res_args.append(thing.data)

		# 			movement_ctrl = InEqController(context.agent.robot, pred.fp(context, *res_args), None, 1, context.log)
		# 			movement_success, gr_lf, t_log = run_ineq_controller(context.agent.robot, movement_ctrl, 10.0, 1.5, context.agent)

		# 			context.display.draw_robot_trajectory('move_action', context.agent.robot, t_log)

		# 			# if DLMultiManipulatorRobot.is_a(context.agent.robot):
		# 			# 	grippers = dict([(g.name, g) for g in context.agent.robot.grippers if pred_state.evaluate(Free, (gripper.name))])
		# 			# elif DLSingleManipulatorRobot.is_a(context.agent.robot):
		# 			# 	grippers = {context.agent.robot.gripper.name: context.agent.robot.gripper}
		# 			# else:
		# 			# 	context.log("The robot lacks manipulation capabilities. Skipping goal {}{} = {}".format(pred.P, args, val))
		# 			# 	continue

		# 			# obj = args[0]
		# 			# pnp_action = PNPAction(context.agent.robot, grippers, pred_state.map_to_data(obj).data, BGA.object_grasp, pred.fp, {}, [pred_state.map_to_data(x).data for x in list(args[1:])])
		# 			# if self.execute_subaction(context, pnp_action) > 0.8:
		# 			# 	context.log("Successfully executed PNP-action to match {}{} = {}".format(pred.P, args, val))
		# 			# else:
		# 			# 	context.log("PNP-action to match {}{} = {} failed. Aborting the entire thing...".format(pred.P, args, val))
		# 			# 	return 0.0

		# 		elif DLGraspPredicate.is_a(pred):
		# 			if not DLManipulationCapable.is_a(context.agent.robot):
		# 				context.log("The robot lacks manipulation capabilities. Skipping goal {}{} = {}".format(pred.P, args, val))
		# 				continue

		# 			gripper = pred_state.map_to_data(args[0]).data.data
		# 			obj = pred_state.map_to_data(args[1]).data
		# 			grasp_action = GraspAction(context.agent.robot, [gripper], obj, BGA.object_grasp)
		# 			if self.execute_subaction(context, grasp_action) > 0.8:
		# 				context.log('Successfully executed grasp-action to match {}{} = {}'.format(pred.P, args, val))
		# 			else:
		# 				context.log("grasp-action to match {}{} = {} failed. Aborting the entire thing...".format(pred.P, args, val))
		# 				return 0.0
		# 		else:
		# 			context.log("I can't classify {} thus don't know what to do about it.".format(pred.P))

		# 	else: