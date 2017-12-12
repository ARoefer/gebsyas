from giskard_affordances.grasp_affordances import BasicGraspAffordances as BGA
from giskard_affordances.actions import Action
from giskard_affordances.pnp_action import PNPAction
from giskard_affordances.grasp_action import GraspAction
from sensor_msgs.msg import JointState
from giskardpy.symengine_wrappers import *
from giskard_affordances.predicates import DLSpacialPredicate, DLGraspPredicate, Free
from giskard_affordances.dl_reasoning import DLMultiManipulatorRobot, DLSingleManipulatorRobot, DLManipulationCapable

class PSAction(Action):
	def __init__(self, goal):
		""" goal = {(Predicate, args): bool} """
		super(PSAction, self).__init__('PSAction')
		self.goal = goal

	def execute(self, context):
		pred_state = context.agent.get_predicate_state()
		data_state = context.agent.get_data_state()
		for pred_inst, val in self.goal.items():
			pred, args = pred_inst
			if pred_state.evaluate(pred, args, context.log) != val:

				if DLSpacialPredicate.is_a(pred):
					if DLMultiManipulatorRobot.is_a(context.agent.robot):
						grippers = dict([(g.name, g) for g in context.agent.robot.grippers if pred_state.evaluate(Free, (gripper.name))])
					elif DLSingleManipulatorRobot.is_a(context.agent.robot):
						grippers = {context.agent.robot.gripper.name: context.agent.robot.gripper}
					else:
						context.log("The robot lacks manipulation capabilities. Skipping goal {}{} = {}".format(pred.P, args, val))
						continue

					obj = args[0]
					pnp_action = PNPAction(context.agent.robot, grippers, pred_state.map_to_data(obj).data, BGA.object_grasp, pred.fp, {}, [pred_state.map_to_data(x).data for x in list(args[1:])])
					if self.execute_subaction(context, pnp_action) > 0.8:
						context.log("Successfully executed PNP-action to match {}{} = {}".format(pred.P, args, val))
					else:
						context.log("PNP-action to match {}{} = {} failed. Aborting the entire thing...".format(pred.P, args, val))
						return 0.0

				elif DLGraspPredicate.is_a(pred):
					if not DLManipulationCapable.is_a(context.agent.robot):
						context.log("The robot lacks manipulation capabilities. Skipping goal {}{} = {}".format(pred.P, args, val))
						continue

					gripper = pred_state.map_to_data(args[0]).data.data
					obj = pred_state.map_to_data(args[1]).data
					grasp_action = GraspAction(context.agent.robot, [gripper], obj, BGA.object_grasp)
					if self.execute_subaction(context, grasp_action) > 0.8:
						context.log('Successfully executed grasp-action to match {}{} = {}'.format(pred.P, args, val))
					else:
						context.log("grasp-action to match {}{} = {} failed. Aborting the entire thing...".format(pred.P, args, val))
						return 0.0
				else:
					context.log("I can't classify {} thus don't know what to do about it.".format(pred.P))

			else:
				context.log('{}{} is already {}... Nothing to do here'.format(pred.P, args, val))

		return 1.0