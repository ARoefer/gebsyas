from time import time

from gebsyas.actions import Action
from gebsyas.basic_controllers import InEqController, run_ineq_controller
from gebsyas.constants import *
from gebsyas.gnp_planner import GNPPlanner
from gebsyas.grasp_action import GripperAction
from giskardpy.symengine_wrappers import *
from sensor_msgs.msg import JointState


def robot_joints_symbol_map(robot):
	return dict([(str(j), 'joint_state/{}'.format(j)) for j in robot._joints.keys()])

class PNPAction(Action):
	def __init__(self, robot, grippers, obj,
				 f_grasp_gen, f_goal_gen,
				 f_grasp_gen_args = {},
				 f_goal_gen_args = {}):
		super(PNPAction, self).__init__('Single PNP')
		self.robot = robot
		self.grippers = grippers
		self.object = obj
		self.f_grasp_gen = f_grasp_gen
		self.f_goal_gen  = f_goal_gen
		self.f_grasp_gen_args = f_grasp_gen_args
		self.f_goal_gen_args  = f_goal_gen_args

	def execute(self, context):
		gnpp = GNPPlanner(self.robot, self.grippers, self.object,
						  self.f_grasp_gen, self.f_goal_gen,
						  self.f_grasp_gen_args, self.f_goal_gen_args)

		t_solve_begin = time()
		best_gripper, rating, grasp_frame, goal_frame = gnpp.solve(200, {}, context.display)
		print('Solving for gnpp solution took {}'.format(time() - t_solve_begin))
		if rating >= grasping_threshold:
			grasp_ctrl = InEqController(self.robot, self.grippers[best_gripper].frame)
			grasp_ctrl.set_goal(grasp_frame)
			grasp_success, gr_lf, t_log = run_ineq_controller(self.robot, grasp_ctrl, 10.0, 1.5, context.agent)
			if grasp_success:
				if self.execute_subaction(context, GripperAction(self.grippers[best_gripper], 0)) > 0.8:
					symbolic_object_pose = self.grippers[best_gripper].frame.inv() * obj.pose
					context.agent.add_tracker(SymbolicObjectPoseTracker(obj.id, context.agent.data_state, symbolic_object_pose, robot_joints_symbol_map(self.robot)))
					place_ctrl = InEqController(self.robot, symbolic_object_pose)
					place_success, mv_lf, t_log = run_ineq_controller(self.robot, move_ctrl, 10.0, 1.5, context.agent)
					if place_success:
						if self.execute_subaction(context, GripperAction(self.grippers[best_gripper], 0)) > 0.8:
							context.agent.remove_tracker(obj.id)
							return 1.0
						else:
							context.log('Whoops, letting go of the object failed. I don\'t know what to do now, so I\'ll just abort...')
					else:
						context.log('Whoops, moving the object failed. I don\'t know what to do now, so I\'ll just abort...')
				else:
					context.log('Whoops, closing the gripper failed. I don\'t know what to do now, so I\'ll just abort...')
			else:
				context.log('Whoops, grasping failed. I don\'t know what to do now, so I\'ll just abort...')
		else:
			context.log('Whoops, grasp-planning failed. I don\'t know what to do now, so I\'ll just abort...')
			context.log('Rating was: {}'.format(rating))

		return 0.0


