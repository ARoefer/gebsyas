import traceback
from giskard_affordances.constants import *
from giskard_affordances.gnp_planner import GNPPlanner
from giskard_affordances.basic_controllers import SimpleExpressionController, run_convergence_controller
from giskard_affordances.actions import Action, GripperAction
from giskard_affordances.dl_reasoning import DLCube
from giskard_affordances.predicates import Free
from giskard_affordances.utils import saturate, tip_at_one

from giskardpy.symengine_wrappers import *
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint

from sensor_msgs.msg import JointState

from time import time

def robot_joints_symbol_map(robot):
	return dict([(str(j), 'joint_state/{}'.format(j)) for j in robot._joints.keys()])

class BoxGraspController(QPController):
	def __init__(self, robot, box, builder_backend=None, weight=1):
		self.weight = weight
		if not DLCube.is_a(box):
			raise Exception('Box Grasp Controller was given the non-box object "{}" of type {}'.format(str(box), str(type(box))))
		self.box = box
		self.feedback = 0.0
		super(BoxGraspController, self).__init__(robot, builder_backend)

	# @profile
	def add_inputs(self, robot):
		pass

	# @profile
	def make_constraints(self, robot):
		t = time()
		# start_position = pos_of(start_pose)
		gx = x_col(robot.gripper.frame)
		gz = z_col(robot.gripper.frame)
		# print(str(gx))
		# print(str(gz))

		bx = x_col(self.box.pose)
		by = y_col(self.box.pose)
		bz = z_col(self.box.pose)
		# print(str(bx))
		# print(str(by))
		# print(str(bz))

		xx_dot = dot(gx, bx)
		zz_dot = dot(gz, bz)

		xy_dot = dot(gx, by)
		xz_dot = dot(gx, bz)
		zx_dot = dot(gz, bx)
		zy_dot = dot(gz, by)

		# dir towards object
		obj_dir = pos_of(self.box.pose)
		obj_dir = vec3(obj_dir[0], obj_dir[1], 0)
		obj_dir = 1 / norm(obj_dir) * obj_dir

		# Distances in box-coords
		b2g = pos_of(robot.gripper.frame) - pos_of(self.box.pose)
		distx  = dot(b2g, bx)
		disty  = dot(b2g, by)
		distz  = dot(b2g, bz)

		# Normed distances
		distx_n = distx / (self.box.length * 0.5)
		disty_n = disty / (self.box.width  * 0.5)
		distz_n = distz / (self.box.height * 0.5)

		dx_sm = self.box.length + 0.02
		dy_sm = self.box.width  + 0.02
		dz_sm = self.box.height + 0.02

		inx = max(distx_n**2 - 1, 0)
		iny = max(disty_n**2 - 1, 0)
		inz = max(distz_n**2 - 1, 0)

		# 'Convex' grasping
		convex_g = dot(gx, obj_dir)
		convex_g_ctrl = -convex_g
		convex_achieved = saturate(convex_g + 0.9)

		# Grasp along x
		#                        1             1                           1

		xz_align_and_open = xx_dot * abs(zz_dot) * saturate(robot.gripper.opening / dy_sm) * sp.heaviside(robot.gripper.max_opening - dy_sm)
		xy_align_and_open = xx_dot * abs(zy_dot) * saturate(robot.gripper.opening / dz_sm) * sp.heaviside(robot.gripper.max_opening - dz_sm)
		x_grasp_pos = inz - abs(distx) - abs(disty)  #tip_at_one(distx_n * -xx_dot) - max(xz_align_and_open - inz - abs(disty_n), xy_align_and_open - iny - abs(distz_n))

		x_rot_align = xz_align_and_open # max(xz_align_and_open, xy_align_and_open)
		x_rot_align_ctrl = 1 - x_rot_align
		x_grasp_pos_ctrl_lower = x_grasp_pos * x_rot_align
		x_grasp_pos_ctrl_upper = x_grasp_pos

		# Grasp along y
		yz_align_and_open = abs(xy_dot) * abs(zz_dot) * saturate(robot.gripper.opening / dx_sm) * sp.heaviside(robot.gripper.max_opening - dx_sm)
		yx_align_and_open = abs(xy_dot) * abs(zx_dot) * saturate(robot.gripper.opening / dz_sm) * sp.heaviside(robot.gripper.max_opening - dz_sm)
		y_grasp_pos = (-xy_dot * disty_n) * max(inz * (-distx_n**2 + 1) * yz_align_and_open, inx * (-distz_n**2 + 1) * yx_align_and_open)

		# Grasp along z
		zx_align_and_open = abs(xz_dot) * abs(zx_dot) * saturate(robot.gripper.opening / dy_sm) * sp.heaviside(robot.gripper.max_opening - dy_sm)
		zy_align_and_open = abs(xz_dot) * abs(zx_dot) * saturate(robot.gripper.opening / dx_sm) * sp.heaviside(robot.gripper.max_opening - dx_sm)
		z_grasp_pos = (-xz_dot * distz_n) * max(inx * (-disty_n**2 + 1) * zx_align_and_open, iny * (-distx_n**2 + 1) * zy_align_and_open)



		# self._soft_constraints['no convex grasps'] = SoftConstraint(lower=convex_g_ctrl,
		# 																upper=1000,
		# 																weight=self.weight*5,
		# 																expression=convex_g)

		self._soft_constraints['adjust orientation x grasp'] = SoftConstraint(lower=x_rot_align_ctrl,
																			  upper=x_rot_align_ctrl,
																			  weight=self.weight,
																			  expression=x_rot_align)

		# self._soft_constraints['adjust position x grasp'] = SoftConstraint(lower=x_grasp_pos_ctrl_lower,
		# 																upper=x_grasp_pos_ctrl_upper,
		# 																weight=self.weight,
		# 																expression=x_grasp_pos)

		self.expression = x_grasp_pos

		self._controllable_constraints = robot.joint_constraints
		self._hard_constraints = robot.hard_constraints
		print('make constraints took {}'.format(time() - t))
		#print('Expression: {}'.format(str(self.expression)))

	def set_goal(self, goal):
		pass

	def get_next_command(self):
		cmd = super(BoxGraspController, self).get_next_command()
		self.feedback = self.expression.subs(self.get_state())
		return cmd

class GraspAction(Action):
	def __init__(self, robot, grippers, obj,
				 f_grasp_gen, f_grasp_gen_args = {}):
		super(GraspAction, self).__init__('Grasp')
		self.robot = robot
		self.grippers = grippers
		self.object = obj
		self.f_grasp_gen = f_grasp_gen
		self.f_grasp_gen_args = f_grasp_gen_args

	def execute(self, context):
		try:
			if DLCube.is_a(self.object):
				grasp_ctrl = BoxGraspController(self.robot, self.object)
			else:
				grasp_ctrl = SimpleExpressionController(self.robot, self.f_grasp_gen(self.grippers[0], self.object, *self.f_grasp_gen_args), 1)
			grasp_rating, gr_dx, t_log = run_convergence_controller(self.robot, grasp_ctrl, 'feedback', 10.0, 1.5, context.agent, GRASP_DT_THRESHOLD, GRASP_THRESHOLD)

			context.display.draw_robot_trajectory('grasp_action', context.agent.robot, t_log)

			if grasp_rating >= GRASP_THRESHOLD:
				if self.execute_subaction(context, GripperAction(self.grippers[0], 0)) > 0.8:
					# symbolic_object_pose = self.grippers[0].frame.inv() * obj.pose
					# context.agent.add_tracker(SymbolicObjectPoseTracker(obj.id, context.agent.data_state, symbolic_object_pose, robot_joints_symbol_map(self.robot)))
					context.agent.get_predicate_state().assert_fact(Free, (self.grippers[0].name), False)
				else:
					context.log('Whoops, closing the gripper failed. I don\'t know what to do now, so I\'ll just abort...')
			else:
				context.log('Whoops, grasping failed. I don\'t know what to do now, so I\'ll just abort...')
		except Exception as e:
			context.log(traceback.format_exc())

		return 0.0