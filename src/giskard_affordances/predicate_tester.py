import rospy
from giskard_affordances.ros_visualizer import ROSVisualizer
from giskard_affordances.dl_reasoning import DLRigidObject, DLCube, DLSphere, DLCylinder, DLPhysicalThing, DLTransform, DLVector, DLScalar, DLObserver, DLDisjunction
from giskard_affordances.utils import StampedData
from giskardpy.input_system import ScalarInput, Vec3Input, FrameInputRPY
from giskard_affordances.actions import Context, Logger
from giskard_affordances.agent import Agent
from giskard_affordances.basic_controllers import InEqController
from giskard_affordances.numeric_scene_state import visualize_obj
from giskardpy.robot import Robot
from giskardpy.qp_problem_builder import HardConstraint, JointConstraint
from random import random
from copy import deepcopy, copy
from collections import OrderedDict
from symengine import Symbol
from time import time

PI2 = 6.28318530717959

class PredicateTester(object):
	def __init__(self):
		self.visualizer = ROSVisualizer('predicate_tester')
		self.metric_boundary = 2.0
		self.generic_scalar_boundary = 0.4
		self.context = None
		self.logger = Logger()
		self.dl_filter = DLDisjunction(DLPhysicalThing, DLObserver)

	def envision_predicates(self, pinstances, data_state, flexible_paths, optimized_paths, tries=1, iteration_cap=200, integration_factor=0.2):
		now = rospy.Time.now()
		flexible_paths = flexible_paths + tuple([op for op in optimized_paths if op not in flexible_paths])

		data_state = deepcopy(data_state)

		initial_inputs  = {}
		for xpath in flexible_paths:
			if xpath in initial_inputs:
				continue

			obj = data_state[xpath].data
			if DLPhysicalThing.is_a(obj):
				flexible_paths.append('{}/pose'.format(xpath))

			if DLCube.is_a(obj):
				flexible_paths.extend(['{}/height'.format(xpath), '{}/width'.format(xpath), '{}/length'.format(xpath)])
			elif DLCylinder.is_a(obj):
				flexible_paths.extend(['{}/height'.format(xpath), '{}/radius'.format(xpath)])
			elif DLSphere.is_a(obj):
				flexible_paths.append('{}/radius'.format(xpath))

			if DLObserver.is_a(obj):
				flexible_paths.append('{}/frame_of_reference'.format(xpath))

			if DLTransform().is_a(obj):
				new_input = FrameInputRPY(xpath)
				initial_inputs[xpath] = new_input
				data_state.insert_data(StampedData(now, new_input.get_expression()), xpath)
			elif DLScalar().is_a(obj):
				new_input = ScalarInput(xpath)
				initial_inputs[xpath] = new_input
				data_state.insert_data(StampedData(now, new_input.get_expression()), xpath)


		context = Context(None, self.logger, self.visualizer)
		constraints = OrderedDict()
		for pinst in pinstances:
			p = pinst.predicate
			objs = [data_state.find_by_path(a, True).data for a in pinst.args]
			constraints.update(p.fp(context, *objs))

		fake_bot = Robot()
		flexible_objects = set()
		change_profile = {}
		for opt_path in optimized_paths:
			inpt = initial_inputs[opt_path]
			flexible_objects.add(opt_path[:opt_path.find('/')])
			if type(inpt) == ScalarInput:
				fake_bot.hard_constraints[opt_path] = HardConstraint(-inpt.get_expression(),
																	 self.generic_scalar_boundary - inpt.get_expression(),
																	 inpt.get_expression())
				fake_bot.joint_constraints[inpt.get_symbol_str()] = JointConstraint(-0.2, 0.2, 0.001)
			elif type(inpt) == FrameInputRPY:
				for fn in inpt.get_float_names():
					fake_bot.joint_constraints[inpt.to_str_symbol(fn)] = JointConstraint(-0.2, 0.2, 0.001)
					if 'r' not in fn:
						fake_bot.hard_constraints[inpt.to_str_symbol(fn)] = HardConstraint(-self.metric_boundary - inpt.to_symbol(fn),
																	 self.metric_boundary - inpt.to_symbol(fn),
																	 inpt.to_symbol(fn))
			change_profile.update({vn: (0,0,0) for vn in [inpt.to_str_symbol(fn) for fn in inpt.get_float_names()]})

		self.visualizer.begin_draw_cycle()
		controller = InEqController(fake_bot, constraints, logging=context.log)


		x = 0
		time_taken = 0
		succesful_runs = 0
		while x < tries and not rospy.is_shutdown():
			x += 1
			start_time = time()

			# Init
			for xpath, inpt in initial_inputs.items():
				if type(inpt) == ScalarInput:
					state = inpt.get_update_dict(random() * 2 * self.generic_scalar_boundary - self.generic_scalar_boundary)
				elif type(inpt) == FrameInputRPY:
					state = inpt.get_update_dict(random() * PI2, random() * PI2, random() * PI2,
												 random() * 2 * self.metric_boundary - self.metric_boundary,
												 random() * 2 * self.metric_boundary - self.metric_boundary,
												 random() * 2 * self.metric_boundary - self.metric_boundary)

				if xpath in optimized_paths:
					fake_bot._update_observables(state)
				else:
					controller.update_observables(state)

			initial_state = copy(fake_bot.get_state())

			y = 0
			while y < iteration_cap and not rospy.is_shutdown():
				y += 1

				cmd = controller.get_next_command()
				if cmd is None:
					controller.qp_problem_builder.log_jacobian()
					controller.qp_problem_builder.log_lb_ub()
					controller.qp_problem_builder.log_lbA_ubA()
					break

				fake_bot._update_observables({jn: p + integration_factor * cmd[jn] for jn, p in fake_bot.get_state().items()})
				if controller.qp_problem_builder.constraints_met(0.0001, -0.0001):
					self.logger('Constraints satisfied after {} iterations'.format(y))
					controller_state = {Symbol(k): v for k, v in controller.get_state().items()}
					self.logger('Final state:\n  {}'.format('\n  '.join(['{}: {}'.format(k, v) for k,v in controller_state.items()])))
					self.logger('Change per input:\n  {}'.format('\n  '.join(['{}: {}'.format(k, controller_state[Symbol(k)] - v) for k,v in initial_state.items()])))

					succesful_runs += 1
					time_taken += time() - start_time
					new_profile = {}
					for vn, t in change_profile.items():
						change = controller_state[Symbol(vn)] - initial_state[vn]
						new_profile[vn] = (min(t[0], change), max(t[1], change), t[2] + change)
					change_profile = new_profile

					now = rospy.Time.now()
					numeric_state = deepcopy(data_state)
					for xpath, inpt in initial_inputs.items():
						resolved_expr = inpt.get_expression().subs(controller_state)
						numeric_state.insert_data(StampedData(now, resolved_expr), xpath)

					for Id, obj in numeric_state.dl_data_iterator(self.dl_filter):
						if Id in flexible_objects:
							color = (random(), random(), 0, 1)
						else:
							color = (0, 0, random(),1)

						if DLRigidObject.is_a(obj.data):
							visualize_obj(obj.data, self.visualizer, obj.data.pose, color=color)
						elif DLObserver.is_a(obj.data):
							self.visualizer.draw_mesh('objects', obj.data.frame_of_reference, (0.4,0.4,0.4), 'package://giskard_affordances/meshes/camera.dae', r=color[0], g=color[1], b=color[2], a=color[3])
					break
			controller.qp_problem_builder.reset_solver()
		self.visualizer.render()

		self.logger('Total time taken: {:>10.4f}\nAvg time per run: {:>10.4f}'.format(time_taken, time_taken / succesful_runs))

		change_profile = {vn: (t[0], t[1], t[2] / succesful_runs) for vn, t in change_profile.items()}

		self.logger('{}\nVariable Changes:\n  {}'.format('-'*20, '\n  '.join(['{:>25}  {:>15}, {:>15}, {:>15}'.format(' ', 'min', 'max', 'avg')] +
			['{:>25}: {:>15.5f}, {:>15.5f}, {:>15.5f}'.format(vn, *t) for vn, t in change_profile.items()])))
