import rospy
from time import time
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
import giskardpy.symengine_wrappers as sw
from giskardpy.input_system import FrameInputRPY
from gebsyas.utils import StampedData
from gebsyas.dl_reasoning import DLSymbolic
from sensor_msgs.msg import JointState
from giskardpy import print_wrapper

class SimpleCartesianController(QPController):
    def __init__(self, robot, movable_frame, builder_backend=None, weight=1, logging=print_wrapper):
        self.weight = weight
        self.movable_frame = movable_frame
        self.feedback = 0.0
        super(SimpleCartesianController, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = FrameInputRPY('', 'goal')

    # @profile
    def make_constraints(self, robot):
        t = time()
        # start_position = pos_of(start_pose)

        current_pose = self.movable_frame
        current_rotation = sw.rot_of(current_pose)

        goal_position = self.goal_eef.get_position()
        goal_rotation = self.goal_eef.get_rotation()
        # goal_r = goal_rotation[:3,:3].reshape(9,1)

        #pos control
        dist = sw.norm(sw.pos_of(current_pose) - goal_position)

        #rot control
        dist_r = 2 - sw.dot(current_rotation[:3, :1], goal_rotation[:3, :1]) - sw.dot(current_rotation[:3, 1:2], goal_rotation[:3, 1:2])

        self.__feedback = 1 - dist - dist_r

        self._soft_constraints['align {} position'.format(eef)] = SoftConstraint(lower=-dist,
                                                                                 upper=-dist,
                                                                                 weight=1,
                                                                                 expression=dist)
        self._soft_constraints['align {} rotation'.format(eef)] = SoftConstraint(lower=-dist_r,
                                                                                 upper=-dist_r,
                                                                                 weight=1,
                                                                                 expression=dist_r)
        self._controllable_constraints = robot.joint_constraints
        self._hard_constraints = robot.hard_constraints
        self.logging('make constraints took {}'.format(time() - t))

    def set_goal(self, goal):
        """
        dict eef_name -> goal_position
        :param goal_pos:
        :return:
        """
        self.update_observables(self.goal_eef.get_update_dict(*goal))

    def get_next_command(self):
    	cmd = super(SimpleCartesianController, self).get_next_command()
    	self.feedback = self.__feedback.subs(self.get_state())
    	return cmd


class SimpleExpressionController(QPController):
    def __init__(self, robot, expression, limit, builder_backend=None, weight=1, logging=print_wrapper):
        self.weight = weight
        if not DLSymbolic().is_a(expression):
            raise Exception('Simple Expression Controller was given the non-symbolic expression "{}" of type {}'.format(str(expression), str(type(expression))))
        self.expression = expression
        self.limit = limit
        self.feedback = 0.0
        super(SimpleExpressionController, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        pass

    # @profile
    def make_constraints(self, robot):
        t = time()
        # start_position = pos_of(start_pose)
        ctrl = self.limit - self.expression

        self._soft_constraints['converge expression towards {}'.format(self.limit)] = SoftConstraint(lower=ctrl,
                                                                                 upper=ctrl,
                                                                                 weight=self.weight,
                                                                                 expression=self.expression)
        self._controllable_constraints = robot.joint_constraints
        self._hard_constraints = robot.hard_constraints
        self.logging('make constraints took {}'.format(time() - t))
        #print('Expression: {}'.format(str(self.expression)))

    def set_goal(self, goal):
        pass

    def get_next_command(self):
        cmd = super(SimpleExpressionController, self).get_next_command()
        self.feedback = self.expression.subs(self.get_state())
        return cmd

class InEqController(QPController):
    def __init__(self, robot, ineq_constraints, builder_backend=None, weight=1, logging=print_wrapper):
        self.weight = weight
        self.ineq_constraints = ineq_constraints
        self.feedback = 0.0
        super(InEqController, self).__init__(robot, builder_backend, logging)

    # @profile
    def add_inputs(self, robot):
        pass

    # @profile
    def make_constraints(self, robot):
        t = time()
        # start_position = pos_of(start_pose)
        self._soft_constraints = self.ineq_constraints.copy()
        self._controllable_constraints = robot.joint_constraints.copy()
        self._hard_constraints = robot.hard_constraints.copy()
        self.logging('make constraints took {}'.format(time() - t))
        #print('Expression: {}'.format(str(self.expression)))

    def set_goal(self, goal):
        pass

    def get_next_command(self):
        cmd = super(InEqController, self).get_next_command()
        return cmd

    def stop(self):
        pass


class InEqRunner(object):
    def __init__(self, robot, controller, tlimit_total,
                 tlimit_convergence, f_send_command,
                 f_add_cb, dt_threshold=0.02, task_constraints=None):
        self.robot = robot
        self.controller = controller
        self.tlimit_total  = rospy.Duration(tlimit_total)
        self.tlimit_convergence = rospy.Duration(tlimit_convergence)
        self.dt_threshold  = dt_threshold
        self.f_send_command = f_send_command
        self.f_add_cb = f_add_cb
        self.last_feedback = 0
        self.last_update   = None
        self.trajectory_log = []
        self.execution_start = None
        self.constraints_met = False
        self.task_constraints = task_constraints

    def run(self):
        now = rospy.Time.now()
        self.terminate = False
        self.total_timeout       = now + self.tlimit_total
        self.convergence_timeout = now + self.tlimit_convergence
        self.execution_start = now

        self.f_add_cb(self.js_callback)

        while not rospy.is_shutdown() and not self.terminate:
            pass

        self.controller.stop()
        return self.constraints_met, self.last_feedback

    def js_callback(self, joint_state):
        if self.terminate: # Just in case
            return

        self.trajectory_log.append(StampedData(rospy.Time.from_sec((rospy.Time.now() - self.execution_start).to_sec()), joint_state.copy()))

        now = rospy.Time.now()
        self.robot.set_joint_state(joint_state)
        command = self.controller.get_next_command()
        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()
        new_feedback = 0
        for jc in command.values():
            new_feedback += abs(jc)

        if self.last_update != None:
            if new_feedback > self.dt_threshold:
                self.convergence_timeout = now + self.tlimit_convergence

        self.last_feedback = new_feedback
        self.constraints_met = self.controller.qp_problem_builder.constraints_met(names=self.task_constraints)
        self.last_update   = now

        self.terminate = now >= self.total_timeout or now >= self.convergence_timeout or self.constraints_met

        self.f_send_command(command)


class InEqFFBRunner(object):
    def __init__(self, robot, controller, tlimit_total,
                 tlimit_convergence, force_threshold_dict, f_send_command,
                 f_add_cb, dt_threshold=0.02):
        self.robot = robot
        self.controller = controller
        self.tlimit_total  = rospy.Duration(tlimit_total)
        self.tlimit_convergence = rospy.Duration(tlimit_convergence)
        self.dt_threshold  = dt_threshold
        self.f_send_command = f_send_command
        self.f_add_cb = f_add_cb
        self.last_feedback = 0
        self.last_update   = None
        self.trajectory_log = []
        self.execution_start = None
        self.constraints_met = False
        self.forces_met = False
        self.force_threshold_dict = force_threshold_dict

    def run(self):
        now = rospy.Time.now()
        self.terminate = False
        self.total_timeout       = now + self.tlimit_total
        self.convergence_timeout = now + self.tlimit_convergence
        self.execution_start = now

        self.f_add_cb(self.js_callback)

        while not rospy.is_shutdown() and not self.terminate:
            pass

        self.controller.stop()
        return (self.constraints_met or self.forces_met), self.last_feedback

    def js_callback(self, joint_state):
        if self.terminate: # Just in case
            return

        self.trajectory_log.append(StampedData(rospy.Time.from_sec((rospy.Time.now() - self.execution_start).to_sec()), joint_state.copy()))

        now = rospy.Time.now()
        self.robot.set_joint_state(joint_state)
        command = self.controller.get_next_command()
        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()
        new_feedback = 0
        for jc in command.values():
            new_feedback += abs(jc)

        if self.last_update != None:
            if new_feedback > self.dt_threshold:
                self.convergence_timeout = now + self.tlimit_convergence

        self.last_feedback = new_feedback
        self.constraints_met = self.controller.qp_problem_builder.constraints_met()
        self.forces_met =  min([abs(joint_state[j].effort) >= t for j, t in self.force_threshold_dict.items()])

        self.last_update = now

        self.terminate = now >= self.total_timeout or now >= self.convergence_timeout or self.constraints_met or self.forces_met
        # if self.terminate:
        #     print('Termination flag was set!\n  Total-Timeout: {}\n  Conv-Timeout: {}\n  Constraints met: {}\n  Forces met: {}'.format(
        #         now >= self.total_timeout, now >= self.convergence_timeout, self.constraints_met, self.forces_met))

        self.f_send_command(command)


def run_ineq_controller(robot, controller,
                 tlimit_total, tlimit_convergence, agent,
                 dt_threshold=0.02, task_constraints=None):
    runner = InEqRunner(robot, controller, tlimit_total, tlimit_convergence, agent.act, agent.add_js_callback, dt_threshold, task_constraints)
    constraints_met, lf = runner.run()
    return constraints_met, lf, runner.trajectory_log


class ConvergenceTimeoutRunner(object):
    def __init__(self, robot, controller, feedback_attr,
                 tlimit_total, tlimit_convergence,
                 f_send_command, f_add_cb,
                 dt_threshold, total_threshold=1.0):
        self.robot = robot
        self.controller = controller
        self.feedback_attr = feedback_attr
        self.tlimit_total  = rospy.Duration(tlimit_total)
        self.tlimit_convergence = rospy.Duration(tlimit_convergence)
        self.dt_threshold  = dt_threshold
        self.total_threshold = total_threshold
        self.f_send_command = f_send_command
        self.f_add_cb = f_add_cb
        self.last_feedback = 0
        self.last_update   = None
        self.last_feedback_dt = 1.0
        self.trajectory_log = []
        self.execution_start = None

    def run(self):
        now = rospy.Time.now()
        self.terminate = False
        self.total_timeout       = now + self.tlimit_total
        self.convergence_timeout = now + self.tlimit_convergence
        self.execution_start = now

        self.f_add_cb(self.js_callback)

        while not rospy.is_shutdown() and not self.terminate:
            pass

        return self.last_feedback, self.last_feedback_dt

    def js_callback(self, joint_state):
        if self.terminate: # Just in case
            return

        #print('Received new joint state')

        self.trajectory_log.append(StampedData(rospy.Time.from_sec((rospy.Time.now() - self.execution_start).to_sec()), joint_state.copy()))

        now = rospy.Time.now()
        self.robot.set_joint_state(joint_state)
        command = self.controller.get_next_command()
        new_feedback = getattr(self.controller, self.feedback_attr)
        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()

        if self.last_update != None:
            dt = now - self.last_update
            self.last_feedback_dt = (new_feedback - self.last_feedback) / dt.to_sec()
            #print('   lf: {}\n   nf: {}\nlf_dt: {}\n   dt: {}'.format(str(self.last_feedback), str(new_feedback), str(self.last_feedback_dt), str(dt.to_sec())))
            if self.last_feedback_dt > self.dt_threshold and new_feedback <= self.total_threshold:
                self.convergence_timeout = now + self.tlimit_convergence

        self.last_feedback = new_feedback
        self.last_update   = now

        self.terminate = now >= self.total_timeout or now >= self.convergence_timeout

        self.f_send_command(command)


def run_convergence_controller(robot, controller, feedback_attr,
                 tlimit_total, tlimit_convergence, agent,
                 dt_threshold, total_threshold=1.0):
    runner = ConvergenceTimeoutRunner(robot, controller, feedback_attr, tlimit_total, tlimit_convergence, agent.act, agent.add_js_callback, dt_threshold, total_threshold)
    cm, lf = runner.run()
    return cm, lf, runner.trajectory_log
