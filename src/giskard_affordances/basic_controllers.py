import rospy
from time import time
from giskardpy.qpcontroller import QPController
from giskardpy.qp_problem_builder import SoftConstraint
from giskardpy.symengine_wrappers import *
from giskardpy.input_system import FrameInput
from giskard_affordances.utils import StampedData
from giskard_affordances.dl_reasoning import DLSymbolic
from sensor_msgs.msg import JointState

class SimpleCartesianController(QPController):
    def __init__(self, robot, movable_frame, builder_backend=None, weight=1):
        self.weight = weight
        self.movable_frame = movable_frame
        self.feedback = 0.0
        super(SimpleCartesianController, self).__init__(robot, builder_backend)

    # @profile
    def add_inputs(self, robot):
        self.goal_eef = FrameInput('', 'goal')

    # @profile
    def make_constraints(self, robot):
        t = time()
        # start_position = pos_of(start_pose)

        current_pose = self.movable_frame
        current_rotation = rot_of(current_pose)

        goal_position = self.goal_eef.get_position()
        goal_rotation = self.goal_eef.get_rotation()
        # goal_r = goal_rotation[:3,:3].reshape(9,1)

        #pos control
        dist = norm(pos_of(current_pose) - goal_position)

        #rot control
        dist_r = 2 - dot(current_rotation[:3, :1], goal_rotation[:3, :1]) - dot(current_rotation[:3, 1:2], goal_rotation[:3, 1:2])

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
        print('make constraints took {}'.format(time() - t))

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
    def __init__(self, robot, expression, limit, builder_backend=None, weight=1):
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
        print('make constraints took {}'.format(time() - t))
        #print('Expression: {}'.format(str(self.expression)))

    def set_goal(self, goal):
        pass

    def get_next_command(self):
        cmd = super(SimpleExpressionController, self).get_next_command()
        self.feedback = self.expression.subs(self.get_state())
        return cmd


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
    lf, lf_dt = runner.run()
    return lf, lf_dt, runner.trajectory_log
