import rospy
import traceback

from time import time
from giskardpy import print_wrapper
from gebsyas.bc_controller_wrapper import BCControllerWrapper
from gebsyas.utils import StampedData


class InEqController(BCControllerWrapper):
    """
    @brief      This controller connects a robot to a set of inequality constraints.
    """
    def __init__(self, robot, logging=print_wrapper):
        """Constructor. Receives a robot to use and soft constraints to abide by.
           Additionally a backend, a weight for the constraints and a custom logger can be supplied.
        """
        super(InEqController, self).__init__(robot, logging)


class InEqRunner(object):
    """This class runs an inequality controller. It processes joint state updates and new commands.
       It also terminates controller execution when all constraints are met, when the sum of all
       commands is smaller than a given value for a given time, or when a timeout is reached.
    """
    def __init__(self, robot, controller, tlimit_total,
                 tlimit_convergence, f_send_command,
                 f_add_cb, dt_threshold=0.02, task_constraints=None):
        """
        Constructor.
        Needs a robot,
        the controller to run,
        a total timeout,
        a timeout for the low activity commands,
        a function to send commands,
        a function to add itself as listener for joint states,
        the threshold for low activity,
        the names of the constraints to monitor for satisfaction
        """
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
        """Starts the run of the controller."""
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
        """Callback processing joint state updates, checking constraints and generating new commands."""
        if self.terminate: # Just in case
            return

        self.trajectory_log.append(StampedData(rospy.Time.from_sec((rospy.Time.now() - self.execution_start).to_sec()), joint_state.copy()))

        now = rospy.Time.now()
        self.controller.set_robot_js(joint_state)
        try:
            command = self.controller.get_cmd()
        except Exception as e:
            self.terminate = True
            traceback.print_exc()
            print(e)
            return
        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()
        new_feedback = 0
        for jc in command.values():
            new_feedback += abs(jc)

        if self.last_update != None:
            if new_feedback > self.dt_threshold:
                self.convergence_timeout = now + self.tlimit_convergence

        self.last_feedback = new_feedback
        self.constraints_met = self.controller.qp_problem_builder.constraints_met(names=self.task_constraints, lbThreshold=0.02, ubThreshold=-0.02)
        self.last_update   = now

        self.terminate = now >= self.total_timeout or now >= self.convergence_timeout or self.constraints_met

        self.f_send_command(command)


class InEqFFBRunner(object):
    """This class runs an inequality controller. It processes joint state updates and new commands.
       It also terminates controller execution when all constraints are met, when the sum of all
       commands is smaller than a given value for a given time, or when a timeout is reached.
       Additionally the runner terminates when specified joints exert a given effort.
    """
    def __init__(self, robot, controller, tlimit_total,
                 tlimit_convergence, force_threshold_dict, f_send_command,
                 f_add_cb, dt_threshold=0.02):
        """
        Constructor.
        Needs a robot,
        the controller to run,
        a total timeout,
        a timeout for the low activity commands,
        a dictionary specifying which joints to monitor for the effort-timeout,
        a function to send commands,
        a function to add itself as listener for joint states,
        the threshold for low activity
        """
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
        self.set_robot_js(joint_state)
        command = self.controller.get_cmd()
        #print('\n'.join(['{:>20}: {}'.format(name, vel) for name, vel in command.items()]))
        #self.controller.qp_problem_builder.print_jacobian()
        new_feedback = 0
        for jc in command.values():
            new_feedback += abs(jc)

        if self.last_update != None:
            if new_feedback > self.dt_threshold:
                self.convergence_timeout = now + self.tlimit_convergence

        self.last_feedback = new_feedback
        self.constraints_met = self.controller.qp_problem_builder.constraints_met(lbThreshold=0.02, ubThreshold=-0.02)
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
    """Comfort function for easily instantiating and running an inequality runner."""
    runner = InEqRunner(robot, controller, tlimit_total, tlimit_convergence, agent.act, agent.add_js_callback, dt_threshold, task_constraints)
    constraints_met, lf = runner.run()
    return constraints_met, lf, runner.trajectory_log
