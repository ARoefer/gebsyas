import rospy

from giskardpy.symengine_wrappers import unitX, vector3
from gebsyas.constants import LBA_BOUND, UBA_BOUND
from gebsyas.bullet_based_controller import InEqBulletController
from gebsyas.data_structures import StampedData, JointState
from gebsyas.utils import pi
from math import sin, cos

class StdIntegrator(object):
    def __init__(self, joints, symbol_map):
        self.joints    = joints
        self.symbol_map = symbol_map

    def integrate(self, positions, velocities, delta_t):
        positions.update({self.symbol_map[j]: max(j.lower, min(positions[n] + delta_t * velocities[n],upper)) for n, j in self.joints.items() if j in velocities and j in positions})

class FetchIntegrator(StdIntegrator):
    def integrate(self, positions, velocities, delta_t):
        lin_vel = 0
        ang_vel = 0
        if 'base_linear_joint' in velocities:
            lin_vel = velocities['base_linear_joint']
            del velocities['base_linear_joint']
        if 'base_angular_joint' in velocities:
            ang_vel = velocities['base_angular_joint']
            del velocities['base_angular_joint']

        c_ang = positions[self.symbol_map['localization_z_ang']]
        c_x   = positions[self.symbol_map['localization_x']]
        c_y   = positions[self.symbol_map['localization_y']]
        # if abs(ang_vel) > 0.001:
        #     radius = lin_vel / (ang_vel * pi) 
        #     new_x  = c_x + cos(c_ang + ang_vel * delta_t) * radius
        #     new_y  = c_y + sin(c_ang + ang_vel * delta_t) * radius
        # else:
        new_x = c_x + cos(c_ang) * lin_vel * delta_t
        new_y = c_y + sin(c_ang) * lin_vel * delta_t
        new_ang = c_ang + ang_vel * delta_t
        positions[self.symbol_map['localization_z_ang']] = new_x
        positions[self.symbol_map['localization_x']]     = new_y
        positions[self.symbol_map['localization_y']]     = new_ang

        super(FetchIntegrator, self).integrate(positions, velocities, delta_t)


class LocalizationIntegrator(StdIntegrator):
    def integrate(self, positions, velocities, delta_t):
        if 'localization_x' in velocities:
            positions[self.symbol_map['localization_x']] = positions[self.symbol_map['localization_x']] + velocities['localization_x'] * delta_t
        if 'localization_y' in velocities:
            positions[self.symbol_map['localization_y']] = positions[self.symbol_map['localization_y']] + velocities['localization_y'] * delta_t
        if 'localization_z_ang' in velocities:
            positions[self.symbol_map['localization_z_ang']] = positions[self.symbol_map['localization_z_ang']] + velocities['localization_z_ang'] * delta_t

        super(LocalizationIntegrator, self).integrate(positions, velocities, delta_t)    


class HeadlessConstraintRunner(object):
    def __init__(self, controller, integrator, visualizer=None):
        self.controller = controller
        self.integrator = integrator
        self.trajectory_log = []
        self.visualizer = visualizer


    def run(self, iterations=100, time_step=0.05):
        for x in range(iterations):
            self.integrator.integrate(self.controller.current_subs, self.controller.get_cmd(), time_step)
            positions = {j: JointState(self.controller.current_subs[s], 0, 0) for j, s in self.integrator.symbol_map.items()}
            self.trajectory_log.append(StampedData(rospy.Time.from_sec(x * time_step), positions))
            if self.visualizer != None:
                self.visualizer.begin_draw_cycle()
                self.visualizer.draw_robot_pose('runner', self.controller.robot, {j: s.position for j, s in positions.items()})
                self.visualizer.render()

            if self.controller.qp_problem_builder.constraints_met(lbThreshold=LBA_BOUND, ubThreshold=UBA_BOUND):
                return True

        if self.controller.qp_problem_builder.constraints_met(lbThreshold=LBA_BOUND, ubThreshold=UBA_BOUND):
            return True
        return False


class CollisionResolverRunner(HeadlessConstraintRunner):
    def __init__(self, controller, integrator, visualizer=None):
        if not isinstance(controller, InEqBulletController):
            raise Exception('This runner expects the controller to be subclassed from InEqBulletController.')
        super(CollisionResolverRunner, self).__init__(controller, integrator, visualizer)

    def run(self, iterations=100, time_step=0.2):
        for x in range(iterations):
            self.integrator.integrate(self.controller.current_subs, self.controller.get_cmd(do_avoidance=False), time_step)
            positions = {j: JointState(self.controller.current_subs[s], 0, 0) for j, s in self.integrator.symbol_map.items()}
            self.trajectory_log.append(StampedData(rospy.Time.from_sec(x * time_step), positions))
            
            if self.visualizer != None:
                self.visualizer.begin_draw_cycle()
                self.visualizer.draw_robot_pose('runner', self.controller.robot, {j: s.position for j, s in positions.items()})
                self.visualizer.render()

            if self.controller.qp_problem_builder.constraints_met(lbThreshold=LBA_BOUND, ubThreshold=UBA_BOUND):
                break
        return super(CollisionResolverRunner, self).run(iterations, time_step)
