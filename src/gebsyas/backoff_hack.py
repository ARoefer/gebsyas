from gebsyas.basic_controllers import InEqController

from giskardpy import print_wrapper
from giskardpy.symengine_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint as SC

from math import atan2

class BackoffHack(InEqController):
    def __init__(self, robot, backoff_dist=0.2, logging=print_wrapper):
        super(BackoffHack, self).__init__(robot, logging, False)

        self.goal_x = Symbol('goal_x')
        self.goal_y = Symbol('goal_y')
        self.goal_theta = Symbol('goal_theta')

        goal    = point3(self.goal_x, self.goal_y, 0)
        normal  = vector3(cos(self.goal_theta), sin(self.goal_theta), 0)

        # Base things
        r_x     = x_of(robot.world_transform)
        r_p     = diag(1,1,0,1) * pos_of(robot.world_transform)
        dist    = dot(normal, r_p - goal)
        o_align = dot(r_x, normal)

        c_dist  = SC(-100, (-backoff_dist - dist) * o_align, 1, dist)
        c_align = SC(1 - o_align, 1 - o_align, 1, o_align)

        # Head things
        camera_y    = y_of(robot.camera.pose)
        h_align   = dot(camera_y, normal)

        c_h_align = SC(-h_align, -h_align, 1, h_align)

        self.init({'orientation alignment': c_align,
                   'positional goal':       c_dist,
                   'head alignment':        c_h_align})

    def set_goal(self, localization_pose, object_pose):
        self.current_subs[self.goal_x] = localization_pose.x
        self.current_subs[self.goal_y] = localization_pose.y

        pos = pos_of(object_pose)
        self.current_subs[self.goal_theta] = atan2(pos[1] - localization_pose.y, 
                                                   pos[0] - localization_pose.x)
