import rospy
import math
import numpy as np
from giskardpy.symengine_wrappers import norm, acos, sin, cos, vector3, eye, diag, point3

STD_OCM_GROWTH = 0.2

def abs_min(x, y):
    return x if abs(x) <= abs(y) else y

def c_dist(alpha, beta):
    return math.acos(min(1, max(-1, math.cos(alpha) * math.cos(beta) + math.sin(alpha) * math.sin(beta))))

def sym_c_dist(alpha, beta):
    return acos(cos(alpha) * cos(beta) + sin(alpha) * sin(beta))

def wrap_to_circle(value):
    return math.atan2(math.sin(value), math.cos(value))


class OcclusionMap(object):
    def __init__(self):

        # Coordinates Yaw, Pitch
        self.min_corner = None
        self.max_corner = None
        self.coords = None

    def update(self, obs_position, center_pos):
        o2c = obs_position - center_pos
        dist = norm(o2c)
        self.coords = diag(1,1,0,1) * center_pos

        out_yaw = 0
        if dist >= 0.0001:
            # from -pi to pi
            yaw   = math.atan2(o2c[1], o2c[0])
            pitch = math.asin(o2c[2] / norm(o2c))
            if self.min_corner is not None:
                min_dist = c_dist(yaw, self.min_corner[0])
                max_dist = c_dist(yaw, self.max_corner[0])
                if min_dist < max_dist:
                    if min_dist < STD_OCM_GROWTH:
                        self.min_corner = (wrap_to_circle(yaw - STD_OCM_GROWTH),
                                           min(self.min_corner[1], pitch - STD_OCM_GROWTH))
                        self.max_corner = (self.max_corner[0],
                                           max(self.max_corner[1], pitch + STD_OCM_GROWTH))
                else:
                    if max_dist < STD_OCM_GROWTH:
                        self.max_corner = (wrap_to_circle(yaw + STD_OCM_GROWTH),
                                           max(self.max_corner[1], pitch + STD_OCM_GROWTH))
                        self.min_corner = (self.min_corner[0],
                                           min(self.min_corner[1], pitch - STD_OCM_GROWTH))
            else:
                self.min_corner = (wrap_to_circle(yaw - STD_OCM_GROWTH), pitch - STD_OCM_GROWTH)
                self.max_corner = (wrap_to_circle(yaw + STD_OCM_GROWTH), pitch + STD_OCM_GROWTH)
                out_yaw = self.max_corner[0]

            #print('Occlusion bounds:\n Yaw-Min: {}\n Yaw-Max: {}'.format(self.min_corner[0], self.max_corner[0]))
            out_yaw = self.min_corner[0] if c_dist(yaw, self.min_corner[0]) < c_dist(yaw, self.max_corner[0]) else self.max_corner[0]

        return (out_yaw, self.max_corner[1])

    def get_closest_corner(self, obs_position, center_pos):
        o2c = obs_position - center_pos
        dist = norm(o2c)

        out_yaw = 0
        if dist >= 0.0001:
            yaw   = math.atan2(o2c[1], o2c[0])
            pitch = math.asin(o2c[2] / norm(o2c))
            out_yaw = self.min_corner[0] if c_dist(yaw, self.min_corner[0]) < c_dist(yaw, self.max_corner[0]) else self.max_corner[0]
        return (out_yaw, self.max_corner[1])

    def get_occluded_area(self):
        if self.min_corner[0] < self.max_corner[0]:
            yaw_dist = self.max_corner[0] - self.min_corner[0]
        else:
            yaw_dist = 2 * math.pi - (self.min_corner[0] - self.max_corner[0])
        return (yaw_dist, self.max_corner[1] - self.min_corner[1])

    def is_closed(self):
        if self.coords != None:
                print norm(self.coords - point3(-3.76, 5.38, 0))
                print self.min_corner[0]
                print self.max_corner[0]
                if norm(self.coords - point3(-3.76, 5.38, 0)) < 1.0 and self.min_corner[0] <= -0.1 and self.max_corner[0] >= 0.1:
                    return True
        return c_dist(self.min_corner[0], self.max_corner[0]) <= 0.2 and wrap_to_circle(self.min_corner[0] - self.max_corner[0]) > 0

    def draw(self, visualizer, position, radius=1.0, h_res=32, v_res=16):
        h_res_factor = 2 * math.pi / h_res
        v_res_factor = 2 * math.pi / v_res

        if self.min_corner[0] < self.max_corner[0]:
            yaw_dist = self.max_corner[0] - self.min_corner[0]
        else:
            yaw_dist = 2 * math.pi - (self.min_corner[0] - self.max_corner[0])

        n_h_steps = max(int(yaw_dist / h_res_factor), 2)
        n_v_steps = max(int((self.max_corner[1] - self.min_corner[1]) / v_res_factor), 2)

        h_step_width = (yaw_dist) / n_h_steps
        v_step_width = (self.max_corner[1] - self.min_corner[1]) / n_v_steps

        points = []
        for x in range(n_h_steps):
            for y in range(n_v_steps):
                yaw = self.min_corner[0] + x * h_step_width
                pitch = self.min_corner[1] + y * v_step_width
                cos_p = math.cos(pitch)
                points.append(position + vector3(math.cos(yaw) * cos_p * radius,
                                                 math.sin(yaw) * cos_p * radius,
                                                 math.sin(pitch) * radius))
        visualizer.draw_cube_batch('occlusion_map', eye(4), 0.05, points)


class ValueLogger(object):
    def __init__(self, log_length=20, initial_value=0, delay=0.1):
        self.log_length = log_length
        self.log_cursor = 0
        self.initial_value = initial_value
        self.last_update = None
        self.delay = delay
        self.reset()

    def log(self, value):
        now = rospy.Time.now()
        if self.last_update != None:
            deltaT = (now - self.last_update).to_sec()
            if deltaT >= self.delay:
                self._true_log(value, deltaT)
                self.log_cursor = (self.log_cursor + 1) % self.log_length
                self.last_update = now
                return True
        else:
            self.last_update = now
            return False

    def _true_log(self, value, deltaT):
        self.values[self.log_cursor] = value

    def avg(self):
        return np.average(self.values)

    def abs_avg(self):
        return np.average(np.abs(self.values))

    def reset(self):
        self.values = np.ones(self.log_length) * self.initial_value


class FirstDerivativeLogger(ValueLogger):
    def __init__(self, log_length=20, initial_value=0, delay=0):
        super(FirstDerivativeLogger, self).__init__(log_length, initial_value, delay)
        self.last_value = None

    def _true_log(self, value, deltaT):
        if self.last_value != None:
            super(FirstDerivativeLogger, self)._true_log((value - self.last_value) / deltaT, deltaT)
        self.last_value  = value

    def reset(self):
        super(FirstDerivativeLogger, self).reset()
        self.last_update = None

class FirstDerivativePositionLogger(ValueLogger):
    def _true_log(self, value, deltaT):
        if self.last_value != None:
            super(FirstDerivativePositionLogger, self)._true_log(norm(value - self.last_value) / deltaT, deltaT)
        self.last_value  = value


class CircularFirstDerivativeLogger(FirstDerivativeLogger):
    def _true_log(self, value, deltaT):
        if self.last_value != None:
            ValueLogger._true_log(self, c_dist(value, self.last_value) / deltaT, deltaT)
        self.last_value = value