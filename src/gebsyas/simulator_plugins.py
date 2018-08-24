import rospy
import math
import numpy as np

from iai_bullet_sim.basic_simulator import SimulatorPlugin, invert_transform
from iai_bullet_sim.rigid_body import RigidBody
from iai_bullet_sim.multibody import SimpleBaseDriver

from gebsyas.msg import Pose2DStamped as Pose2DStampedMsg
from gebsyas.utils import expr_to_rosmsg
from gebsyas.simulator import frame_tuple_to_sym_frame
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.np_transformations import *
from giskardpy.symengine_wrappers import *
from gop_gebsyas_msgs.msg import ProbObject as POMsg
from gop_gebsyas_msgs.msg import ProbObjectList as POLMsg
from gop_gebsyas_msgs.msg import SearchObject as SearchObjectMsg
from gop_gebsyas_msgs.msg import ObjectPoseGaussianComponent as OPGCMsg

from symengine import ones


class FetchDriver(SimpleBaseDriver):
    def update_velocities(self, robot_data, velocities_dict):
        """Updates a given velocity command."""
        super(FetchDriver, self).update_velocities(robot_data, velocities_dict)
        if 'gripper_joint' in velocities_dict:
            gripper_vel = velocities_dict['gripper_joint']
            velocities_dict['r_gripper_finger_joint'] = gripper_vel
            velocities_dict['l_gripper_finger_joint'] = gripper_vel

    @classmethod
    def factory(cls, config_dict):
        return cls(config_dict['max_lin_vel'], config_dict['max_ang_vel'])


class FullPerceptionPublisher(SimulatorPlugin):
    def __init__(self, topic_prefix=''):
        super(FullPerceptionPublisher, self).__init__('FullPerceptionPublisher')
        self.topic_prefix = topic_prefix
        self.publisher = rospy.Publisher('{}/perceived_objects'.format(topic_prefix), POLMsg, queue_size=1, tcp_nodelay=True)
        self.message_templates = {}
        self._enabled = True
        self.msg_list = POLMsg()


    def post_physics_update(self, simulator, deltaT):
        """Implements post physics step behavior.

        :type simulator: BasicSimulator
        :type deltaT: float
        """
        if not self._enabled:
            return

        self.msg_list.header.stamp = rospy.Time.now()

        for name, body in simulator.bodies.items():
            if isinstance(body, RigidBody):
                if not name in self.message_templates:
                    msg = POMsg()
                    msg.id = body.bId()
                    msg.name = name.split('.')[0]
                    # if body.type == 'box':
                    #     for a, v in zip(['x', 'y', 'z'], body.extents):
                    #         setattr(msg.dimensions, a, v)
                    # elif body.type == 'sphere':
                    #     for a in ['x', 'y', 'z']:
                    #         setattr(msg.dimensions, a, body.radius * 2)
                    # else:
                    #     for a in ['x', 'y']:
                    #         setattr(msg.dimensions, a, body.radius * 2)
                    #     msg.dimensions.z = body.height
                    msg.cov_pose.covariance = ([0,0,0,0,0,0,0] * 6)[:36]
                    self.message_templates[name] = msg
                    self.msg_list.objects.append(msg)
                else:
                    msg = self.message_templates[name]
                
                msg.header.stamp   = rospy.Time.now()
                pose = body.pose()
                msg.cov_pose.pose.position  = expr_to_rosmsg(pose.position)
                msg.cov_pose.pose.orientation.x = pose.quaternion[0]
                msg.cov_pose.pose.orientation.y = pose.quaternion[1]
                msg.cov_pose.pose.orientation.z = pose.quaternion[2]
                msg.cov_pose.pose.orientation.w = pose.quaternion[3]
        
        self.publisher.publish(self.msg_list)


    def disable(self, simulator):
        """Stops the execution of this plugin.

        :type simulator: BasicSimulator
        """
        self._enabled = False
        self.publisher.unregister()


    def to_dict(self, simulator):
        """Serializes this plugin to a dictionary.

        :type simulator: BasicSimulator
        :rtype: dict
        """
        return {'topic_prefix': self.topic_prefix}

    @classmethod
    def factory(cls, simulator, init_dict):
        return cls(init_dict['topic_prefix'])


class ProbPerceptionPublisher(SimulatorPlugin):
    def __init__(self, multibody, camera_link, fov, near, far, h_precision_gain, d_precision_gain, topic_prefix=''):
        super(ProbPerceptionPublisher, self).__init__('ProbPerceptionPublisher')
        self.topic_prefix = topic_prefix
        self.publisher = rospy.Publisher('{}/perceived_prob_objects'.format(topic_prefix), SearchObjectMsg, queue_size=1, tcp_nodelay=True)
        self.message_templates = {}
        self.multibody   = multibody
        self.camera_link = camera_link
        self.fov         = fov
        self.near        = near
        self.far         = far
        self.camera_h_gain = h_precision_gain
        self.camera_d_gain = d_precision_gain 
        self._enabled = True
        self.object_cov = {}
        self.visualizer = ROSVisualizer('probabilistic_vis', 'map')


    def post_physics_update(self, simulator, deltaT):
        """Implements post physics step behavior.

        :type simulator: BasicSimulator
        :type deltaT: float
        """
        if not self._enabled:
            return

        cf_tuple = self.multibody.get_link_state(self.camera_link).worldFrame
        camera_frame = frame3_quaternion(cf_tuple.position.x, cf_tuple.position.y, cf_tuple.position.z, *cf_tuple.quaternion)
        cov_proj = rot_of(camera_frame)[:3, :3]
        inv_cov_proj = cov_proj.T

        self.visualizer.begin_draw_cycle()


        for name, body in simulator.bodies.items():
            if isinstance(body, RigidBody):
                if not name in self.message_templates:
                    msg = SearchObjectMsg()
                    msg.id = body.bId()
                    msg.name = name.split('.')[0]
                    # if body.type == 'box':
                    #     for a, v in zip(['x', 'y', 'z'], body.extents):
                    #         setattr(msg.dimensions, a, v)
                    # elif body.type == 'sphere':
                    #     for a in ['x', 'y', 'z']:
                    #         setattr(msg.dimensions, a, body.radius * 2)
                    # else:
                    #     for a in ['x', 'y']:
                    #         setattr(msg.dimensions, a, body.radius * 2)
                    #     msg.dimensions.z = body.height
                    opgc = OPGCMsg()
                    opgc.weight = 1.0
                    msg.object_pose_gmm.append(opgc)
                    object_cov = eye(6)#.col_join(zeros(3)).row_join(zeros(3).col_join(ones(3)))
                    self.object_cov[name] = object_cov
                    self.message_templates[name] = msg
                else:
                    msg = self.message_templates[name]
                    object_cov = self.object_cov[name]

                tpose   = body.pose()
                obj_pos = point3(*tpose.position)
                c2o  = obj_pos - pos_of(camera_frame)
                dist = norm(c2o) 
                if dist < self.far and dist > self.near and dot(c2o, x_of(camera_frame)) > cos(self.fov * 0.5) * dist:
                    
                    s_h = min(1, max(0.01, 1 - self.camera_h_gain / dist * deltaT))
                    s_d = min(1, max(0.01, 1 - self.camera_d_gain / dist * deltaT))
                    S_pos = diag(s_d, s_h, s_h)
                    S_rot = diag(s_h, s_d, s_d)
                    new_pos_cov = cov_proj * S_pos * inv_cov_proj * object_cov[:3, :3]
                    new_rot_cov = cov_proj * S_rot * inv_cov_proj * object_cov[3:, 3:]
                    for x in range(3):
                        new_pos_cov[x,x] = max(0.0001, new_pos_cov[x, x])

                    object_cov = new_pos_cov.col_join(zeros(3)).row_join(zeros(3).col_join(new_rot_cov))

                    #print(object_cov)
                    self.object_cov[name] = object_cov

                    #print(object_cov)

                np_pos_cov = np.array(object_cov[:3, :3].tolist(), dtype=float).reshape((3,3))
                w, v = np.linalg.eig(np_pos_cov)
                pos_eig = v * w


                if np.isrealobj(pos_eig):
                    x_vec = vector3(*pos_eig[:, 0])
                    y_vec = vector3(*pos_eig[:, 1])
                    z_vec = vector3(*pos_eig[:, 2])

                    self.visualizer.draw_vector('cov', obj_pos, x_vec, r=0.5, g=0.5, b=0.5)
                    self.visualizer.draw_vector('cov', obj_pos, y_vec, r=0.5, g=0.5, b=0.5)
                    self.visualizer.draw_vector('cov', obj_pos, z_vec, r=0.5, g=0.5, b=0.5)

                msg.header.stamp = rospy.Time.now()
                msg.object_pose_gmm[0].cov_pose.pose.position  = expr_to_rosmsg(tpose.position)
                msg.object_pose_gmm[0].cov_pose.pose.orientation.x = tpose.quaternion[0]
                msg.object_pose_gmm[0].cov_pose.pose.orientation.y = tpose.quaternion[1]
                msg.object_pose_gmm[0].cov_pose.pose.orientation.z = tpose.quaternion[2]
                msg.object_pose_gmm[0].cov_pose.pose.orientation.w = tpose.quaternion[3]
                msg.object_pose_gmm[0].cov_pose.covariance = list(object_cov)
        
                self.publisher.publish(msg)

        self.visualizer.render()



    def disable(self, simulator):
        """Stops the execution of this plugin.

        :type simulator: BasicSimulator
        """
        self._enabled = False
        self.publisher.unregister()


    def to_dict(self, simulator):
        """Serializes this plugin to a dictionary.

        :type simulator: BasicSimulator
        :rtype: dict
        """
        return {'body': simulator.get_body_id(self.body.bId()),
                'camera_link':  self.camera_link,
                'fov':          self.fov,
                'near':         self.near,
                'far':          self.far,
                'h_precision_gain':  self.h_precision_gain,
                'd_precision_gain':  self.d_precision_gain,
                'topic_prefix': self.topic_prefix}

    @classmethod
    def factory(cls, simulator, init_dict):
        body = simulator.get_body(init_dict['body'])
        if body is None:
            raise Exception('Body "{}" does not exist in the context of the given simulation.'.format(init_dict['body']))
        return cls(body,
                   init_dict['camera_link'],
                   init_dict['fov'],
                   init_dict['near'],
                   init_dict['far'],
                   init_dict['h_precision_gain'],
                   init_dict['d_precision_gain'],
                   init_dict['topic_prefix'])


    def reset(self, simulator):
        for name, msg in self.message_templates.items():
            msg.object_pose_gmm[0].cov_pose.covariance = ([1,0,0,0,0,0,0] * 6)[:36]
            self.object_cov[name] = eye(6)#.col_join(zeros(3)).row_join(zeros(3).col_join(ones(3)))

class LocalizationPublisher(SimulatorPlugin):
    def __init__(self, body, topic_prefix=''):
        super(LocalizationPublisher, self).__init__('LocalizationPublisher')
        self.body = body
        self.topic_prefix = topic_prefix
        self.publisher = rospy.Publisher('{}/localization'.format(topic_prefix), Pose2DStampedMsg, queue_size=1, tcp_nodelay=True)
        self._enabled = True


    def post_physics_update(self, simulator, deltaT):
        """Implements post physics step behavior.

        :type simulator: BasicSimulator
        :type deltaT: float
        """
        if not self._enabled:
            return

        pose = self.body.pose()
        msg = Pose2DStampedMsg()
        msg.header.stamp = rospy.Time.now()
        msg.pose.x = pose.position[0]
        msg.pose.y = pose.position[1]
        x2 = pose.quaternion[0] * pose.quaternion[0]
        y2 = pose.quaternion[1] * pose.quaternion[1]
        z2 = pose.quaternion[2] * pose.quaternion[2]
        w2 = pose.quaternion[3] * pose.quaternion[3]
        msg.pose.theta = z = math.atan2(2 * pose.quaternion[0] * pose.quaternion[1] + 2 * pose.quaternion[3] * pose.quaternion[2], w2 + x2 - y2 - z2)
        self.publisher.publish(msg)

    def disable(self, simulator):
        """Stops the execution of this plugin.

        :type simulator: BasicSimulator
        """
        self._enabled = False
        self.publisher.unregister()


    def to_dict(self, simulator):
        """Serializes this plugin to a dictionary.

        :type simulator: BasicSimulator
        :rtype: dict
        """
        return {'body': simulator.get_body_id(self.body.bId()),
                'topic_prefix': self.topic_prefix}

    @classmethod
    def factory(cls, simulator, init_dict):
        body = simulator.get_body(init_dict['body'])
        if body is None:
            raise Exception('Body "{}" does not exist in the context of the given simulation.'.format(init_dict['body']))
        return cls(body, init_dict['topic_prefix'])
