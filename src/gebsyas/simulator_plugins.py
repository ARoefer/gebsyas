import rospy
import math
import numpy as np
import gebsyas.np_transformations as nt

from iai_bullet_sim.basic_simulator import SimulatorPlugin, invert_transform
from iai_bullet_sim.rigid_body import RigidBody
from iai_bullet_sim.multibody import SimpleBaseDriver

from gebsyas.msg import Pose2DStamped as Pose2DStampedMsg
from gebsyas.utils import expr_to_rosmsg
from gebsyas.simulator import frame_tuple_to_sym_frame
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.np_transformations import *
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
        return FullPerceptionPublisher(init_dict['topic_prefix'])


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
        self._enabled = True
        self.object_cov = {}
        self.visualizer = ROSVisualizer('probabilistic_vis', 'map')
        self.projection = np.identity(4)
        self.last_camera_position = None#np.array([0, 0, 0, 1])
        self.R  = np.random.random((3, 3))
        self.Q  = np.concatenate((np.concatenate((np.random.random((3, 3)) * 0.01, np.array([[0]*3]))), np.array([[0]]*4)), 1)


        print(self.c_pos_igain_matrix)


    def post_physics_update(self, simulator, deltaT):
        """Implements post physics step behavior.

        :type simulator: BasicSimulator
        :type deltaT: float
        """
        if not self._enabled:
            return

        cf_tuple = self.multibody.get_link_state(self.camera_link).worldFrame
        camera_frame = nt.translation_matrix(cf_tuple.position) * nt.quaternion_matrix(cf_tuple.quaternion)
        

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
                    object_cov = np.identity(6)#.col_join(zeros(3)).row_join(zeros(3).col_join(ones(3)))
                    self.object_cov[name] = object_cov
                    self.message_templates[name] = msg
                else:
                    msg = self.message_templates[name]
                    object_cov = self.object_cov[name]

                tpose   = body.pose()
                obj_pos = point3(*tpose.position)
                c2o     = obj_pos - pos_of(camera_frame) 
                dist    = norm(c2o)
                if dist < self.far and dist > self.near and acos(dot(c2o, x_of(camera_frame)) / dist) < self.fov * 0.5:

                    subs = {self.dt_sym: deltaT, self.dist_sym: dist}

                    inv_pose  = frame_tuple_to_sym_frame(invert_transform(tpose))[:3, :3]
                    current_pos_gain_in_obj  = (inv_pose * current_pos_gain_in_world).subs(subs) 
                    positive_pos_gain_in_obj = Matrix([[min(1, abs(current_pos_gain_in_obj[x,y])) 
                                                        for x in range(current_pos_gain_in_obj.cols)] 
                                                        for y in range(current_pos_gain_in_obj.rows)])
                    current_rot_gain_in_obj  = (inv_pose * current_rot_gain_in_world).subs(subs)
                    positive_rot_gain_in_obj = Matrix([[min(1, abs(current_rot_gain_in_obj[x,y])) 
                                                        for x in range(current_rot_gain_in_obj.cols)] 
                                                        for y in range(current_rot_gain_in_obj.rows)])
                    gainmatrix = positive_pos_gain_in_obj.col_join(zeros(3)).row_join(zeros(3).col_join(current_rot_gain_in_obj))
                    object_cov = object_cov * gainmatrix
                    print(object_cov)
                    self.object_cov[name] = object_cov

                    #print(object_cov)

                x_vec = x_of(object_cov)
                y_vec = y_of(object_cov)
                z_vec = z_of(object_cov)

                self.visualizer.draw_vector('cov', obj_pos, x_vec, g=0, b=0)
                self.visualizer.draw_vector('cov', obj_pos, y_vec, r=0, b=0)
                self.visualizer.draw_vector('cov', obj_pos, z_vec, r=0, g=0)

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
        return ProbPerceptionPublisher(body,
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
            self.object_cov[name]   = eye(6)#.col_join(zeros(3)).row_join(zeros(3).col_join(ones(3)))

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
        return LocalizationPublisher(body, init_dict['topic_prefix'])
