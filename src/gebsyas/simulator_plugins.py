import rospy
import math
import numpy as np
import pybullet as pb

from iai_bullet_sim.basic_simulator import SimulatorPlugin, invert_transform, hsva_to_rgba
from iai_bullet_sim.rigid_body import RigidBody
from iai_bullet_sim.multibody import SimpleBaseDriver

from gebsyas.msg import LocalizedPoseStamped as LPSMsg
from gebsyas.utils import expr_to_rosmsg
from gebsyas.simulator import frame_tuple_to_sym_frame
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.np_transformations import *
from giskardpy.symengine_wrappers import *
from gop_gebsyas_msgs.msg import ProbObject as POMsg
from gop_gebsyas_msgs.msg import ProbObjectList as POLMsg
from gop_gebsyas_msgs.msg import SearchObject as SearchObjectMsg
from gop_gebsyas_msgs.msg import SearchObjectList as SearchObjectListMsg
from gop_gebsyas_msgs.msg import ObjectPoseGaussianComponent as OPGCMsg

from sensor_msgs.msg import LaserScan as LaserScanMsg

from symengine import ones
import symengine as sp


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
                    #z_vec = vector3(*pos_eig[:, 2])
                    x_vec *= 1.0 / norm(x_vec)
                    y_vec *= 1.0 / norm(y_vec)
                    #z_vec *= 1.0 / norm(z_vec)
                    M = x_vec.row_join(y_vec).row_join(cross(x_vec, y_vec)).row_join(obj_pos)

                    self.visualizer.draw_shape('cov', M, w.astype(float), 2, r=0.5, g=0.5, b=0.5, a=0.7)

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
        self.publisher = rospy.Publisher('{}/localization'.format(topic_prefix), LPSMsg, queue_size=1, tcp_nodelay=True)
        self._enabled = True


    def post_physics_update(self, simulator, deltaT):
        """Implements post physics step behavior.

        :type simulator: BasicSimulator
        :type deltaT: float
        """
        if not self._enabled:
            return

        pose = self.body.pose()
        msg = LPSMsg()
        msg.header.stamp  = rospy.Time.now()
        msg.pose.linear.x = pose.position[0]
        msg.pose.linear.y = pose.position[1]
        x2 = pose.quaternion[0] * pose.quaternion[0]
        y2 = pose.quaternion[1] * pose.quaternion[1]
        z2 = pose.quaternion[2] * pose.quaternion[2]
        w2 = pose.quaternion[3] * pose.quaternion[3]
        msg.pose.angular.z = math.atan2(2 * pose.quaternion[0] * pose.quaternion[1] + 2 * pose.quaternion[3] * pose.quaternion[2], w2 + x2 - y2 - z2)
        inv_pos, inv_rot = pb.invertTransform(pose.position, pose.quaternion)
        lv, trash = pb.multiplyTransforms((0,0,0), inv_rot, self.body.linear_velocity(), (0,0,0,1))
        av, trash = pb.multiplyTransforms((0,0,0), inv_rot, self.body.angular_velocity(), (0,0,0,1))
        msg.local_velocity.linear.x = lv[0]
        msg.local_velocity.linear.y = lv[1]
        msg.local_velocity.linear.z = lv[2]
        msg.local_velocity.angular.x = av[0]
        msg.local_velocity.angular.y = av[1]
        msg.local_velocity.angular.z = av[2]
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


class GMMObjectPublisher(SimulatorPlugin):
    def __init__(self, topic_prefix=''):
        super(GMMObjectPublisher, self).__init__('GMMObjectPublisher')
        self.publisher = rospy.Publisher('{}/perceived_prob_objects'.format(topic_prefix), SearchObjectListMsg, queue_size=1, tcp_nodelay=True)
        self.visualizer = ROSVisualizer('gmm_vis', 'map')
        self.message_templates = {}
        self.topic_prefix = topic_prefix
        self._enabled = True

    def post_physics_update(self, simulator, deltaT):
        if not self._enabled:
            return

        msg_total = SearchObjectListMsg()
        self.visualizer.begin_draw_cycle()
        for name, gmm in simulator.gpcs.items():
            if not name in self.message_templates:
                body = simulator.bodies[name]
                msg = SearchObjectMsg()
                msg.id = body.bId()
                msg.name = name.split('.')[0]
                msg.object_pose_gmm.extend([OPGCMsg() for gc in gmm])
                self.message_templates[name] = msg
            else:
                msg = self.message_templates[name]
                if len(gmm) > len(msg.object_pose_gmm):
                    msg.object_pose_gmm.extend([OPGCMsg() for x in range(len(gmm) - len(msg.object_pose_gmm))])
                elif len(gmm) < len(msg.object_pose_gmm):
                    msg.object_pose_gmm = msg.object_pose_gmm[:len(gmm)]

            for x in range(len(gmm)):
                gc = gmm[x]
                gc_cov = gc.cov
                np_pos_cov = np.array(gc_cov[:3, :3].tolist(), dtype=float).reshape((3,3))
                w, v = np.linalg.eig(np_pos_cov)
                pos_eig = v * w

                if np.isrealobj(pos_eig):
                    x_vec = vector3(*pos_eig[:, 0])
                    y_vec = vector3(*pos_eig[:, 1])
                    #z_vec = vector3(*pos_eig[:, 2])
                    x_vec *= 1.0 / norm(x_vec)
                    y_vec *= 1.0 / norm(y_vec)
                    #z_vec *= 1.0 / norm(z_vec)
                    M = x_vec.row_join(y_vec).row_join(cross(x_vec, y_vec)).row_join(point3(*gc.pose.position))

                    if not gc.occluded:
                        color = hsva_to_rgba((1.0 - gc.weight) * 0.65, 1, 1, 0.7)
                    else:
                        color = (0.3, 0.3, 0.3, 0.5)

                    self.visualizer.draw_shape('cov', M, w.astype(float), 2, *color)

                msg.object_pose_gmm[x].id = gc.id
                msg.object_pose_gmm[x].cov_pose.pose.position  = expr_to_rosmsg(gc.pose.position)
                msg.object_pose_gmm[x].cov_pose.pose.orientation.x = gc.pose.quaternion[0]
                msg.object_pose_gmm[x].cov_pose.pose.orientation.y = gc.pose.quaternion[1]
                msg.object_pose_gmm[x].cov_pose.pose.orientation.z = gc.pose.quaternion[2]
                msg.object_pose_gmm[x].cov_pose.pose.orientation.w = gc.pose.quaternion[3]
                msg.object_pose_gmm[x].cov_pose.covariance = list(gc_cov)
                msg.object_pose_gmm[x].weight = gc.weight
                msg.object_pose_gmm[x].occluded = gc.occluded
            msg.header.stamp = rospy.Time.now()
            msg_total.search_object_list.append(msg)
        if len(msg_total.search_object_list) > 0:
            msg_total.weights.extend([1.0 / len(msg_total.search_object_list)] * len(msg_total.search_object_list))
        self.publisher.publish(msg_total)
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
        return {'topic_prefix': self.topic_prefix}

    def reset(self, simulator):
        """Implements reset behavior.

        :type simulator: BasicSimulator
        :type deltaT: float
        """
        self.message_templates = {}

    @classmethod
    def factory(cls, simulator, init_dict):
        return cls(init_dict['topic_prefix'])

class FakeGMMObjectPublisher(SimulatorPlugin):
    def __init__(self, topic_prefix=''):
        super(FakeGMMObjectPublisher, self).__init__('FakeGMMObjectPublisher')
        self.publisher = rospy.Publisher('{}/perceived_prob_objects'.format(topic_prefix), SearchObjectListMsg, queue_size=1, tcp_nodelay=True)
        self.message_templates = {}
        self.topic_prefix = topic_prefix
        self._enabled = True

    def post_physics_update(self, simulator, deltaT):
        if not self._enabled:
            return

        msg_total = SearchObjectListMsg()
        for name, gmm in simulator.gpcs.items():
            if not name in self.message_templates:
                body = simulator.bodies[name]
                msg = SearchObjectMsg()
                msg.id = body.bId()
                msg.name = name.split('.')[0]
                msg.object_pose_gmm.append(OPGCMsg())
                msg.object_pose_gmm[0].cov_pose.covariance = [0.0]*36
                msg.object_pose_gmm[0].weight = 1.0
                self.message_templates[name] = msg
            else:
                msg = self.message_templates[name]

            gc = gmm[0]
            msg.object_pose_gmm[0].cov_pose.pose.position  = expr_to_rosmsg(gc.pose.position)
            msg.object_pose_gmm[0].cov_pose.pose.orientation.x = gc.pose.quaternion[0]
            msg.object_pose_gmm[0].cov_pose.pose.orientation.y = gc.pose.quaternion[1]
            msg.object_pose_gmm[0].cov_pose.pose.orientation.z = gc.pose.quaternion[2]
            msg.object_pose_gmm[0].cov_pose.pose.orientation.w = gc.pose.quaternion[3]
            msg.header.stamp = rospy.Time.now()
            msg_total.search_object_list.append(msg)
        if len(msg_total.search_object_list) > 0:
            msg_total.weights.extend([1.0 / len(msg_total.search_object_list)] * len(msg_total.search_object_list))
        self.publisher.publish(msg_total)

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

    def reset(self, simulator):
        """Implements reset behavior.

        :type simulator: BasicSimulator
        :type deltaT: float
        """
        self.message_templates = {}

    @classmethod
    def factory(cls, simulator, init_dict):
        return cls(init_dict['topic_prefix'])



class InstantiateSearchObjects(SimulatorPlugin):
    def __init__(self, simulator, topic=''):
        super(InstantiateSearchObjects, self).__init__('InstantiateSearchObjects')
        self.simulator  = simulator
        self.topic      = topic
        self._enabled   = True
        self.subscriber = rospy.Subscriber(self.topic, SearchObjectListMsg, self.add_objects, queue_size=1)

    def add_objects(self, msg):
        for obj_msg in msg.search_object_list:
            dl_object = ros_msg_to_expr(obj_msg)
            sorted_gmm = list(reversed(sorted(dl_object.gmm)))
            pose = sorted_gmm[0].pose
            if DLCube.is_a(dl_object):
                bullet_obj = self.simulator.create_box([dl_object.length, dl_object.width, dl_object.height],
                                       vec3_to_list(pos_of(pose)),
                                       list(quaternion_from_matrix(pose)),
                                       dl_object.mass, name_override=Id)
            elif DLCylinder.is_a(dl_object):
                bullet_obj = self.simulator.create_cylinder(dl_object.radius, dl_object.height,
                                       vec3_to_list(pos_of(pose)),
                                       list(quaternion_from_matrix(pose)),
                                       dl_object.mass, name_override=Id)
            elif DLSphere.is_a(dl_object):
                bullet_obj = self.simulator.create_sphere(dl_object.radius,
                                       vec3_to_list(pos_of(pose)),
                                       list(quaternion_from_matrix(pose)),
                                       dl_object.mass, name_override=Id)
            else:
                Exception('Cannot generate Bullet-body for object which is not a sphere, box, or cylinder.\nObject: {}'.format(str(dl_object)))  

            self.simulator.gpcs[dl_object.id] = sorted_gmm
            self.simulator.initial_weights[dl_object.id] = [gc.weight for gc in sorted_gmm]
        self.disable(self.simulator)


    def disable(self, simulator):
        """Stops the execution of this plugin.

        :type simulator: BasicSimulator
        """
        self._enabled = False
        self.subscriber.unregister()

    def to_dict(self, simulator):
        """Serializes this plugin to a dictionary.

        :type simulator: BasicSimulator
        :rtype: dict
        """
        return {'topic': self.topic}

    @classmethod
    def factory(cls, simulator, init_dict):
        return cls(simulator, init_dict['topic'])    


def create_search_object_message(body, name):
    msg = SearchObjectMsg()
    msg.id = body.bId()
    msg.name = name.split('.')[0]
    opgc = OPGCMsg()
    opgc.weight = 1.0
    msg.object_pose_gmm.append(opgc)
    return msg


