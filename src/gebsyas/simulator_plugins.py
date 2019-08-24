import rospy
import math
import numpy as np
import pybullet as pb

from iai_bullet_sim.basic_simulator import SimulatorPlugin, invert_transform, hsva_to_rgba, transform_point
from iai_bullet_sim.rigid_body import RigidBody
from iai_bullet_sim.multibody import SimpleBaseDriver
from iai_bullet_sim.utils import Frame, AABB

from gebsyas.msg import LocalizedPoseStamped as LPSMsg
from gebsyas.utils import expr_to_rosmsg
from gebsyas.simulator import frame_tuple_to_sym_frame
from gebsyas.ros_visualizer import ROSVisualizer
from giskardpy.symengine_wrappers import *
from gop_gebsyas_msgs.msg import ProbObject as POMsg
from gop_gebsyas_msgs.msg import ProbObjectList as POLMsg
from gop_gebsyas_msgs.msg import SearchObject as SearchObjectMsg
from gop_gebsyas_msgs.msg import SearchObjectList as SearchObjectListMsg
from gop_gebsyas_msgs.msg import ObjectPoseGaussianComponent as OPGCMsg

from sensor_msgs.msg import JointState as JointStateMsg
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

    @profile
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

    @profile
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

    @profile
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


class RobotMirror(SimulatorPlugin):
    """Plugin which publishes an object's current pose as nav_msgs/Odometry message."""
    def __init__(self, multibody, state_topic='/joint_states', localization_topic='/localization'):
        """Initializes the plugin.

        :param simulator: Simulator
        :type  simulator: BasicSimulator
        :param multibody: Object to observe.
        :type  multibody: iai_bullet_sim.multibody.MultiBody
        :param child_frame_id: Name of the frame being published.
        :type  child_frame_id: str
        """
        super(RobotMirror, self).__init__('Robot Mirror')
        self.body = multibody
        multibody.register_deletion_cb(self.on_obj_deleted)
        self.state_topic = state_topic
        self.localization_topic = localization_topic
        self.sub_js = rospy.Subscriber(state_topic, JointStateMsg, callback=self.on_new_js, queue_size=1)
        self.sub_loc = rospy.Subscriber(localization_topic, LPSMsg, callback=self.on_new_loc, queue_size=1)

        self.js = None
        self.location = None

        self.__enabled = True

    def on_new_js(self, msg):
        self.js = {msg.name[x]: msg.position[x] for x in range(len(msg.name)) if msg.name[x] in self.body.joints}

    def on_new_loc(self, msg):
        roll_half  = msg.pose.angular.x / 2.0
        pitch_half = msg.pose.angular.y / 2.0
        yaw_half   = msg.pose.angular.z / 2.0

        c_roll  = cos(roll_half)
        s_roll  = sin(roll_half)
        c_pitch = cos(pitch_half)
        s_pitch = sin(pitch_half)
        c_yaw   = cos(yaw_half)
        s_yaw   = sin(yaw_half)

        cc = c_roll * c_yaw
        cs = c_roll * s_yaw
        sc = s_roll * c_yaw
        ss = s_roll * s_yaw

        x = c_pitch * sc - s_pitch * cs
        y = c_pitch * ss + s_pitch * cc
        z = c_pitch * cs - s_pitch * sc
        w = c_pitch * cc + s_pitch * ss

        self.location = Frame([msg.pose.linear.x, msg.pose.linear.y, msg.pose.linear.z], [x, y, z, w])

    def on_obj_deleted(self, simulator, Id, obj):
        self.disable(simulator)
        simulator.deregister_plugin(self)

    def pre_physics_update(self, simulator, deltaT):
        """Updates the body's state according to the latest update.

        :type simulator: iai_bullet_sim.basic_simulator.BasicSimulator
        :type deltaT: float
        """
        if self.__enabled is False:
            return

        if self.location is not None:
            self.body.set_pose(self.location)
            self.location = None

        if self.js is not None:
            self.body.set_joint_positions(self.js)
            self.js = None

    def disable(self, simulator):
        """Disables the publisher.
        :type simulator: iai_bullet_sim.basic_simulator.BasicSimulator
        """
        self.__enabled = False
        self.sub_loc.unregister()
        self.sub_js.unregister()

    def to_dict(self, simulator):
        """Serializes this plugin to a dictionary.
        :type simulator: iai_bullet_sim.basic_simulator.BasicSimulator
        :rtype: dict
        """
        return {'body': simulator.get_body_id(self.body.bId()),
                'state_topic': self.state_topic,
                'localization_topic': self.localization_topic}

    @classmethod
    def factory(cls, simulator, init_dict):
        """Instantiates the plugin from a dictionary in the context of a simulator.

        :type simulator: iai_bullet_sim.basic_simulator.BasicSimulator
        :type init_dict: dict
        :rtype: OdometryPublisher
        """
        body = simulator.get_body(init_dict['body'])
        if body is None:
            raise Exception('Body "{}" does not exist in the context of the given simulation.'.format(init_dict['body']))
        return cls(body, init_dict['state_topic'], init_dict['localization_topic'])


from sensor_msgs.msg import Image as ImageMsg
import cv2
import cv_bridge

class RGBDCamera(SimulatorPlugin):
    def __init__(self, multibody, camera_link, offset, topic_prefix, type, width, height, fov, near, far):
        self.multibody    = multibody
        self.camera_link  = camera_link
        self.topic_prefix = topic_prefix
        self.offset       = offset
        self.width        = width
        self.height       = height
        self.fov          = fov
        self.near         = near
        self.far          = far
        self.type         = type
        self.projection_matrix = pb.computeProjectionMatrixFOV(fov * (180 / np.pi), width / float(height), near, far)
        self._bridge   = cv_bridge.CvBridge()
        self.__enabled = False

        self._frame_id = None

        self.pub_image = rospy.Publisher('{}/rgb'.format(self.topic_prefix), ImageMsg, queue_size=1, tcp_nodelay=True)
        self.pub_depth = rospy.Publisher('{}/depth'.format(self.topic_prefix), ImageMsg, queue_size=1, tcp_nodelay=True)
        self.pub_mask  = rospy.Publisher('{}/mask'.format(self.topic_prefix), ImageMsg, queue_size=1, tcp_nodelay=True)
        self.__enabled = True

    def disable(self, simulator):
        """Disables the publisher.
        :type simulator: iai_bullet_sim.basic_simulator.BasicSimulator
        """
        self.__enabled = False
        self.pub_image.unregister()
        self.pub_depth.unregister()
        self.pub_mask.unregister()

    @profile
    def post_physics_update(self, simulator, deltaT):
        if not self.__enabled:
            return

        if self._frame_id is None:
            self._frame_id = '{}/{}'.format(simulator.get_body_id(self.multibody.bId()), self.camera_link)

        camera_transform = self.multibody.get_link_state(self.camera_link).worldFrame
        view_matrix = pb.computeViewMatrix(transform_point(camera_transform, self.offset), transform_point(camera_transform, [5, 0, 0]), (0,0,1))

        if simulator.mode == 'gui':
            _, _, rgb, depth, segmenatation = pb.getCameraImage(self.width, self.height, view_matrix, self.projection_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        else:
            _, _, rgb, depth, segmenatation = pb.getCameraImage(self.width, self.height, view_matrix, self.projection_matrix)#, shadow=0, renderer=pb.ER_TINY_RENDERER)



        rgb_message = self._bridge.cv2_to_imgmsg(rgb.reshape((self.height, self.width, 4))[:,:,:3].astype(np.uint8), 'rgb8')
        depth_message = self._bridge.cv2_to_imgmsg(depth.reshape((self.height, self.width)).astype(np.float32))
        #mask_message  = self._bridge.cv2_to_imgmsg(segmenatation.reshape((self.height, self.width)))
        rgb_message.header.frame_id = self._frame_id
        rgb_message.header.stamp    = rospy.Time.now()
        depth_message.header = rgb_message.header
        #mask_message.header  = rgb_message.header
        self.pub_image.publish(rgb_message)
        self.pub_depth.publish(depth_message)
        #self.pub_mask.publish(mask_message)


    def to_dict(self, simulator):
        return {'body': simulator.get_body_id(self.multibody.bId()),
                'camera_link' : self.camera_link,
                'offset'      : self.offset,
                'topic_prefix': self.topic_prefix,
                'type'        : self.type,
                'width'       : self.width,
                'height'      : self.height,
                'fov'         : self.fov,
                'near'        : self.near,
                'far'         : self.far}

    @classmethod
    def factory(cls, simulator, init_dict):
        body = simulator.get_body(init_dict['body'])
        if body is None:
            raise Exception('Body "{}" does not exist in the context of the given simulation.'.format(init_dict['body']))
        return cls(body, init_dict['camera_link'], 
                         init_dict['offset'], 
                         init_dict['topic_prefix'],
                         init_dict['type'],
                         init_dict['width'],
                         init_dict['height'],
                         init_dict['fov'],
                         init_dict['near'],
                         init_dict['far'])


def create_search_object_message(body, name):
    msg = SearchObjectMsg()
    msg.id = body.bId()
    msg.name = name.split('.')[0]
    opgc = OPGCMsg()
    opgc.weight = 1.0
    msg.object_pose_gmm.append(opgc)
    return msg


from iai_bullet_sim.basic_simulator import vec3_to_list
from iai_bullet_sim.ros_plugins import rotation3_quaternion
from faster_rcnn_object_detector.msg import BBoxInfo as BBoxInfoMsg

def aabb_to_matrix(aabb):
    pass


class ObjectDetector(SimulatorPlugin):
    def __init__(self, multibody, camera_link, topic, width, height, fov, near, far):
        self.multibody   = multibody
        self.camera_link = camera_link
        self.topic       = topic
        self.width       = width
        self.height      = height
        self.fov         = fov
        self.fov_threshold = fov * 0.5
        self.near        = near
        self.far         = far
        self.__enabled = True

        self.__numpy_mode = hasattr(pb, 'rayTestBatch_numpy')
        self._frame_id = None


        r = np.tan(0.5 * fov) * near
        t = r * (width / float(height))
        # self._projection_matrix = np.array([[near / r,        0,                            0,                                0],
        #                                     [       0, near / t,                            0,                                0],
        #                                     [       0,        0, (-far - near) / (far - near), (-2 * far * near) / (far - near)],
        #                                     [       0,        0,                           -1,                                0]])
        # # Project along x-axis to be aligned with the "x forwards" convention
        # self._projection_matrix = np.array([[0, 1, 0, 0], 
        #                                     [0, 0, 1, 0], 
        #                                     [1, 0, 0, 0], 
        #                                     [0, 0, 0, 1]]).dot(self._projection_matrix)
        # Primitive projection. Projects y into x, z into y- points in view should range from -0.5 to 0.5
        self._projection_matrix = np.array([[0,        -r, 0, 0],
                                            [0,         0, t, 0],
                                            [1 / near,  0, 0, 0]])
        self._screen_translation = np.array([[0.5 * width,             0, 0.5 * width],
                                             [          0, -0.5 * height, 0.5 * height]])
        self._frustum_vertices = np.array([[near, -r,  t, 1],
                                           [near, -r, -t, 1],
                                           [near,  r, -t, 1],
                                           [near,  r,  t, 1],
                                           [ far, -r,  t, 1],
                                           [ far, -r, -t, 1],
                                           [ far,  r, -t, 1],
                                           [ far,  r,  t, 1]]).T
        self._aabb_vertex_template = np.array([[1, 1, 1, 1],
                                               [1, 1, 0, 1],
                                               [1, 0, 1, 1],
                                               [1, 0, 0, 1],
                                               [0, 1, 1, 1],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 0, 1]])
        self.pub_bbox = rospy.Publisher(self.topic, BBoxInfoMsg, queue_size=20, tcp_nodelay=True)


    def disable(self, simulator):
        """Disables the publisher.
        :type simulator: iai_bullet_sim.basic_simulator.BasicSimulator
        """
        self.__enabled = False
        self.pub_bbox.unregister()


    @profile
    def post_physics_update(self, simulator, deltaT):
        if not self.__enabled:
            return

        if self._frame_id is None:
            self._frame_id = '{}/{}'.format(simulator.get_body_id(self.multibody.bId()), self.camera_link)

        camera_transform = self.multibody.get_link_state(self.camera_link).worldFrame

        transform = np.hstack((np.zeros((4,3)), np.array([[camera_transform.position.x],[camera_transform.position.y],[camera_transform.position.z],[0]])))
        transform = rotation3_quaternion(camera_transform.quaternion.x,
                                         camera_transform.quaternion.y,
                                         camera_transform.quaternion.z,
                                         camera_transform.quaternion.w) + transform
        transformed_frustum_points = transform.dot(self._frustum_vertices)
        max_coords = transformed_frustum_points.max(1)
        min_coords = transformed_frustum_points.min(1)

        aabbs       = [(body, body.get_AABB()) for body, _ in simulator.get_overlapping(AABB(min_coords, max_coords), {self.multibody})]
        aabb_matrix = np.array([[vec3_to_list(aabb.min) + [1], vec3_to_list(aabb.max) + [1]] for _, aabb in aabbs]) # n_aabbs x 4 x 2
        positions   = (aabb_matrix.sum(1) * 0.5).T # 4 x n_aabbs
        dimensions  = (aabb_matrix[:,1,:] - aabb_matrix[:,0,:])
        looped_position = np.repeat(transform[:, 3:], len(aabbs), 1)
        r_positions = positions - looped_position # 4 x n_aabbs

        dirs = r_positions / np.sqrt((r_positions * r_positions).sum(0))
        ray_starts  = (looped_position + dirs * self.near) # Start positions at "near" distance from camera. Not quite accurate...
        print('positions: {}\ndimensions: {}\nlooped_position: {}\nr_positions: {}\nray_starts: {}'.format(positions.shape, dimensions.shape, looped_position.shape, r_positions.shape, ray_starts.shape))



        if self.__numpy_mode:
            #print('Passing numpy array to pybullet...')
            indices, hits = pb.rayTestBatch_numpy(ray_starts.T, positions.T, physicsClientId=simulator.client_id())
        else:
            raise Exception('Object Detector is only built for use with a numpy-compiled bullet engine.')
        
        idx_matrix = np.array([body.bId() for body, _ in aabbs]).reshape((len(aabbs), 1))
        positive_hits = np.equal(indices[:,:1], idx_matrix)

        print('hits:\n{}\nconcat matrix:\n{}\npositive hits: {}'.format(hits, np.hstack((indices[:,:1], idx_matrix)), positive_hits))

        inv_transform = np.eye(4)
        inv_transform[:3, :3] = transform[:3, :3].T
        inv_transform[:3,  3] = -inv_transform[:3, :3].dot(transform[:3, 3])

        projection_matrix = self._projection_matrix.dot(inv_transform)
        for x in range(len(aabbs)):
            if positive_hits[x, 0]:
                msg = BBoxInfoMsg()

                vertices = (self._aabb_vertex_template * dimensions[x]).T
                print('vertices: {}\nprojection_matrix: {}'.format(vertices.shape, projection_matrix.shape))
                projected_vertices = projection_matrix.dot(vertices)
                projected_vertices = self._screen_translation.dot(np.divide(projected_vertices, projected_vertices[2,:]).clip(-0.5, 0.5))
                print(projected_vertices)
                msg.bbox_xmin, msg.bbox_ymin = projected_vertices.min(axis=1).flatten()
                msg.bbox_xmax, msg.bbox_ymax = projected_vertices.max(axis=1)
                msg.max_label = simulator.get_body_id(idx_matrix[x,0]).split('.')[0]
                msg.max_score = 1.0
                print(msg)
                self.pub_bbox.publish(msg)


    def to_dict(self, simulator):
        return {'body': simulator.get_body_id(self.multibody.bId()),
                'camera_link': self.camera_link,
                'topic'      : self.topic,
                'width'      : self.width,
                'height'     : self.height,
                'fov'        : self.fov,
                'near'       : self.near,
                'far'        : self.far}

    @classmethod
    def factory(cls, simulator, init_dict):
        body = simulator.get_body(init_dict['body'])
        if body is None:
            raise Exception('Body "{}" does not exist in the context of the given simulation.'.format(init_dict['body']))
        return cls(body, init_dict['camera_link'], 
                         init_dict['topic'],
                         init_dict['width'],
                         init_dict['height'],
                         init_dict['fov'],
                         init_dict['near'],
                         init_dict['far'])