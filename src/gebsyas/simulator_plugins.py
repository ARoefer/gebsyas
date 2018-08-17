import rospy
from math import atan2

from gebsyas.msg import Pose2DStamped as Pose2DStampedMsg

from iai_bullet_sim.basic_simulator import SimulatorPlugin
from iai_bullet_sim.rigid_body import RigidBody

from gop_gebsyas_msgs.msg import ProbObject as POMsg
from gop_gebsyas_msgs.msg import ProbObjectList as POLMsg
from gebsyas.utils import expr_to_rosmsg

from iai_bullet_sim.multibody import SimpleBaseDriver


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
        msg.pose.theta = z = atan2(2 * pose.quaternion[0] * pose.quaternion[1] + 2 * pose.quaternion[3] * pose.quaternion[2], w2 + x2 - y2 - z2)
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
