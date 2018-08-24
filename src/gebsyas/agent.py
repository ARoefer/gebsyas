import rospy
import sensor_msgs
import std_msgs
import yaml


from control_msgs.msg import GripperCommandActionGoal
from geometry_msgs.msg import Twist as TwistMsg

from gebsyas.msg import Pose2DStamped as Pose2DStampedMsg

from gebsyas.actions import Context, Logger, ActionManager
from gebsyas.dl_reasoning import DLAtom, DLRigidObject, DLIded
from gebsyas.data_structures import SymbolicData, StampedData
from gop_gebsyas_msgs.msg import ProbObject as PObject
from gop_gebsyas_msgs.msg import ProbObjectList as PObjectList
from gop_gebsyas_msgs.msg import SearchObject   as SearchObjectMsg
from gebsyas.numeric_scene_state import DataSceneState, DataDrivenPredicateState
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.sensors import TopicSensor
from gebsyas.simple_agent_actions import SimpleAgentIOAction
from gebsyas.trackers import GaussianObjectTracker, JointStateTracker, LocalizationTracker
from gebsyas.utils import StampedData, ros_msg_to_expr, cmdDictToJointState, Blank
from giskardpy.symengine_wrappers import *
from copy import copy

class Agent(object):
	"""
	@brief      Superclass for agents.
	"""
	def __init__(self, name, reasoner, predicates, capabilities=[], logger=None, visualizer=None, data_state=None):
		"""Constructor. Receives a name for the agent, a reasoner, a set of known predicates.
		   Additionally, a list of PActionWrappers, a custom logger, a custom visualizer and a custom data state can be supplied.
		"""
		super(Agent, self).__init__()
		self.name        = name
		self.reasoner    = reasoner
		self.logger      = logger     if     logger != None else Logger()
		self.visualizer  = visualizer if visualizer != None else ROSVisualizer('{}_vis'.format(name), 'map')
		self.data_state  = data_state if data_state != None else DataSceneState()
		self.predicate_state = DataDrivenPredicateState(self.reasoner, predicates, self.data_state)
		self.sensors     = {}
		self.trackers    = {}
		self.action_manager = ActionManager(capabilities)

	def awake(self):
		"""Activates the agent's sensors."""
		for s in self.sensors.values():
			s.enable()

	def sleep(self):
		"""Disables the agent's sensors."""
		for s in self.sensors.values():
			s.disable()

	def add_sensor(self, name, sensor):
		"""Adds a sensor to the agent and its reasoner."""
		self.sensors[name] = sensor
		self.reasoner.add_to_abox((str(sensor), self.reasoner.get_expanded_concept('Sensor')))

	def add_tracker(self, tracker):
		"""Adds a tracker to the agent and its reasoner. If a tracker for the same data already exists, it is replaced-"""
		self.remove_tracker(tracker.data_id)
		self.trackers[tracker.data_id] = tracker
		self.reasoner.add_to_abox((str(tracker), self.reasoner.get_expanded_concept('Tracker')))

	def get_tracker(self, data_id):
		"""Returns a tracker by its tracked data's Id."""
		if data_id in self.trackers:
			return self.trackers[data_id]
		else:
			return None

	def remove_tracker(self, data_id):
		"""Removes a tracker by its tracked data's Id."""
		if data_id in self.trackers:
			self.reasoner.remove_from_abox(str(self.trackers[data_id]))
			self.trackers[data_id].disable()
			del self.trackers[data_id]

	def get_actions(self):
		"""Returns the agent's action manager."""
		return self.action_manager


class BasicAgent(Agent):
	"""
	This agent implementation runs a simple IO behavior through wich a user can issue simple commands to the agent.
	The agent controls a robot, perceives objects from a topic and can memorize data across sessions.
	"""
	def __init__(self, name, reasoner, predicates, robot, capabilities=[], memory_path=None, logger=None, visualizer=None):
		super(BasicAgent, self).__init__(name, reasoner, predicates, capabilities, logger, visualizer, DataSceneState())
		self.add_sensor('object sensor', TopicSensor(self.on_objects_sensed, '/perceived_prob_objects', SearchObjectMsg, 12))
		self.add_sensor('joint sensor', TopicSensor(self.on_joint_state_sensed, '/{}/joint_states'.format(robot._urdf_robot.name), sensor_msgs.msg.JointState))
		self.add_tracker(JointStateTracker('joint_state', self.data_state))
		self.add_tracker(LocalizationTracker('localization', self.data_state))
		self.add_sensor('localization', TopicSensor(self.trackers['localization'].process_data, '/{}/localization'.format(robot._urdf_robot.name), Pose2DStampedMsg, 1))
		self.reasoner.add_to_abox((self.name, DLAgent()))
		self.robot = robot
		self.frame_of_reference = self.robot.get_fk_expression('map', 'base_link')[:4, :3].row_join(pos_of(self.robot.camera.pose))
		#self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=self.robot.gripper, f_convert=self.robot.do_gripper_fk, args=['joint_state'])), 'gripper')
		self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=self.robot.camera, f_convert=self.robot.do_camera_fk, args=['joint_state'])), 'camera')
		self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=self, f_convert=self.convert_to_numeric, args=['joint_state'])), 'me')
		self.data_state.insert_data(StampedData(rospy.Time.now(), Agent('You', reasoner, predicates)), 'you')
		self.command_publisher = rospy.Publisher('/{}/commands/joint_velocities'.format(robot._urdf_robot.name), sensor_msgs.msg.JointState, queue_size=5) #  /velocity_controller/joint_velocity_controller/joint_velocity_commands
		self.gripper_command_publisher = rospy.Publisher('/gripper_controller/gripper_action/goal', GripperCommandActionGoal, queue_size=5)
		self.base_command_publisher = rospy.Publisher('/base_command', TwistMsg, queue_size=1)
		self.smwyg_pub = rospy.Publisher('/show_me_what_you_got', std_msgs.msg.Empty, queue_size=1)
		self.memory_path = memory_path if memory_path != None else '{}_memory.yaml'.format(name)

		try:
			fmem = open(self.memory_path, 'r')
			self.memory = yaml.load(fmem)
			fmem.close()
		except IOError:
			self.memory = {}
			pass

	def add_data_cb(self, Id, cb):
		self.data_state.register_on_change_cb(Id, cb)

	def remove_data_cb(self, Id, cb):
		self.data_state.deregister_on_change_cb(Id, cb)

	def on_objects_sensed(self, stamped_object):
		"""Callback for a sensed object."""
		if stamped_object.data.id not in self.trackers:
			self.trackers[stamped_object.data.id] = GaussianObjectTracker(stamped_object.data.id, self.data_state)

		self.trackers[stamped_object.data.id].process_data(stamped_object)

	def on_joint_state_sensed(self, joint_state):
		"""Callback for a sensed joint state"""
		self.trackers['joint_state'].process_data(joint_state)

	def awake(self, initial_action):
		"""Activates the sensors and starts the IO behavior."""
		super(BasicAgent, self).awake()
		rospy.sleep(0.3)
		self.smwyg_pub.publish(std_msgs.msg.Empty())
		initial_action.execute(Context(self, self.logger, self.visualizer))
		f = open(self.memory_path, 'w+')
		f.write(yaml.dump(self.memory))
		f.close()
		#self.data_state.dump_to_file('data_state_{}.yaml'.format(self.name))

	def get_tbox(self):
		"""Returns the agent's reasoner."""
		return self.reasoner

	def get_abox(self):
		"""Returns the reasoner's a box."""
		return self.reasoner.abox

	def get_data_state(self):
		"""Returns the agent's data state."""
		return self.data_state

	def get_predicate_state(self):
		"""Returns the agent's predicate state."""
		return self.predicate_state

	def add_js_callback(self, callback):
		"""Adds an additional joint state callback."""
		self.data_state.register_on_change_cb('joint_state', callback)

	def act(self, js_command):
		"""Interfaces between internal commands and ROS-commands. Accepts joint velocity commands."""
		cmd = cmdDictToJointState(js_command)
		self.command_publisher.publish(cmd)
		if 'gripper_joint' in js_command:
			gripper_command = GripperCommandActionGoal()
			vel_cmd = js_command['gripper_joint']
			gripper_command.goal.command.max_effort = 60.0
			if vel_cmd > 0.005:
				gripper_command.goal.command.position = 0.1
			elif vel_cmd < -0.005:
				gripper_command.goal.command.position = 0.0
			else:
				gripper_command.goal.command.position = self.data_state['joint_state/gripper_joint/position'].data
			self.gripper_command_publisher.publish(gripper_command)

		base_msg = TwistMsg()
		if 'base_linear_joint' in js_command:
			base_msg.linear.x = js_command['base_linear_joint']
		if 'base_angular_joint' in js_command:
			base_msg.angular.z = js_command['base_angular_joint']
		self.base_command_publisher.publish(base_msg)

	def convert_to_numeric(self, joint_state):
		"""Creates a fully numeric version of this agent."""
		a = copy(self)
		js = {name: state.position for name, state in joint_state.items()}
		a.frame_of_reference = a.frame_of_reference.subs(js)
		return a


class DLAgent(DLAtom):
	"""Description logical concept representing an agent."""
	def __init__(self):
		super(DLAgent, self).__init__('Agent')

	def is_a(self, obj):
		return isinstance(obj, Agent)

# TBox for this file
TBOX_LIST = [DLAgent()]
TBOX = TBOX_LIST
