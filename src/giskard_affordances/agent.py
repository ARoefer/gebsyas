import rospy
import sensor_msgs
import std_msgs
from control_msgs.msg import GripperCommandActionGoal
from giskard_affordances.actions import Context, Logger, ActionManager
from giskard_affordances.dl_reasoning import DLAtom, DLRigidObject, DLIded, SymbolicData
from giskard_affordances.msg import ProbabilisticObject as PObject
from giskard_affordances.numeric_scene_state import DataSceneState, DataDrivenPredicateState
from giskard_affordances.ros_visualizer import ROSVisualizer
from giskard_affordances.sensors import TopicSensor
from giskard_affordances.simple_agent_actions import SimpleAgentIOAction
from giskard_affordances.trackers import VisualObjectTracker, JointStateTracker
from giskard_affordances.utils import StampedData, ros_msg_to_expr, cmdDictToJointState, Blank
from giskardpy.symengine_wrappers import *
from copy import copy
import yaml

class Agent(object):
	"""docstring for Agent"""
	def __init__(self, name, reasoner, predicates, capabilities=[], logger=None, visualizer=None, data_state=None):
		super(Agent, self).__init__()
		self.name        = name
		self.reasoner    = reasoner
		self.logger      = logger     if     logger != None else Logger()
		self.visualizer  = visualizer if visualizer != None else ROSVisualizer('{}_vis'.format(name))
		self.data_state  = data_state if data_state != None else DataSceneState()
		self.predicate_state = DataDrivenPredicateState(self.reasoner, predicates, self.data_state)
		self.sensors     = {}
		self.trackers    = {}
		self.action_manager = ActionManager(capabilities)

	def awake(self):
		for s in self.sensors.values():
			s.enable()

	def sleep(self):
		for s in self.sensors.values():
			s.disable()

	def add_sensor(self, name, sensor):
		self.sensors[name] = sensor
		self.reasoner.add_to_abox((str(sensor), self.reasoner.get_expanded_concept('Sensor')))

	def add_tracker(self, tracker):
		self.remove_tracker(tracker.data_id)
		self.trackers[tracker.data_id] = tracker
		self.reasoner.add_to_abox((str(tracker), self.reasoner.get_expanded_concept('Tracker')))

	def get_tracker(self, tracker_id):
		if tracker_id in self.trackers:
			return self.trackers[tracker_id]
		else:
			return None

	def remove_tracker(self, data_id):
		if data_id in self.trackers:
			self.reasoner.remove_from_abox(str(self.trackers[data_id]))
			self.trackers[data_id].disable()
			del self.trackers[data_id]

	def get_actions(self):
		return self.action_manager


class SimpleAgent(Agent):
	"""docstring for SimpleAgent"""
	def __init__(self, name, reasoner, predicates, robot, capabilities=[], memory_path=None, logger=None, visualizer=None):
		super(SimpleAgent, self).__init__(name, reasoner, predicates, capabilities, logger, visualizer, DataSceneState())
		self.add_sensor('object sensor', TopicSensor(self.on_object_sensed, '/perceived_objects', PObject, 12))
		self.add_sensor('joint sensor', TopicSensor(self.on_joint_state_sensed, '/joint_states', sensor_msgs.msg.JointState))
		self.add_tracker(JointStateTracker('joint_state', self.data_state))
		self.reasoner.add_to_abox((self.name, DLAgent()))
		self.robot = robot
		self.frame_of_reference = self.robot.frames['base_link'][:4, :3].row_join(pos_of(self.robot.camera.pose))
		self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=self.robot.gripper, f_convert=self.robot.do_gripper_fk, args=['joint_state'])), 'gripper')
		self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=self.robot.camera, f_convert=self.robot.do_camera_fk, args=['joint_state'])), 'camera')
		self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=self, f_convert=self.convert_to_numeric, args=['joint_state'])), 'me')
		self.data_state.insert_data(StampedData(rospy.Time.now(), Agent('You', reasoner, predicates)), 'you')
		self.js_callbacks = []
		self.command_publisher = rospy.Publisher('/simulator/commands', sensor_msgs.msg.JointState, queue_size=5) #  /velocity_controller/joint_velocity_controller/joint_velocity_commands
		self.gripper_command_publisher = rospy.Publisher('/gripper_controller/gripper_action/goal', GripperCommandActionGoal, queue_size=5)
		self.smwyg_pub = rospy.Publisher('/show_me_what_you_got', std_msgs.msg.Empty, queue_size=1)
		self.memory_path = memory_path if memory_path != None else '{}_memory.yaml'.format(name)

		try:
			fmem = open(self.memory_path, 'r')
			self.memory = yaml.load(fmem)
			fmem.close()
		except IOError:
			self.memory = {}
			pass


	def on_object_sensed(self, stamped_object):
		if stamped_object.data.id not in self.trackers:
			self.trackers[stamped_object.data.id] = VisualObjectTracker(stamped_object.data.id, self.data_state)

		self.trackers[stamped_object.data.id].process_data(stamped_object)

	def on_joint_state_sensed(self, joint_state):
		self.trackers['joint_state'].process_data(joint_state)
		new_js = self.data_state['joint_state'].data
		for cb in self.js_callbacks:
			cb(new_js)

	def awake(self):
		super(SimpleAgent, self).awake()
		rospy.sleep(0.3)
		self.smwyg_pub.publish(std_msgs.msg.Empty())
		action = SimpleAgentIOAction(self)
		action.execute(Context(self, self.logger, self.visualizer))
		f = open(self.memory_path, 'w+')
		f.write(yaml.dump(self.memory))
		f.close()
		#self.data_state.dump_to_file('data_state_{}.yaml'.format(self.name))

	def get_tbox(self):
		return self.reasoner

	def get_abox(self):
		return self.reasoner.abox

	def get_data_state(self):
		return self.data_state

	def get_predicate_state(self):
		return self.predicate_state

	def add_js_callback(self, callback):
		self.js_callbacks.append(callback)

	def act(self, js_command):
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


	def convert_to_numeric(self, joint_state):
		a = copy(self)
		js = {name: state.position for name, state in joint_state.items()}
		a.frame_of_reference = a.frame_of_reference.subs(js)
		return a


class DLAgent(DLAtom):
	def __init__(self):
		super(DLAgent, self).__init__('Agent')

	def is_a(self, obj):
		return isinstance(obj, Agent)

TBOX_LIST = [DLAgent()]
TBOX = TBOX_LIST
