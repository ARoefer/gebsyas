import rospy
import sensor_msgs
from giskard_affordances.actions import Context, Logger
from giskard_affordances.dl_reasoning import DLAtom, DLRigidObject, DLIded, SymbolicData
from giskard_affordances.msg import ProbabilisticObject as PObject
from giskard_affordances.numeric_scene_state import DataSceneState, PredicateSceneState
from giskard_affordances.ros_visualizer import ROSVisualizer
from giskard_affordances.simple_agent_actions import SimpleAgentIOAction
from giskard_affordances.utils import StampedData, ros_msg_to_expr, cmdDictToJointState

class Agent(object):
	"""docstring for Agent"""
	def __init__(self, name, reasoner, predicates, logger=None, visualizer=None, data_state=None):
		super(Agent, self).__init__()
		self.name        = name
		self.reasoner    = reasoner
		self.logger      = logger     if     logger != None else Logger()
		self.visualizer  = visualizer if visualizer != None else ROSVisualizer('{}_vis'.format(name))
		self.data_state  = data_state if data_state != None else DataSceneState()
		self.predicate_state = PredicateSceneState(self.reasoner, self.data_state, predicates)
		self.sensors     = {}
		self.trackers    = {}

	def awake(self):
		for s in self.sensors.values():
			s.enable()

	def sleep(self):
		for s in self.sensors.values():
			s.disable()

	def add_sensor(self, name, sensor):
		self.sensors[name] = sensor
		self.reasoner.add_to_abox((str(sensor), self.reasoner.get_concept('Sensor')))

	def add_tracker(self, tracker):
		self.remove_tracker(tracker.data_id)
		self.trackers[tracker.data_id] = tracker
		self.reasoner.add_to_abox((str(tracker), self.reasoner.get_concept('Tracker')))

	def remove_tracker(self, data_id):
		if data_id in self.trackers:
			self.reasoner.remove_from_abox(str(self.trackers[data_id]))
			self.trackers[data_id].disable()
			del self.trackers[data_id]


class SimpleAgent(Agent):
	"""docstring for SimpleAgent"""
	def __init__(self, name, reasoner, predicates, robot, logger=None, visualizer=None):
		super(SimpleAgent, self).__init__(name, reasoner, predicates, logger, visualizer, DataSceneState())

		self.add_sensor('object sensor', TopicSensor(self.on_object_sensed, '/perceived_objects', PObject))
		self.add_sensor('joint sensor', TopicSensor(self.on_joint_state_sensed, '/joint_states', sensor_msgs.msg.JointState))
		self.add_tracker(JointStateTracker('joint_state', self.data_state))
		self.reasoner.add_to_abox((self.name, DLAgent()))
		self.robot = robot
		self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=self.robot.gripper, f_convert=self.robot.do_gripper_fk, args=['joint_state'])), 'gripper')
		self.js_callbacks = []
		self.command_publisher = rospy.Publisher('/simulator/commands', sensor_msgs.msg.JointState, queue_size=5)

	def on_object_sensed(self, stamped_object):
		if stamped_object.data.id not in self.trackers:
			self.trackers[stamped_object.data.id] = VisualObjectTracker(stamped_object.data.id, self.data_state)

		self.trackers[stamped_object.data.id].proccess_data(stamped_object)

	def on_joint_state_sensed(self, joint_state):
		self.trackers['joint_state'].proccess_data(joint_state)
		new_js = self.data_state['joint_state'].data
		for cb in self.js_callbacks:
			cb(new_js)

	def awake(self):
		super(SimpleAgent, self).awake()
		rospy.sleep(0.3)
		action = SimpleAgentIOAction(self)
		action.execute(Context(self, self.logger, self.visualizer))
		#self.data_state.dump_to_file('data_state_{}.yaml'.format(self.name))

	def get_tbox(self):
		return self.reasoner.tbox

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

class Sensor(object):
	def __init__(self, name, callback):
		self.name = name
		self.callback = callback

	def enable(self):
		raise (NotImplementedError)

	def disable(self):
		raise (NotImplementedError)

	def __str__(self):
		return self.name


class TopicSensor(Sensor):
	def __init__(self, callback, topic, topic_type):
		super(TopicSensor, self).__init__('topic sensor for {} on {}'.format(str(topic_type), topic), callback)
		self.topic = topic
		self.topic_type = topic_type

	def topic_cb(self, msg):
		self.callback(StampedData(msg.header.stamp, ros_msg_to_expr(msg)))

	def enable(self):
		self.subscriber = rospy.Subscriber(self.topic, self.topic_type, self.topic_cb)

	def disable(self):
		if self.subscriber != None:
			self.subscriber.unregister()


class Tracker(object):
	def __init__(self, data_id, data_state):
		self.data_id    = data_id
		self.data_state = data_state

	def process_data(self, data_set):
		raise (NotImplementedError)

	def disable(self):
		pass


class VisualObjectTracker(Tracker):
	def __init__(self, data_id, data_state):
		super(VisualObjectTracker, self).__init__(data_id, data_state)

	def proccess_data(self, data_set):
		if type(data_set) == StampedData and DLRigidObject.is_a(data_set.data) and DLIded.is_a(data_set.data):
			self.data_state.insert_data(data_set, data_set.data.id)


class JointStateTracker(Tracker):
	def __init__(self, data_id, data_state):
		super(JointStateTracker, self).__init__(data_id, data_state)

	def proccess_data(self, data_set):
		if type(data_set) == StampedData and type(data_set.data) == dict:
			old_js = self.data_state[self.data_id]
			if old_js.data != None:
				for name, state in data_set.data.items():
					if not name in old_js.data or old_js.stamp < data_set.stamp:
						old_js.data[name] = state

				self.data_state.insert_data(StampedData(data_set.stamp, old_js.data), self.data_id)
			else:
				self.data_state.insert_data(data_set, self.data_id)


class SymbolicObjectPoseTracker(Tracker):
	def __init__(self, data_id, data_state, sym_pose, subs_map):
		super(SymbolicObjectTracker, self).__init__(data_id, data_state)
		self.subs_map = subs_map
		self.obj = self.data_state[data_id].data

		symbolic_object = StampedData(rospy.Time.now(), self.__convert_to_symbolic(sym_pose, self.obj))
		self.data_state.insert_data(symbolic_object, self.data_id)

	def __convert_to_symbolic(self, pframe, obj):
		if DLCompoundObject.is_a(obj):
			if type(obj.subObject) == list:
				obj.subObject = [self.__convert_to_symbolic(pframe * obj.pose, x) for x in obj.subObject]
			else:
				obj.subObject = self.__convert_to_symbolic(pframe * obj.pose, obj.subObject)

		obj.pose = SymbolicData(pframe, self.convert_frame, [pframe])
		return SymbolicData(obj, self.convert_object, [obj])

	def __convert_to_numeric(self, pframe, obj):
		obj.pose = self.convert_frame(pframe)
		if DLCompoundObject.is_a(obj):
			if type(obj.subObject) == list:
				obj.subObject = [self.__convert_to_numeric(obj.pose * x.pose.data, x) for x in obj.subObject]
			else:
				obj.subObject = self.__convert_to_numeric(obj.pose * x.pose.data, obj.subObject)

		return obj

	def convert_frame(self, frame):
		subs_dict = dict([(sym, self.data_state[path].data) for sym, path in self.subs_map.items()])
		return frame.subs(subs_dict)

	def convert_object(self, obj):
		self.obj = self.__convert_to_numeric(self.convert_frame(self.obj.pose.data), self.obj)
		return self.obj

	def disable(self):
		resolved_object = self.convert_object(self.obj)
		self.data_state.insert_data(resolved_object, self.data_id)


class DLSensor(DLAtom):
	def __init__(self):
		super(DLSensor, self).__init__('Sensor')

	def is_a(self, obj):
		return isinstance(obj, Sensor)

class DLTracker(DLAtom):
	def __init__(self):
		super(DLTracker, self).__init__('Tracker')

	def is_a(self, obj):
		return isinstance(obj, Tracker)

class DLAgent(DLAtom):
	def __init__(self):
		super(DLAgent, self).__init__('Agent')

	def is_a(self, obj):
		return isinstance(obj, Agent)

AGENT_TBOX_LIST = [DLSensor(), DLTracker(), DLAgent()]
AGENT_TBOX = dict([(str(c), c) for c in AGENT_TBOX_LIST])
