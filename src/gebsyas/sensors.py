import rospy

from gebsyas.dl_reasoning import DLAtom
from gebsyas.utils import ros_msg_to_expr
from gebsyas.data_structures import StampedData


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
	def __init__(self, callback, topic, topic_type, queue_size=1):
		super(TopicSensor, self).__init__('topic sensor for {} on {}'.format(str(topic_type), topic), callback)
		self.topic = topic
		self.topic_type = topic_type
		self.queue_size = queue_size

	def topic_cb(self, msg):
		self.callback(StampedData(msg.header.stamp, ros_msg_to_expr(msg)))

	def enable(self):
		self.subscriber = rospy.Subscriber(self.topic, self.topic_type, self.topic_cb, queue_size=self.queue_size)

	def disable(self):
		if self.subscriber != None:
			self.subscriber.unregister()


class DLSensor(DLAtom):
	def __init__(self):
		super(DLSensor, self).__init__('Sensor')

	def is_a(self, obj):
		return isinstance(obj, Sensor)

TBOX_LIST = [DLSensor()]
TBOX = TBOX_LIST