import rospy

from gebsyas.io.sensors import Sensor
from gebsyas.ros.utils  import ros_msg_to_ks

class TopicSensor(Sensor):
    def __init__(self, callback, topic, topic_type, queue_size=1):
        super(TopicSensor, self).__init__('topic sensor for {} on {}'.format(str(topic_type), topic), callback)
        self.topic = topic
        self.topic_type = topic_type
        self.queue_size = queue_size

    def topic_cb(self, msg):
        self.callback(msg)

    def enable(self):
        self.subscriber = rospy.Subscriber(self.topic, self.topic_type, self.topic_cb, queue_size=self.queue_size)

    def disable(self):
        if self.subscriber != None:
            self.subscriber.unregister()