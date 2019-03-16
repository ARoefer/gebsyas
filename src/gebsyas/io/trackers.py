import rospy

from gebsyas.core.dl_reasoning import DLAtom

class Tracker(object):
	"""Trackers are tasked with keeping track of one specific piece of data.
	Whenever a new data point is sensed, it is the tracker's job to process
	that information and decide what to do with it.
	"""
	def __init__(self, data_id, data_state):
		self.data_id    = data_id
		self.data_state = data_state

	def process_data(self, data_set):
		raise (NotImplementedError)

	def disable(self):
		pass


class DLTracker(DLAtom):
	def __init__(self):
		super(DLTracker, self).__init__('Tracker')

	def is_a(self, obj):
		return isinstance(obj, Tracker)

DLTracker = DLTracker()

TBOX_LIST = [DLTracker]
TBOX = TBOX_LIST
