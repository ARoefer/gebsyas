from gebsyas.dl_reasoning import DLAtom
from gebsyas.utils import ros_msg_to_expr
from gebsyas.data_structures import StampedData
from multiprocessing import Lock

class Sensor(object):
	"""Sensors are the Input-abstraction in gebsyas. 
	They are supposed to hide technical communication details 
	associated with input methods.
	""" 
	def __init__(self, name, callback):
		self.name = name
		self.callback = callback

	def enable(self):
		raise (NotImplementedError)

	def disable(self):
		raise (NotImplementedError)

	def __str__(self):
		return self.name


class DLSensor(DLAtom):
	def __init__(self):
		super(DLSensor, self).__init__('Sensor')

	def is_a(self, obj):
		return isinstance(obj, Sensor)

DLSensor = DLSensor()

TBOX_LIST = [DLSensor]
TBOX = TBOX_LIST