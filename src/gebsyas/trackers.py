import rospy

from gebsyas.utils import StampedData, JointState
from gebsyas.dl_reasoning import DLIded, DLAtom, SymbolicData, DLCompoundObject, DLPhysicalThing, DLRigidObject
from copy import copy

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

	def process_data(self, data_set):
		if type(data_set) == StampedData and DLRigidObject.is_a(data_set.data) and DLIded.is_a(data_set.data):
			self.data_state.insert_data(data_set, data_set.data.id)


class JointStateTracker(Tracker):
	def __init__(self, data_id, data_state):
		super(JointStateTracker, self).__init__(data_id, data_state)

	def process_data(self, data_set):
		if type(data_set) == StampedData and type(data_set.data) == dict:
			old_js = self.data_state[self.data_id]
			if 'r_gripper_finger_joint' in data_set.data and 'gripper_joint' not in data_set.data:
				finger_state = data_set.data['r_gripper_finger_joint']
				data_set.data['gripper_joint'] = JointState(finger_state.position * 2, finger_state.velocity * 2, finger_state.effort)
				#print('added gripper joint to joint state')

			if old_js.data != None:
				for name, state in data_set.data.items():
					if not name in old_js.data or old_js.stamp < data_set.stamp:
						old_js.data[name] = state

				self.data_state.insert_data(StampedData(data_set.stamp, old_js.data), self.data_id)
			else:
				self.data_state.insert_data(data_set, self.data_id)


class SymbolicObjectPoseTracker(Tracker):
	def __init__(self, data_id, pred_state, anchor_id):
		super(SymbolicObjectPoseTracker, self).__init__(data_id, pred_state.data_state)
		self.pred_state = pred_state
		self.data_id = data_id
		self.anchor_id = anchor_id
		obj_data  = self.pred_state.map_to_numeric(data_id).data
		anchor = self.pred_state.map_to_data(anchor_id).data
		if type(anchor) != SymbolicData:
			raise Exception('Id "{}" does not refer to a symbolic data structure.'.format(anchor_id))
		sym_anchor = anchor.data
		if not DLPhysicalThing.is_a(sym_anchor):
			raise Exception('Id "{}" does not refer to a physical thing.'.format(anchor_id))
		num_anchor = self.pred_state.map_to_numeric(anchor_id).data
		w2a = num_anchor.pose.inv()
		self.o2a = w2a * obj_data.pose
		obj_data.pose = sym_anchor.pose * self.o2a
		self.data_state.insert_data(StampedData(rospy.Time.now(), SymbolicData(data=obj_data, f_convert=self.convert_to_numeric, args=[anchor_id])), data_id)

	def convert_to_numeric(self, anchor):
		sym_obj = copy(self.data_state[self.data_id].data.data)
		sym_obj.pose = anchor.pose * self.o2a
		return sym_obj

	def disable(self):
		resolved_object = self.convert_to_numeric(self.pred_state.map_to_numeric(self.anchor_id))
		self.data_state.insert_data(StampedData(rospy.Time.now(), resolved_object), self.data_id)

	def process_data(self, data_set):
		pass

class DLTracker(DLAtom):
	def __init__(self):
		super(DLTracker, self).__init__('Tracker')

	def is_a(self, obj):
		return isinstance(obj, Tracker)

TBOX_LIST = [DLTracker()]
TBOX = TBOX_LIST
