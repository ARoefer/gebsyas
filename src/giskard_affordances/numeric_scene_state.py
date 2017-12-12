import rospy
from collections import namedtuple
from giskard_affordances.dl_reasoning import SymbolicData, DLSphere, DLCube, DLCylinder, DLCompoundObject, DLRigidObject, DLShape
from giskard_affordances.utils import StampedData, ros_msg_to_expr
from yaml import load, dump

Predicate = namedtuple('Predicate', ['P', 'fp', 'dp'])

def log(msg, prefix=''):
	if prefix == '':
		print(msg)
	else:
		print('{}: {}'.format(prefix, msg))

class PredicateSceneState(object):

	def __init__(self, tbox, data_state, predicates):
		super(PredicateSceneState, self).__init__()

		self.tbox = tbox
		self.predicate_cache = {}
		self.predicates = predicates
		self.data_state = data_state if data_state != None else DataSceneState()


	def evaluate(self, predicate, args_tuple, f_logging=log):
		if predicate in self.predicate_cache:
			p_cache = self.predicate_cache[predicate]
			if args_tuple in p_cache:
				cached = p_cache[args_tuple]
				fargs = []
				reeval = False
				for a in args_tuple:
					numeric = self.map_to_numeric(a)
					if numeric.stamp > cached.stamp:
						reeval = True
					fargs.append(numeric.data)

				if reeval:
					f_logging('Re-eval triggered for {}{}'.format(predicate.P, args_tuple), 'Predicate State')
					new_val = predicate.fp(*fargs)
					f_logging('New value is {}. True-threshold is {}'.format(new_val, predicate.dp), 'Predicate State')
					self.predicate_cache[predicate][args_tuple] = StampedData(rospy.Time.now(), new_val >= predicate.dp)
					return new_val >= predicate.dp
				else:
					f_logging('Returning cached value for {}{}'.format(predicate.P, args_tuple), 'Predicate State')
					return cached.data
		else:
			f_logging('Predicate {} has never been evaluated before'.format(predicate.P), 'Predicate State')
			self.predicate_cache[predicate] = {}

		new_val = predicate.fp(*[self.map_to_numeric(s).data for s in args_tuple])
		f_logging('New value is {}. True-threshold is {}'.format(new_val, predicate.dp), 'Predicate State')
		self.predicate_cache[predicate][args_tuple] = StampedData(rospy.Time.now(), new_val >= predicate.dp)
		return new_val >= predicate.dp

	def map_to_numeric(self, symbol):
		symbolic = self.map_to_data(symbol)
		if type(symbolic.data) == SymbolicData:
			n_args = [self.map_to_numeric(a).data for a in symbolic.data.args]
			return StampedData(rospy.Time.now(), symbolic.data.f_convert(*n_args))
		else:
			return symbolic

	def map_to_data(self, symbol):
		if type(symbol) is bool:
			return StampedData(rospy.Time(0), int(symbol))
		elif type(symbol) is str:
			return self.data_state[symbol]

	def assert_fact(self, predicate, args_tuple, value):
		if not predicate in self.predicate_cache:
			self.predicate_cache[predicate] = {}

		self.predicate_cache[predicate][args_tuple] = StampedData(rospy.Time.now(), value)


def visualize_obj(obj, display, pose, ns='objects'):
	if DLShape.is_a(obj):
		if DLCube.is_a(obj):
			display.draw_cube(ns, pose, (obj.length, obj.width, obj.height))
		elif DLCylinder.is_a(obj):
			display.draw_cylinder(ns, pose, obj.height, obj.radius)
		elif DLSphere.is_a(obj):
			display.draw_sphere(ns, pos_of(pose), obj.radius)

	if DLCompoundObject.is_a(obj):
		if type(obj.subObject) == list:
			for x in obj.subObject:
				visualize_obj(x, display, pose * x.pose)
		else:
			visualize_obj(obj.subObject, display, pose * obj.subObject.pose)


class DataSceneState(object):
	def __init__(self, searchable_objects=[]):
		super(DataSceneState, self).__init__()

		self.id_map = {}
		self.searchable_objects = searchable_objects

	def __getitem__(self, key):
		return self.find_by_path(key)

	def dump_to_file(self, filepath):
		stream = file(filepath, 'w')
		dump(self.id_map, stream)
		stream.close()

	def find_by_path(self, path):
		path_stack = path.split('/')
		if len(path_stack) > 0:
			if path_stack[0] in self.id_map:
				return self.__find_by_path(self.id_map[path_stack[0]].data, path_stack[1:], self.id_map[path_stack[0]].stamp)
			else:
				for root in self.searchable_objects:
					out = self.__find_by_path(root, path_stack, rospy.Time.now())
					if out.data != None:
						return out

		return StampedData(rospy.Time.now(), None)


	def __find_by_path(self, obj, path_stack, current_stamp):
		if len(path_stack) == 0:
			return StampedData(current_stamp, obj)
		else:
			subdata = getattr(obj, path_stack[0])
			if type(subdata) == StampedData:
				return self.__find_by_path(subdata.data, path_stack[1:], subdata.stamp)
			else:
				return self.__find_by_path(subdata, path_stack[1:], current_stamp)


	def insert_data(self, stamped_data, Id):
		if Id not in self.id_map or self.id_map[Id].stamp < stamped_data.stamp:
			self.id_map[Id] = stamped_data

	def get_data_map(self):
		out = self.id_map.copy()
		for root in self.searchable_objects:
			fields = [x for x in dir(root) if x[0] != '_' and not callable(getattr(root, x)) and not x in out]
			for f in fields:
				out[f] = StampedData(rospy.Time.now(), getattr(root, f))
		return out

	def visualize(self, display, ns='data_state'):
		for sd in self.id_map.values():
			obj = sd.data
			if DLRigidObject.is_a(obj):
				visualize_obj(obj, display, obj.pose, ns)
