import rospy
from collections import namedtuple
from gebsyas.constants import LBA_BOUND, UBA_BOUND
from gebsyas.dl_reasoning import SymbolicData, DLSphere, DLCube, DLCylinder, DLCompoundObject, DLRigidObject, DLShape
from gebsyas.utils import StampedData, ros_msg_to_expr
from gebsyas.predicates import Predicate
from giskardpy.symengine_wrappers import pos_of
from yaml import load, dump


def log(msg, prefix=''):
	if prefix == '':
		print(msg)
	else:
		print('{}: {}'.format(prefix, msg))


def pfd_str(predicate_dict, con='\n'):
	flat_predicates = []
	for p, args in predicate_dict.items():
		for a, v in args.items():
			flat_predicates.append('{}({}) : {}'.format(p.P, ', '.join(a), v))
	return con.join(flat_predicates)


def visualize_obj(obj, display, pose, ns='objects', color=(0,0,1,1)):
	if DLShape.is_a(obj):
		if DLCube.is_a(obj):
			display.draw_cube(ns, pose, (obj.length, obj.width, obj.height), color[0], color[1], color[2], color[3])
		elif DLCylinder.is_a(obj):
			display.draw_cylinder(ns, pose, obj.height, 2*obj.radius, color[0], color[1], color[2], color[3])
		elif DLSphere.is_a(obj):
			display.draw_sphere(ns, pos_of(pose), 2*obj.radius, color[0], color[1], color[2], color[3])

	if DLCompoundObject.is_a(obj):
		if type(obj.subObject) == list:
			for x in obj.subObject:
				visualize_obj(x, display, pose * x.pose, ns, color)
		else:
			visualize_obj(obj.subObject, display, pose * obj.subObject.pose, ns, color)


class PredicateState(object):
	def evaluate(self, context, predicate, args_tuple):
		raise (NotImplementedError)

	def map_to_numeric(self, symbol):
		raise (NotImplementedError)

	def map_to_data(self, symbol):
		raise (NotImplementedError)

	# Computes a new state S = Self / Other
	# S should be a {P: {args: Bool*}*}
	def difference(self, context, other):
		raise (NotImplementedError)

	# Erase specific predicate statements from this State
	# facts should be {P: {args: Bool*}*}
	def erase(self, facts):
		raise (NotImplementedError)

	# Creates a new predicate state uniting self and other
	# Self takes precident over the other state
	def union(self, other):
		raise (NotImplementedError)

	def __str__(self):
		raise (NotImplementedError)


class AssertionDrivenPredicateState(PredicateState):
	def __init__(self, fact_state = {}, parent_state=None):
		self.predicates = fact_state
		self.parent = None
		if parent_state != None:
			if isinstance(parent_state, AssertionDrivenPredicateState):
				self.union(parent_state)
			else:
				self.parent = parent_state


	def evaluate(self, context, predicate, args_tuple):
		if p in self.predicates:
			if args_tuple in self.predicates[p]:
				return self.predicates[p][args_tuple]

		if self.parent != None:
			return self.parent.evaluate(context, predicate, args_tuple)

		if not p in self.predicates:
			self.predicates[p] = {}

		self.predicates[p][args_tuple] = False
		return False

	def difference(self, context, other):
		if self.parent != None:
			out = self.parent.difference(context, other)
		else:
			out = {}

		for p, value_map in self.predicates.items():
			for args, value in value_map.items():
				if value != other.evaluate(context, p, args):
					if not p in out:
						out[p] = {}
					out[p][args] = value
		return out

	# Erase specific predicate statements from this State
	# facts should be {P: {args}*} or {P: {args: _*}*}
	# DOES NOT COPY!
	def erase(self, facts):
		for P, insts in facts.items():
			for args in insts:
				if P in self.predicates and args in self.predicates[P]:
					self.predicates[P].pop(args, None)

		return self

	# Unifies this state with another one. DOES NOT COPY!
	def union(self, other):
		if not isinstance(other, AssertionDrivenPredicateState):
			raise Exception('AssertionDrivenPredicateState.union is only implemented for assertive states')

		for P, insts in other.predicates.items():
			for args, val in insts.items():
				if P not in self.predicates:
					self.predicates[P] = {}

				if args not in self.predicates[P]:
					self.predicates[P][args] = val

		return self

	def get_facts_of_type(self, dl_type, relating_to=None):
		out = {}
		for P, insts in self.predicates.items():
			if dl_type.is_a(P):
				out[P] = {}
				for args, truth in insts.items():
					if relating_to == None or len(relating_to.intersection(args)) != 0:
						out[P][args] = truth
		return out

	def __str__(self):
		return 'State:\n   {}'.format(pfd_str(self.predicates, '\n   '))



class DataDrivenPredicateState(PredicateState):
	def __init__(self, reasoner, predicates, data_state):
		super(DataDrivenPredicateState, self).__init__()

		self.reasoner   = reasoner
		self.predicates = predicates #{pn: Predicate(p.P, p.fp, [self.reasoner[dlc] for dlc in p.dl_args if dlc in self.resoner else dlc]) for p in predicates}
		self.predicate_cache = {}
		self.data_state = data_state if data_state != None else DataSceneState()

	def __eval_ineq_constraints(self, context, predicate, fargs, args_tuple):
		if not min([predicate.dl_args[x].is_a(fargs[x]) for x in range(len(predicate.dl_args))]):
			raise Exception('Can not evaluate {}({}), because the type signature does not match ({}). Given types: \n  {}'.format(predicate.P, ', '.join(args_tuple), ', '.join([str(dla) for dla in predicate.dl_args]),
			'\n  '.join('{}: {}'.format(args_tuple[x], ', '.join([str(dla) for dla in self.reasoner.classify(fargs[x])])) for x in range(len(args_tuple)))))

		ineq_constraints = predicate.fp(context, *fargs)
		if len(ineq_constraints) == 0:
			context.log('Predicate is not evaluable for the given symbols. Defaulting to False.', 'Predicate State')
			return False

		for scname, sc in ineq_constraints.items():
			try:
				if not sc.lower <= LBA_BOUND or not sc.upper >= UBA_BOUND:
					context.log('New value is False. This is caused by {}: lbA = {}, ubA = {}'.format(scname, sc.lower, sc.upper), 'Predicate State')

					return False
				else:
					context.log('{}: lbA = {}, ubA = {}'.format(scname, sc.lower, sc.upper))
			except Exception as e:
				raise Exception('Error occured while evaluating constraint "{}". Error: \n{}'.format(scname, e))
		context.log('New value is True.', 'Predicate State')
		return True


	def evaluate(self, context, predicate, args_tuple):
		new_val = False
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
					context.log('Re-eval triggered for {}{}'.format(predicate.P, args_tuple), 'Predicate State')
					new_val = self.__eval_ineq_constraints(context, predicate, fargs, args_tuple)
				else:
					context.log('Returning cached value for {}{}'.format(predicate.P, args_tuple), 'Predicate State')
					return cached.data
		else:
			context.log('Predicate {} has never been evaluated before'.format(predicate.P), 'Predicate State')
			self.predicate_cache[predicate] = {}
			new_val = self.__eval_ineq_constraints(context, predicate, [self.map_to_numeric(s).data for s in args_tuple], args_tuple)

		self.predicate_cache[predicate][args_tuple] = StampedData(rospy.Time.now(), new_val)
		return new_val


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

	def visualize(self, display, ns='predicate_state'):
		for sd in self.data_state.id_map.keys():
			obj = self.map_to_numeric(sd).data
			if DLRigidObject.is_a(obj):
				if type(self.map_to_data(sd).data) == SymbolicData:
					visualize_obj(obj, display, obj.pose, ns, (1,0,0,1))
				else:
					visualize_obj(obj, display, obj.pose, ns)

	def difference(self, context, other):
		out = {}
		cache_copy = {}
		for p, args in self.predicate_cache.items():
			cache_copy[p] = args.copy()

		for p, values in cache_copy.items():
			for args in values.keys():
				my_value = self.evaluate(context, p, args)
				if my_value != other.evaluate(context, p, args):
					if not p in out:
						out[p] = {}
					out[p][args] = my_value
		return out


class DataSceneState(object):
	def __init__(self, searchable_objects=[], parent=None):
		super(DataSceneState, self).__init__()

		self.id_map = {}
		self.searchable_objects = searchable_objects
		self.parent = parent

	def __getitem__(self, key):
		return self.find_by_path(key)

	def dump_to_file(self, filepath):
		stream = file(filepath, 'w')
		dump(self.id_map, stream)
		stream.close()

	def find_by_path(self, path, transparent_symbolics=False):
		path_stack = path.split('/')
		if len(path_stack) > 0:
			if path_stack[0] in self.id_map:
				return self.__find_by_path(self.id_map[path_stack[0]].data, path_stack[1:], self.id_map[path_stack[0]].stamp, transparent_symbolics)
			else:
				for root in self.searchable_objects:
					out = self.__find_by_path(root, path_stack, rospy.Time.now(), transparent_symbolics)
					if out.data != None:
						return out

		if self.parent != None:
			return self.parent[path]

		return StampedData(rospy.Time.now(), None)


	def __find_by_path(self, obj, path_stack, current_stamp, transparent_symbolics):
		if (transparent_symbolics or len(path_stack) > 1) and type(obj) == SymbolicData:
			obj = obj.data

		if len(path_stack) == 0:
			return StampedData(current_stamp, obj)
		else:
			try:
				subdata = getattr(obj, path_stack[0])
				if type(subdata) == StampedData:
					return self.__find_by_path(subdata.data, path_stack[1:], subdata.stamp, transparent_symbolics)
				else:
					return self.__find_by_path(subdata, path_stack[1:], current_stamp, transparent_symbolics)
			except AttributeError as e:
				if type(obj) == dict and path_stack[0] in obj:
					subdata = obj[path_stack[0]]
					if type(subdata) == StampedData:
						return self.__find_by_path(subdata.data, path_stack[1:], subdata.stamp, transparent_symbolics)
					else:
						return self.__find_by_path(subdata, path_stack[1:], current_stamp, transparent_symbolics)
				else:
					raise e

	def dl_iterator(self, dl_concept, transparent_symbolics=True):
		return DLSymbolIterator(self, dl_concept, transparent_symbolics)

	def dl_data_iterator(self, dl_concept, transparent_symbolics=True):
		return DLIterator(self, dl_concept, transparent_symbolics)

	def insert_data(self, stamped_data, Id):
		if '/' in Id:
			path = Id[:Id.find('/')]
			Id  = Id[len(path)+1:]
			stamped_obj = self.find_by_path(path)
			if stamped_obj.stamp > stamped_data.stamp:
				return
			obj = stamped_obj.data
			if type(obj) == dict:
				obj[Id] = stamped_data.data
			else:
				setattr(obj, Id, stamped_data.data)
			if path[0] in self.id_map:
				root_obj = self.find_by_path(path[0]).data
				self.id_map[path[0]] = StampedData(stamped_data.stamp, root_obj)
		else:
			if Id not in self.id_map or self.id_map[Id].stamp < stamped_data.stamp:
				self.id_map[Id] = stamped_data

	def get_data_map(self):
		out = self.id_map.copy()
		for root in self.searchable_objects:
			fields = [x for x in dir(root) if x[0] != '_' and not callable(getattr(root, x)) and not x in out]
			for f in fields:
				out[f] = StampedData(rospy.Time.now(), getattr(root, f))
		return out


class DataIterator(object):
	def __init__(self, data_state):
		self.data_state = data_state
		self.__state_iter = None
		self.__state = 0

	def __iter__(self):
		return self

	def next(self): # Python 3: def __next__(self)
		if self.__state_iter == None:
			self.__state_iter = self.data_state.id_map.iteritems()

		try:
			return self.__state_iter.next()
		except StopIteration as e:
			self.__state += 1

		if self.__state == 1:
			if self.data_state.parent != None:
				self.data_state = self.data_state.parent
				self.__state_iter = None
				self.__state = 0
			else:
				raise StopIteration


class DLIterator(DataIterator):
	def __init__(self, data_state, dl_concept, transparent_symbolics=True):
		super(DLIterator, self).__init__(data_state)
		self.dl_concept = dl_concept
		self.t_symbolics = transparent_symbolics

	def next(self): # Python 3: def __next__(self)
		while True:
			Id, nextObj = super(DLIterator, self).next()
			data = nextObj.data
			if self.t_symbolics and type(data) == SymbolicData:
				data = data.data

			if self.dl_concept.is_a(data):
				return Id, nextObj

class StamplessIterator(object):
	def __init__(self, iterator):
		self.iterator = iterator

	def __iter__(self):
		return self

	def next(self):
		Id, data = self.iterator.next()
		if type(data) == StampedData:
			return Id, data.data
		return Id, data


class DLSymbolIterator(DLIterator):
	def next(self): # Python 3: def __next__(self)
		Id, nextObj = super(DLSymbolIterator, self).next()
		return Id
