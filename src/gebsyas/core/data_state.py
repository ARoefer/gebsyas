from collections import namedtuple
from yaml import load, dump

from multiprocessing import RLock


class DataState(object):
	def __init__(self, parent=None):
		super(DataSceneState, self).__init__()

		self.parent = parent
		self.data_tree = {}
		self.value_table = {}
		self.data_change_callbacks = {}
		# Mapping of {DLConcept: set}
		self.new_data_callbacks = {}
		self.lock = RLock()

	def __getitem__(self, key):
		return self.find_data(key)

	def dump_to_file(self, filepath):
		stream = file(filepath, 'w')
		dump(self.id_map, stream)
		stream.close()

	def find_data(self, Id):
		path = Id
		if type(Id) is str:
			path = Id.split('/')
		
		with self.lock:
			container = self.data_tree
			try:
				for part in path:
					if type(container) is dict:
						container = container[part]
					elif type(container) is list:
						container = container[int(part)]
					else:
						container = getattr(container, part)
				return container
			except (KeyError, IndexError, AttributeError):
				if self.parent is not None:
					return self.parent.find_data(path)
				else:
					raise Exception('Unknown data id "{}"'.format(Id))

	def insert_data(self, data, Id):
		path = Id
		if type(Id) is str:
			path = Id.split('/')
		
		with self.lock:	
			container = self.data_tree
			try:
				for part in path[:-1]:
					if type(container) is dict:
						container = container[part]
					elif type(container) is list:
						container = container[int(part)]
					else:
						container = getattr(container, part)
			except (KeyError, IndexError, AttributeError):
				raise Exception('Can not insert data at "{}", "{}" does not exist.'.format(path, part))

			if type(container) is dict:
				is_new = path[-1] in container
				container[path[-1]] = data
			elif type(container) is list:
				idx = int(path[-1])
				is_new = idx == len(container)
				if is_new:
					container.append(data)
				else:
					container[idx] = data
			else:
				is_new = hasattr(container, path[-1])
				setattr(container, path[-1], data)

			if is_new:
				for dl_type, cbs in self.new_data_callbacks.items():
					if dl_type.is_a(data):
						for cb in cbs:
							cb(data)
			
			if Id in self.data_change_callbacks:
				for cb in self.data_change_callbacks[Id]:
					cb(data)

	def get_data_map(self):
		return self.data_tree.copy()

	def event_data_refreshed(self, Id):
		"""Temporary, dirty way of triggering the update callbacks for value updates."""
		if Id in self.data_change_callbacks:
			for cb in self.data_change_callbacks[Id]:
				cb(data)		

	def register_on_change_cb(self, Id, cb):
		if Id not in self.data_change_callbacks:
			self.data_change_callbacks[Id] = set()

		self.data_change_callbacks[Id].add(cb)

	def deregister_on_change_cb(self, Id, cb):
		if Id in self.data_change_callbacks:
			self.data_change_callbacks[Id].remove(cb)

	def register_new_data_cb(self, dl_type, cb):
		if dl_type not in self.new_data_callbacks:
			self.new_data_callbacks[dl_type] = set()
		self.new_data_callbacks[dl_type].add(cb)

	def deregister_new_data_cb(self, dl_type, cb):	
		if dl_type in self.new_data_callbacks:
			self.new_data_callbacks[dl_type].remove(cb)

	def dl_iterator(self, dl_concept):
		return DLSymbolIterator(self, dl_concept)

	def dl_data_iterator(self, dl_concept):
		return DLIterator(self, dl_concept)


class DataIterator(object):
	def __init__(self, data_state):
		self.data_state = data_state
		self.__state_iter = None
		self.__state = 0
		self.lock = data_state.lock

	def __iter__(self):
		return self

	def next(self): # Python 3: def __next__(self)
		if self.__state_iter == None:
			self.__state_iter = iter(self.data_state.data_tree.items())

		try:
			with self.lock:
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
	def __init__(self, data_state, dl_concept):
		super(DLIterator, self).__init__(data_state)
		self.dl_concept = dl_concept

	def next(self): # Python 3: def __next__(self)
		while True:
			Id, nextObj = super(DLIterator, self).next()
			data = nextObj.data

			if self.dl_concept.is_a(data):
				return Id, nextObj


class StamplessIterator(object):
	def __init__(self, iterator):
		self.iterator = iterator

	def __iter__(self):
		return self

	def next(self):
		return self.iterator.next()


class DLSymbolIterator(DLIterator):
	def next(self): # Python 3: def __next__(self)
		Id, nextObj = super(DLSymbolIterator, self).next()
		return Id
