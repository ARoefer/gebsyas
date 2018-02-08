from collections import namedtuple
from pprint import pprint

from gebsyas.basic_controllers import InEqController, run_ineq_controller
from gebsyas.dl_reasoning import DLDisjunction, DLConjunction
from gebsyas.numeric_scene_state import AssertionDrivenPredicateState, pfd_str
from gebsyas.predicates import Predicate
from gebsyas.ros_visualizer import ROSVisualizer
import rospy


Context = namedtuple('Context', ['agent', 'log', 'display'])
LogData = namedtuple('LogData', ['stamp', 'name', 'data'])

def pbainst_str(pba):
	return '{}({})'.format(pba.action_wrapper.action_name, ', '.join(['{}={}'.format(a,v) for a, v in pba.assignments.items()]))

class Logger():
	def __init__(self):
		self.log_birth = rospy.Time.now()
		self.indentation = 0

	def log(self, data, name=''):
		title_str = '{:>12.4f}:{} | '.format((rospy.Time.now() - self.log_birth).to_sec(), '  ' * self.indentation)
		pure_data_str = str(data)
		if pure_data_str[-1] == '\n':
			data_str = pure_data_str[:-1].replace('\n', '\n{}| '.format(' ' * (len(title_str) - 2)))
		else:
			data_str = pure_data_str.replace('\n', '\n{}| '.format(' ' * (len(title_str) - 2)))

		if len(name) > 0:
			print('{}{}: {}'.format(title_str, name, data_str))
		else:
			print('{}{}'.format(title_str, data_str))

	def indent(self):
		self.indentation = self.indentation + 1

	def unindent(self):
		if self.indentation > 0:
			self.indentation = self.indentation - 1

	def __call__(self, data, name=''):
		self.log(data, name)


class Action(object):
	def __init__(self, name):
		self.name = name

	def __str__(self):
		return 'action({})'.format(self.name)

	def execute(self, context):
		raise (NotImplementedError)

	def execute_subaction(self, context, action):
		context.log(action.name, 'Starting sub action')
		context.log.indent()
		out = action.execute(context)
		context.log.unindent()
		context.log.log('\b{}\nExecution of sub action {} results in {}'.format('-' * 25, action.name, out))
		return out


class PBasedActionSequence(Action):
	def __init__(self, sequence=[], cost=0.0):
		super(PBasedActionSequence, self).__init__('P-ActionSequence')
		self.sequence = sequence
		self.cost = cost

	def execute(self, context):
		pred_state = context.agent.get_predicate_state()
		for a_inst in self.sequence:
			precon_diff = AssertionDrivenPredicateState(a_inst.precons).difference(context, pred_state)
			if len(precon_diff) == 0:
				if self.execute_subaction(context, a_inst.action_wrapper.instantiate_action(context, a_inst.assignments)) < 0.8:
					return 0.0
			else:
				context.log('Execution of action sequence terminated because the preconditions for {} are not met. Mismatch:\n   {}'.format(
					pbainst_str(a_inst),
					pfd_str(precon_diff, '\n   ')))
				return 0.0
		return 1.0

	def __add__(self, other):
		if type(other) != PBasedActionSequence:
			raise TypeError
		return PBasedActionSequence(self.sequence + other.sequence, self.cost + other.cost)

	def append(self, action):
		return PBasedActionSequence(self.sequence + [action], self.cost + action.action_wrapper.cost)

	def push(self, action):
		return PBasedActionSequence([action] + self.sequence, self.cost + action.action_wrapper.cost)

	def __str__(self):
		return ' -> '.join([pbainst_str(a) for a in self.sequence])


class ActionManager(object):
	def __init__(self, capabilities):
		self.__precon_action_map  = {}
		self.__postcon_action_map = {}
		for c in capabilities:
			for p in c.precons.keys():
				if p not in self.__precon_action_map:
					self.__precon_action_map[p] = set()
				self.__precon_action_map[p].add(c)

			for p in c.postcons.keys():
				if p not in self.__postcon_action_map:
					self.__postcon_action_map[p] = set()
				self.__postcon_action_map[p].add(c)

	def get_postcon_map(self):
		return self.__postcon_action_map

	def get_precon_map(self):
		return self.__precon_action_map


class PActionInterface(object):
	def __init__(self, action_name, precons, postcons, cost=1.0):
		self.action_name = action_name
		self.cost        = cost
		self.signature   = {}
		self.precons     = {}
		self.postcons    = {}
		self._arg_p_map  = {}
		self._arg_pre_p_map  = {}
		self._arg_post_p_map = {}

		for pI in (precons + postcons):
			for x in range(len(pI.predicate.dl_args)):
				a = pI.args[x]
				if not a in self.signature:
					self.signature[a]  = pI.predicate.dl_args[x]
					self._arg_p_map[a] = []
				else:
				 	# implication_intersection = self.signature[a].implication_intersection(pI.predicate.dl_args[x])
				 	# if len(implication_intersection) == 0:
						# raise Exception('Invalid action signature! Symbol {} was first defined as:\n   {}\nbut is redefined as:\n   {}\n in {}({}) = {}'.format(a, str(self.signature[a]), str(pI.predicate.dl_args[x]), pI.predicate.P, ', '.join(pI.args), pI.value))
			 		# elif len(implication_intersection) < len(self.signature[a].implied_by):
			 		if type(self.signature[a]) == DLConjunction:
			 			self.signature[a] = DLConjunction(pI.predicate.dl_args[x], *list(self.signature[a].concepts))
					else:
						self.signature[a] = DLConjunction(self.signature[a], pI.predicate.dl_args[x])
				self._arg_p_map[a].append(pI)

		for pI in precons:
			if not pI.predicate in self.precons:
				self.precons[pI.predicate] = {}
			if pI.args in self.precons[pI.predicate]:
				raise Exception('Redefinition of precondition {}({}) for action {}. '.format(pI.predicate.P, ', '.join(pI.args), action_name))
			self.precons[pI.predicate][pI.args] = pI.value
			for a in pI.args:
				if a not in self._arg_pre_p_map:
					self._arg_pre_p_map[a] = []
				self._arg_pre_p_map[a].append(pI)

		for pI in postcons:
			if not pI.predicate in self.postcons:
				self.postcons[pI.predicate] = {}
			if pI.args in self.postcons[pI.predicate]:
				raise Exception('Redefinition of postcondition {}({}) for action {}. '.format(pI.predicate.P, ', '.join(pI.args), action_name))
			self.postcons[pI.predicate][pI.args] = pI.value
			for a in pI.args:
				if a not in self._arg_post_p_map:
					self._arg_post_p_map[a] = []
				self._arg_post_p_map[a].append(pI)

		param_len = max([len(n) for n in self.signature.keys()])
		self.__str_representation = '{}:\n  Parameters:\n    {}\n  Preconditions:\n    {}\n  Postconditions:\n    {}'.format(self.action_name,
			'\n    '.join(['{:>{:d}}: {}'.format(a, param_len, str(t)) for a, t in self.signature.items()]),
			pfd_str(self.precons, '\n    '),
			pfd_str(self.postcons, '\n    '))


	def parameterize_by_postcon(self, context, postcons):
		return PostconditionAssignmentIterator(context, self, postcons)

	def parameterize_by_precon(self, context, precons):
		return PreconditionAssignmentIterator(context, self, precons)

	def __str__(self):
		return self.__str_representation

	def __hash__(self):
		return hash(self.__str_representation)

	def instantiate_action(self, context, assignments):
		raise (NotImplementedError)



PWrapperInstance = namedtuple('PWrapperInstance', ['assignments', 'precons', 'postcons', 'action_wrapper'])

class PermutationIterator(object):
	def __init__(self, iterator_dict):
		self.iterators = []
		self.names = []
		for name, iterator in iterator_dict.items():
			self.iterators.append(iterator)
			self.names.append(name)

		self.iterator_cache = []
		self.it_index = len(self.iterators) - 1
		self.current_permutation = []
		for x in range(len(self.iterators) - 1):
			self.iterator_cache.append([self.iterators[x].next()])
			self.current_permutation.append(self.iterator_cache[x][0])
		self.iterator_cache.append([])
		self.current_permutation.append(None)
		self.use_cached = len(self.iterators) - 1

	def __iter__(self):
		return self

	def next(self):
		while True:
			try:
				next_elem = self.iterators[self.it_index].next()
				if self.use_cached < self.it_index:
					self.iterator_cache[self.it_index].append(next_elem)
				self.current_permutation[self.it_index] = next_elem
				if self.it_index+1 == len(self.iterators):
					return dict(zip(self.names, self.current_permutation))
				else:
					self.it_index += 1
					self.iterators[self.it_index] = iter(self.iterator_cache[self.it_index])
			except StopIteration:
				if self.it_index == 0:
					raise StopIteration
				else:
					self.use_cached -= 1
					self.it_index   -= 1


class PostconditionAssignmentIterator(object):
	def __init__(self, context, action_wrapper, postcon_constraints):
		self.context = context
		self.action_wrapper = action_wrapper
		self.assignment_it  = ParameterAssignmentIterator(action_wrapper._arg_post_p_map, action_wrapper.postcons, postcon_constraints)
		self.perm_iter = None

	def __iter__(self):
		return self

	def next(self):
		while True:
			if self.perm_iter == None:
				self.assignment, self.unbound = self.assignment_it.next()

				if len(self.unbound) != 0:
					ds = self.context.agent.get_data_state()
					self.perm_iter = PermutationIterator({a: ds.dl_iterator(self.action_wrapper.signature[a]) for a in self.unbound})
					continue
			else:
				try:
					self.assignment.update(self.perm_iter.next())
				except StopIteration:
					self.perm_iter = None
					continue

			out_post = {}
			out_pre  = {}
			for p, args in self.action_wrapper.precons.items():
				out_pre[p] = {}
				for t, v in args.items():
					out_pre[p][tuple([self.assignment[x] for x in t])] = v

			for p, args in self.action_wrapper.postcons.items():
				out_post[p] = {}
				for t, v in args.items():
					out_post[p][tuple([self.assignment[x] for x in t])] = v

			return PWrapperInstance(assignments=self.assignment, precons=out_pre, postcons=out_post, action_wrapper=self.action_wrapper)


class PreconditionAssignmentIterator(object):
	def __init__(self, context, action_wrapper, precon_constraints):
		self.context = context
		self.action_wrapper = action_wrapper
		self.assignment_it  = ParameterAssignmentIterator(action_wrapper._arg_pre_p_map, action_wrapper.precons, precon_constraints)
		self.perm_iter = None

	def __iter__(self):
		return self

	def next(self):
		while True:
			if self.perm_iter == None:
				self.assignment, self.unbound = self.assignment_it.next()

				if len(self.unbound) == 0:
					ds = self.context.agent.get_data_state()
					self.perm_iter = PermutationIterator({a: ds.dl_iterator(self.action_wrapper.signature[a]) for a in self.unbound})
			else:
				try:
					self.assignment.update(self.perm_iter.next())
				except StopIteration:
					self.perm_iter = None
					continue

			out_post = {}
			out_pre  = {}
			for p, args in self.action_wrapper.precons.items():
				out_pre[p] = {}
				for t, v in args.items():
					out_pre[p][tuple([self.assignment[x] for x in t])] = v

			for p, args in self.action_wrapper.postcons.items():
				out_post[p] = {}
				for t, v in args.items():
					out_post[p][tuple([self.assignment[x] for x in t])] = v

			return PWrapperInstance(assignments=self.assignment, precons=out_pre, postcons=out_post, action_wrapper=self.action_wrapper)


class ParameterAssignmentIterator(object):
	def __init__(self, param_constraint_map, constraint_map, clue_set):
		self.param_constraint_map = param_constraint_map
		self.constraint_map = constraint_map
		self.remaining = set()
		for p in clue_set.keys():
			if p in self.constraint_map:
				for args in self.constraint_map[p].keys():
					for a in args:
						self.remaining.add(a)
		self.free_set = set(self.param_constraint_map.keys()).difference(self.remaining)
		self.pIit = iter(self.param_constraint_map[self.remaining.pop()])
		self.clue_set = clue_set
		self.assignments = {}
		self.problem_stack = []
		self.poIter = None
		self.previous_assignments = set()

	def __iter__(self):
		return self

	def next(self):
		while True:
			try:
				if self.poIter == None:
					# print('New param-value iterator needed')
					while True:
						self.pI = self.pIit.next()
						if self.pI.predicate in self.clue_set:
							self.poIter = self.clue_set[self.pI.predicate].iteritems()
							# print('{}New param-value iterator for {} will iterate over postconditions. It should do {} steps.\n   Id:{}'.format('  '*len(self.problem_stack), self.pI.predicate.P, len(self.clue_set[self.pI.predicate]), id(self.poIter)))
							# print('{}It will iterate over:\n  {}{}'.format('  '*len(self.problem_stack), '  '*len(self.problem_stack), '\n  {}'.format('  '*len(self.problem_stack)).join(['({}) : {}'.format(', '.join(params), value) for params, value in self.clue_set[self.pI.predicate].items()])))
							break
						if self.poIter != None:
							raise Exception('Param-value iterator should be None when this point is reached!')
				# If this predicate is contained in the post conditions
				try:
					tab_space = '  '*len(self.problem_stack)
					# print('{}Param-value iterator is about to do next. Id: {}'.format(tab_space, id(self.poIter)))
					params, value = self.poIter.next()
					if value == self.pI.value:
						# print('{}- {}({})'.format('  ' * len(self.problem_stack), self.pI.predicate.P, ', '.join(params)))
						new_remainder = self.remaining.copy()
						for x in range(len(params)):
							self.assignments[self.pI.args[x]] = params[x]
							if self.pI.args[x] in self.remaining:
								new_remainder.remove(self.pI.args[x])
							# print('{} Assigned: {}'.format(tab_space, self.pI.args[x]))

						if len(new_remainder) == 0:
							hashable_dict = tuple(sorted(self.assignments.items()))
							if not hashable_dict in self.previous_assignments:
								ok = True
								for pI, pAssignments in self.constraint_map.items():
									if pI in self.clue_set:
										for args, v in pAssignments.items():
											rArgs = tuple([self.assignments[a] for a in args])
											if rArgs in self.clue_set[pI] and self.clue_set[pI][rArgs] != v:
												ok = False
												break
									if not ok:
										break
								if ok:
									# print('New assignment checked out. Returning it.')
									self.previous_assignments.add(hashable_dict)
									return self.assignments.copy(), self.free_set
								# print('New assignment did not check out.')
						else:
							# print('New assignment is incomplete. Pushing current state to stack and trying to solve for the next parameter.')
							self.problem_stack.append((self.remaining, self.pIit, self.poIter, self.pI))
							# print('{}Old remainder: ( {} )'.format(tab_space, ', '.join(self.remaining)))
							self.remaining = new_remainder
							# print('{}New remainder: ( {} )'.format(tab_space, ', '.join(self.remaining)))
							self.pIit = iter(self.param_constraint_map[self.remaining.pop()])
							self.poIter = None
							self.pI = None
				except StopIteration:
					# print('Next of {} killed it.'.format(id(self.poIter)))
					self.poIter = None
					# print('Param-value iterator has reached the end of set')
			except StopIteration:
				if len(self.problem_stack) > 0:
					# print('Predicate iterator has reached the end of its set. Luckily there\'s still at least one on the stack.')
					self.remaining, self.pIit, self.poIter, self.pI = self.problem_stack.pop()
				else:
					# print('Predicate iterator has reached the end of its set. The stack is empty, so we are aborting.')
					raise StopIteration
