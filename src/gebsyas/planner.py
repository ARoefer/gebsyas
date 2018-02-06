import traceback
from Queue import PriorityQueue
from gebsyas.actions import PBasedActionSequence
from gebsyas.numeric_scene_state import AssertionDrivenPredicateState

class PlanningProblem:
	def __init__(self, context, Id, initial, goal, lhs, rhs):
		self.Id = Id
		self.initial = initial
		self.goal = goal
		self.lhs = lhs
		self.rhs = rhs
		self.post_con_action_map = context.agent.get_actions().get_postcon_map()
		self.__sub_counter = 0

		self.state_diff = goal.difference(context, initial)

		self.action_map = {}
		for p, args in self.state_diff.items():
			truth_values = set(args.values())
			if p not in self.post_con_action_map:
				raise Exception('Unsolvable planning problem. There\'s no action that can change the value of {}'.format(p.P))
			else:
				for action in self.post_con_action_map[p]:
					if len(set(action.postcons[p].values()) & truth_values) > 0:
						if action in self.action_map:
							self.action_map[action] += 1
						else:
							self.action_map[action] = 1

		# There should be a re-evaluation of the cost based on the context here

		# Actions with a high degree of effectiveness are moved to the front
		self.actionIt       = iter([t[1] for t in sorted([(a.cost / r, a) for a, r in self.action_map.items()])])
		self.current_action = self.actionIt.next()
		self.instanceIt     = None

	def __lt__(self, other):
		return self.lhs.cost + self.rhs.cost + self.current_action.cost < other.lhs.cost + other.rhs.cost + other.current_action.cost

	def plan(self, context, problem_heap):
		while True:
			if self.instanceIt == None:
				self.instanceIt = self.current_action.parameterize_by_postcon(context, self.state_diff)
			try:
				instance = self.instanceIt.next()
				#raise Exception('TODO: Reconsider this construction. Postcons would actually needed to be removed, or matched with their value in the initial state.')
				new_goal_state = AssertionDrivenPredicateState(instance.precons.copy(), self.goal).erase(instance.postcons)


				context.log('   {}'.format('\n   '.join(['{}: {}'.format(a, x) for a, x in instance.assignments.items()]) ))
				context.log('New goal: {}'.format(str(new_goal_state)))
				if len(new_goal_state.difference(context, self.initial)) == 0:
					context.log('I believe to have found a solution to my problem')
					return self.lhs.append(instance) + self.rhs
				else:
					try:
						self.__sub_counter += 1
						new_problem = PlanningProblem(context,
													  '{}:{}'.format(self.Id, self.__sub_counter),
													  self.initial,
													  new_goal_state,
													  self.lhs,
													  self.rhs.push(instance))
						problem_heap.put(new_problem)
					except Exception as e:
						traceback.print_exc()
						context.log(e)
			except StopIteration:
				self.instanceIt = None
				self.current_action = self.actionIt.next() # The cost of solving this problem changes after this call
				return None

	def __str__(self):
		flat_state = []
		for p, args in self.state_diff.items():
			for a, v in args.items():
				flat_state.append('{}({}) : {}'.format(p.P, ', '.join(a), v))
		return 'PlanningProblem:\n   {}'.format('\n   '.join(flat_state))


class PlanIterator(object):
	def __init__(self, context, goal):
		self.context = context
		self.goal = goal
		self.problem_heap = PriorityQueue()
		self.problem_heap.put(PlanningProblem(context, 'S', self.context.agent.get_predicate_state(), self.goal, PBasedActionSequence(), PBasedActionSequence()))

	def __iter__(self):
		return self

	def next(self):
		while True:
			if self.problem_heap.empty():
				raise StopIteration

			problem = self.problem_heap.get()
			try:
				next_plan = problem.plan(self.context, self.problem_heap)
				self.problem_heap.put(problem)
				if next_plan != None:
					return next_plan
			except StopIteration:
				self.context.log('Could not solve {}:\n {}'.format(problem.Id, str(problem)))


