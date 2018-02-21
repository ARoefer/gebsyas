import rospy
from gebsyas.actions import Action
from gebsyas.expression_parser import parse_bool_expr, parse_path, normalize, UnaryOp, BinaryOp, Function, parse_name
from gebsyas.dl_reasoning import bool_expr_tree_to_dl, SymbolicData
from gebsyas.predicate_state_action import PSAction
from gebsyas.utils import YAML
import sys
import traceback

class _Getch:
	"""Gets a single character from standard input.  Does not echo to the
screen."""
	def __init__(self):
		import tty, sys

	def __call__(self):
		import sys, tty, termios
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch


getch = _Getch()

def and_to_set(expr):
	if type(expr) == BinaryOp and expr.op == 'and':
		return and_to_set(expr.a).union(and_to_set(expr.b))
	return {expr}


class SimpleAgentIOAction(Action):
	def __init__(self, agent):
		super(SimpleAgentIOAction, self).__init__('{}\'s IO-Action'.format(agent.name))
		self.classQuery   = 'objects: '
		self.objectQuery  = 'what is '
		self.actionQuery  = 'do: '
		self.dumpQuery    = 'dump: '
		self.memorize     = 'memorize: '
		self.type_structure_query = 'what makes a '
		self.len_last_line = 0

	def printstr(self, string):
		self.len_last_line += len(string)
		sys.stdout.write(string)

	def reprint(self, string):
		sys.stdout.write('{}{}'.format('\b' * (self.len_last_line - len(string)), string))
		self.len_last_line = len(string)

	def println(self, string):
		# self.stdscr.addstr(string)
		# x, y = curses.getsyx()
		self.len_last_line = 0
		# self.stdscr.move(y + 1, 0)
		print(string)

	def execute(self, context):
		# self.stdscr = curses.initscr()

		# curses.noecho()
		# curses.cbreak()
		# self.stdscr.keypad(1)

		self.println("Hi, I'm {}! How may I help you?".format(context.agent.name))
		self.println("I can answer queries about what I know about the world.")
		self.println("I can also try to change something about the world, so that it may be more to your liking.")
		self.println("Queries:")
		self.println("{}X? -- Where X is a dl class".format(self.classQuery))
		self.println("   >> {}Cars and Red?".format(self.classQuery))
		self.println("{}X? -- Where X is the Id of something that I know".format(self.objectQuery))
		self.println("   >> {}object4?".format(self.objectQuery))
		self.println("X? -- You can also inquire about my knowledge of the world using predicates")
		self.println("   >> onTop(box1, table)?")
		self.println("{}X -- I will try to come up with a way to affect the changes needed to make X true".format(self.actionQuery))
		self.println("   >> {}onTop(box1, table) and rightOf(box1, glass1, table/forward)!".format(self.actionQuery))
		self.println("If you're done for the moment you can just say 'bye'.")

		command_history = []
		command_index   = -1

		end_msg = ""


		while not rospy.is_shutdown():
			context.display.begin_draw_cycle()


			command =  'Do: LeftOf(coke, pringles, me) and Upright(coke) and OnTop(coke, table) and PointingAt(camera, coke)'# raw_input('> ') #'Do: LeftOf(coke1, box1, me) and Free(gripper)'

			if command == 'bye' or rospy.is_shutdown():
				break

			if len(command) == 0:
				continue

			if len(command_history) == 0 or command_history[-1] != command:
				command_history.append(command)

			if command[:len(self.classQuery)].lower() == self.classQuery.lower() and command[-1] == '?':
				parser_input = command[len(self.classQuery):-1]
				try:
					concept = bool_expr_tree_to_dl(parse_bool_expr(parser_input)[0], context.agent.get_tbox())
				except Exception as e:
					traceback.print_exc()
					self.printstr(str(e))
					continue
				results = [Id for Id in context.agent.get_data_state().dl_iterator(concept)]

				if results != []:
					self.println('The objects with the following Ids are {}: \n   {}'.format(concept, '\n   '.join(results)))
				else:
					self.println('I know no objects which are {}'.format(concept))

			elif command[:len(self.type_structure_query)].lower() == self.type_structure_query.lower() and command[-1] == '?':
				parser_input = command[len(self.type_structure_query):-1]
				try:
					name, remainder = parse_name(parser_input)
					if remainder == '':
						concept = context.agent.get_tbox()[name]
						self.println('Concept {} is subsumed by:\n  {}'.format(name, str(concept)))
					else:
						self.println('That is not a valid concept name')
				except Exception as e:
					traceback.print_exc()
					self.println(str(e))
					continue
			elif command[:len(self.objectQuery)].lower() == self.objectQuery.lower() and command[-1] == '?':
				parser_input = command[len(self.objectQuery):-1]
				Id, remainder = parse_path(parser_input)
				if remainder == '':
					# Just for nicer interactions
					if Id[:3] == 'you':
						Id = 'me' + Id[3:]
					elif Id[:2] == 'me':
						Id = 'you' + Id[2:]
					################

					try:
						obj = context.agent.get_data_state().find_by_path(Id, True)
						if obj.data != None:
							results = []
							if type(obj.data) == SymbolicData:
								obj = obj.data
								results.append('Symbolic')

							results.extend(context.agent.get_tbox().classify(obj.data))
							if results != []:
								self.println('{} is a... \n   {}'.format(Id, '\n   '.join([str(x) for x in results])))
							else:
								self.println('{} matches nothing in my TBox.'.format(Id))
						else:
							self.println('I don\'t know anything called "{}".'.format(Id))
					except Exception as e:
						traceback.print_exc()
						self.println(e)
						continue
				else:
					self.println('That\'s not a valid Id. Remaining: {}'.format(remainder))
			elif command[:len(self.actionQuery)].lower() == self.actionQuery.lower():
				try:
					expr = parse_bool_expr(command[len(self.actionQuery):])[0]
					goal_set = and_to_set(normalize(expr))

					pred_state = context.agent.get_predicate_state()
					goal_state = {}

					for s in goal_set:
						ts = type(s)
						truth = True
						if ts == UnaryOp and s.op == 'not':
							truth = False
							s = s.a

						if type(s) == Function:
							if s.name in pred_state.predicates:
								p = pred_state.predicates[s.name]
								if p not in goal_state:
									goal_state[p] = {}

								if len(p.dl_args) != len(s.args):
									raise Exception('The predicate {} requires {} arguments. You gave {}'.format(p.P, len(p.dl_args), len(s.args)))
								goal_state[p][s.args] = truth
							else:
								raise Exception('I don\'t know any predicate called {}'.format(s.name))
						else:
							raise Exception('I can currently only do conjunctive goals, so please only use "and" and "not".')

					self.execute_subaction(context, PSAction(goal_state))
				except Exception as e:
					traceback.print_exc()
					self.println(e)
					continue

			elif command[-1] == '?':
				try:
					expr = parse_bool_expr(command[:-1])[0]
					self.println('   {}'.format(str(self.evaluate_bool_query(context.agent.get_predicate_state(), expr, context))))
				except Exception as e:
					traceback.print_exc()
					self.println(e)
					continue
			elif command[:len(self.dumpQuery)].lower() == self.dumpQuery:
				parser_input = command[len(self.dumpQuery):]
				Id, remainder = parse_path(parser_input)
				if remainder == '':
					obj = context.agent.get_data_state()[Id]
					try:
						self.println('[{:>10.4f}] {}'.format(obj.stamp.to_sec(), YAML.dump(obj.data)))
					except Exception as e:
						traceback.print_exc()
						self.println(e)
						continue
				else:
					self.println('That\'s not a valid Id. Remaining: {}'.format(remainder))
			elif command[:len(self.memorize)].lower() == self.memorize:
				command = command[len(self.memorize):]
				parts = [part.replace(' ', '') for part in command.split(' as ')]
				if len(parts) != 2:
					self.println('The syntax of memorize is "Memorize: Thing_I_know as Name_of_memory')
					continue

				Id, remainder = parse_path(parts[0])
				if remainder == '':
					obj = context.agent.get_predicate_state().map_to_data(Id)
					context.agent.memory[parts[1]] = obj.data
					self.println('Memorized {} as {}'.format(parts[0], parts[1]))
				else:
					self.println('That\'s not a valid Id. Remaining: {}'.format(remainder))
			else:
				self.println('I don\'t understand what you mean by "{}"'.format(command))

			context.agent.get_predicate_state().visualize(context.display)
			context.display.render()
			break

	def evaluate_bool_query(self, predicate_state, query, context):
		tq = type(query)
		if tq == bool:
			return query
		elif tq == UnaryOp and query.op == 'not':
			return not self.evaluate_bool_query(predicate_state, query.a, context)
		elif tq == BinaryOp:
			if query.op == 'and':
				return self.evaluate_bool_query(predicate_state, query.a, context) and self.evaluate_bool_query(predicate_state, query.b, context)
			elif query.op == 'or':
				return self.evaluate_bool_query(predicate_state, query.a, context) or self.evaluate_bool_query(predicate_state, query.b, context)
		elif tq == Function:
			if query.name in predicate_state.predicates:
				p = predicate_state.predicates[query.name]
				return predicate_state.evaluate(context, p, tuple(self.__change_pov(query.args)))
		raise Exception('Can\'t evaluate "{}"'.format(str(query)))

	def __change_pov(self, ids):
		out = []
		for x in ids:
			if x[:2] == 'me':
				out.append('you' + x[2:])
			elif x[:3] == 'you':
				out.append('me' + x[3:])
			else:
				out.append(x)
		return out
