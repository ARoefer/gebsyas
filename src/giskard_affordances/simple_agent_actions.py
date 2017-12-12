import rospy
from giskard_affordances.actions import Action
from giskard_affordances.expression_parser import parse_bool_expr, parse_path, UnaryOp, BinaryOp, Function
from giskard_affordances.dl_reasoning import bool_expr_tree_to_dl, SymbolicData
from giskard_affordances.predicate_state_action import PSAction
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


class SimpleAgentIOAction(Action):
	def __init__(self, agent):
		super(SimpleAgentIOAction, self).__init__('{}\'s IO-Action'.format(agent.name))
		self.classQuery   = 'Objects: '
		self.objectQuery  = 'What is '
		self.actionQuery  = 'Do: '
		self.dumpQuery    = 'Dump: '
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


			command = 'Do: Grasped(gripper, box1)' #raw_input('> ') #

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
				results = []
				for Id, obj in context.agent.get_data_state().get_data_map().items():
					if concept.is_a(obj.data):
						results.append(Id)

				if results != []:
					self.println('The objects with the following Ids are {}: \n   {}'.format(concept, '\n   '.join(results)))
				else:
					self.println('I know no objects which are {}'.format(concept))

			elif command[:len(self.objectQuery)].lower() == self.objectQuery.lower() and command[-1] == '?':
				parser_input = command[len(self.objectQuery):-1]
				Id, remainder = parse_path(parser_input)
				if remainder == '':
					obj = context.agent.get_data_state()[Id]
					if obj.data != None:
						results = []
						if type(obj.data) == SymbolicData:
							obj = obj.data
							results.append('Symbolic')

						for k, c in context.agent.get_tbox().items():
							if c.is_a(obj.data):
								results.append(k)
						if results != []:
							self.println('{} is a... \n   {}'.format(Id, '\n   '.join(results)))
						else:
							self.println('{} matches nothing in my TBox.'.format(Id))
					else:
						self.println('I don\'t know anything called "{}".'.format(Id))
				else:
					self.println('That\'s not a valid Id. Remaining: {}'.format(remainder))
			elif command[:len(self.actionQuery)].lower() == self.actionQuery.lower():
				try:
					expr = parse_bool_expr(command[len(self.actionQuery):])[0]
				except Exception as e:
					self.println(e)
					continue
				goal = {}
				predicate_state = context.agent.get_predicate_state()
				if type(expr) == Function:
					if expr.name in predicate_state.predicates:
						p = predicate_state.predicates[expr.name]
						goal[(p, tuple(expr.args))] = True
					else:
						self.println('Unknown predicate "{}"'.format(expr.name))
						continue
				elif type(expr) == UnaryOp and expr.op == 'not' and type(expr.a) == Function:
					expr = expr.a
					if expr.name in predicate_state.predicates:
						p = predicate_state.predicates[expr.name]
						goal[(p, tuple(expr.args))] = False
					else:
						self.println('Unknown predicate "{}"'.format(expr.name))
				else:
					self.println('Currently I can only work with goals of the form "P(...)" or "not P(...)"')
				self.execute_subaction(context, PSAction(goal))
			elif command[-1] == '?':
				try:
					expr = parse_bool_expr(command[:-1])[0]
					self.println('   {}'.format(str(self.evaluate_bool_query(context.agent.get_predicate_state(), expr, context))))
				except Exception as e:
					traceback.print_exc()
					self.println(e)
					continue
			elif command[:len(self.dumpQuery)] == self.dumpQuery:
				parser_input = command[len(self.dumpQuery):]
				Id, remainder = parse_path(parser_input)
				if remainder == '':
					obj = context.agent.get_data_state()[Id]
					self.println('[{:>10.4f}] {}'.format(obj.stamp.to_sec(), str(obj.data)))
				else:
					self.println('That\'s not a valid Id. Remaining: {}'.format(remainder))
			else:
				self.println('I don\'t understand what you mean by "{}"'.format(command))

			context.display.render()
			break

		# curses.nocbreak()
		# self.stdscr.keypad(0)
		# curses.echo()
		# curses.endwin()

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
				return predicate_state.evaluate(p, tuple(query.args), context.log)
		raise Exception('Can\'t evaluate "{}"'.format(str(query)))

