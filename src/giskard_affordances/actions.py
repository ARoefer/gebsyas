import rospy
from collections import namedtuple
from giskard_affordances.ros_visualizer import ROSVisualizer

Context = namedtuple('Context', ['agent', 'log', 'display'])
LogData = namedtuple('LogData', ['stamp', 'name', 'data'])

class Logger():
	def __init__(self):
		self.log_birth = rospy.Time.now()
		self.indentation = 0

	def log(self, data, name=''):
		title_str = '{:>6.4f}:{} | '.format((rospy.Time.now() - self.log_birth).to_sec(), '  ' * self.indentation)
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

class GripperAction(Action):
	def __init__(self, gripper, goal_pos, effort=100):
		super(GripperAction, self).__init__('GripperAction: {}'.format(gripper.name))
		self.gripper = gripper
		self.goal    = goal_pos
		self.effort  = effort

	def execute(self, context):
		context.log('Let\'s assume for now that gripper "{}" is now opened {} and did so with {} Nm'.format(self.gripper.name, self.goal, self.effort))
		return 1.0


