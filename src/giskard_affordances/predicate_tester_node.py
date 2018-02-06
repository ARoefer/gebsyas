#!/usr/bin/env python
import sys
import rospy

from symengine import *

from giskard_affordances.sensors import TopicSensor
from giskard_affordances.msg import ProbabilisticObject as PObject
from giskard_affordances.numeric_scene_state import DataSceneState
from giskard_affordances.utils import Blank, bb
from giskard_affordances.expression_parser import *
from giskard_affordances.predicates import *
from giskard_affordances.predicate_tester import PredicateTester
from giskardpy.symengine_wrappers import *
import std_msgs
from pprint import pprint

class PredicateTesterNode(object):
	def __init__(self, predicates):
		self.obj_sensor = TopicSensor(self.on_object_seen, '/perceived_objects', PObject, 12)
		self.smwyg_pub = rospy.Publisher('/show_me_what_you_got', std_msgs.msg.Empty, queue_size=1)
		self.data_state = DataSceneState()
		self.observer = bb(frame_of_reference=frame3_rpy(0, 0, 0, point3(0,0,1)))
		self.data_state.insert_data(StampedData(rospy.Time.now(), self.observer), 'observer')
		self.predicates = predicates
		self.tester = PredicateTester()
		self.obj_sensor.enable()

	def on_object_seen(self, stamped_object):
		self.data_state.insert_data(stamped_object, stamped_object.data.id)

	def run(self):
		rospy.sleep(0.3)
		self.smwyg_pub.publish(std_msgs.msg.Empty())
		last_command = ''
		rospy.sleep(0.3)
		while not rospy.is_shutdown():
			print('known predicates:\n  {}'.format('\n  '.join(['{}({})'.format(p.P, len(p.dl_args)) for p in self.predicates.values()])))
			print('Known objects:\n  {}'.format('\n  '.join(self.data_state.id_map.keys())))
			#command = 'Upright(sugar) INIT: sugar/pose OPT: sugar/pose' #raw_input('Please give: PREDICATE(x1, ..., xn),* INIT: VALUE* OPT: VALUE*\n')
			command = 'Below(sugar, table, observer), Upright(sugar) INIT: sugar/pose OPT: sugar/pose'
			if command == 'quit' or rospy.is_shutdown():
				break

			if command == 'l':
				command = last_command
			else:
				last_command = command

			predicates_str, v_opt_str = tuple(command.split('INIT:'))
			values_str, opts_str = tuple(v_opt_str.split('OPT:'))
			values, r = parse_homogenous_list(values_str, parse_path)
			opts, r   = parse_homogenous_list(opts_str, parse_path)
			plist, r = parse_homogenous_list(predicates_str, parse_bool_atom)
			pprint(plist)
			predicates = [PInstance(self.predicates[fn.name], fn.args, True) for fn in plist]
			self.tester.envision_predicates(predicates, self.data_state, values, opts, tries=20)
			break


if __name__ == '__main__':
	rospy.init_node('predicate_tester')

	node = PredicateTesterNode(PREDICATES)
	node.run()
