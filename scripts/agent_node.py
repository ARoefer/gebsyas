#!/usr/bin/env python
import sys
import rospy
import imp
import os
from time import time

from guppy import hpy
h = hpy()
from gebsyas.predicates import * # PREDICATE_TBOX, PREDICATES
from gebsyas.agent import BasicAgent
from gebsyas.dl_reasoning import Reasoner, BASIC_TBOX, DLRobot, DLManipulationCapable, DLGripper
from gebsyas.utils import Blank, StampedData
from gebsyas.actions import Context
from gebsyas.generic_motion_action import ACTIONS as MACTIONS
from gebsyas.grasp_action import ACTIONS as GACTIONS
from gebsyas.simple_agent_action import SimpleAgentIOAction

from fetch_giskard.fetch_robot import Fetch
from giskardpy.symengine_robot import Robot
import symengine as sp

from pprint import pprint

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: <FETCH URDF PATH> <MEMORY FILE> [<COMMAND TOPIC>]')
		exit(0)

	rospy.init_node('agent_node')

	from gebsyas.agent import TBOX as ATBOX
	from gebsyas.sensors import TBOX as STBOX
	from gebsyas.trackers import TBOX as TTBOX

	tbox = BASIC_TBOX + ATBOX + STBOX + TTBOX + PREDICATE_TBOX

	reasoner = Reasoner(tbox, {})
	robot = Fetch(sys.argv[1])
	PREDICATES = {p.P: Predicate(p.P, p.fp, tuple([reasoner[dlc] if dlc in reasoner else dlc for dlc in p.dl_args])) for p in PREDICATES.values()}
	capabilities = MACTIONS + GACTIONS

	print('{}'.format('\n'.join([str(action) for action in capabilities])))

	if len(sys.argv) == 2:
		agent = BasicAgent('Elliot', reasoner, PREDICATES, robot, capabilities)
	else:
		agent = BasicAgent('Elliot', reasoner, PREDICATES, robot, capabilities, sys.argv[2])

	# print(reasoner.inclusions_str())
	# print(reasoner.included_str())
	# print(reasoner.tbox_str())

	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), coke), coke.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), pringles), pringles.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), box), box.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), leg1), leg1.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), leg2), leg2.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), leg3), leg3.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), leg4), leg4.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), shelf_floor1), shelf_floor1.id)
	# agent.get_data_state().insert_data(StampedData(rospy.Time.now(), shelf_floor2), shelf_floor2.id)

	# testWrapper = GraspActionInterface()
	# testPostMap = {IsControlled: {('box1',): True}}
	# print(str(testWrapper))

	# last = time()
	# for x in testWrapper.parameterize_by_postcon(Context(agent, agent.logger, agent.visualizer), testPostMap):
	# 	pprint(x)

	agent.awake(SimpleAgentIOAction(agent))