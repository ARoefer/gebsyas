#!/usr/bin/env python
import sys
import rospy
from giskard_affordances.predicates import PREDICATE_TBOX, PREDICATES
from giskard_affordances.agent import SimpleAgent, AGENT_TBOX
from giskard_affordances.dl_reasoning import Reasoner, BASIC_TBOX, DLRobot, DLManipulationCapable, DLGripper
from giskard_affordances.utils import Blank, StampedData
from fetch_giskard.fetch_robot import Fetch
from giskardpy.robot import Robot
import symengine as sp

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: <FETCH URDF PATH> [<COMMAND TOPIC>]')
		exit(0)

	rospy.init_node('agent_node')

	coke = Blank()
	coke.height = 0.126
	coke.id = 'coke1'
	coke.pose = sp.Matrix([[0.999999981757682, -0.000270128250122051, 6.38437810455662e-09, 0.701818287372589],
						   [0.00027012825012209, 0.999999981757682, -6.14578899971901e-09, -0.169894263148308],
						   [-6.38271772042708e-09, 6.14751337634618e-09, 1.00000001824232, 0.778657257556915],
						   [0, 0, 0, 1]])

	coke.presurized = True
	coke.probability_class = 1.0
	coke.probability_position = 1.0
	coke.probability_rotation = 1.0
	coke.radius = 0.034
	coke.semantic_class = 'coke'


	table = Blank()
	table.id = 'table'
	table.height = 0.004
	table.length = 0.6
	table.pose = sp.Matrix([[0.999956657779764, -9.4670595661624e-05, 0.00931411520722289, 0.759454309940338],
							[6.42468572674571e-05, 0.999994701968877, 0.00326665719189807, 0.02177594602108],
							[-0.00931437474855048, -0.0032659170758556, 0.999951326491139, 0.711342215538025],
							[0, 0, 0, 1]])
	table.probability_class = 1.0
	table.probability_position = 1.0
	table.probability_rotation = 1.0
	table.semantic_class = 'cube'
	table.width = 0.6

	pringles = Blank()
	pringles.height = 0.248
	pringles.id = 'pringles1'
	pringles.pose = sp.Matrix([[1.0, 0, 0, 0.736358106136322],
							   [0, 1.0, 0, 0.153913110494614],
							   [0, 0, 1.0, 0.835449755191803],
							   [0, 0, 0, 1]])

	pringles.probability_class = 1.0
	pringles.probability_position = 1.0
	pringles.probability_rotation = 1.0
	pringles.radius = 0.078 * 0.5
	pringles.semantic_class = 'pringles'

	box = Blank()
	box.id = 'box1'
	box.height = 0.22
	box.length = 0.06
	box.width = 0.06
	box.pose = sp.Matrix([[1.0, 0, 0, 0.787988841533661],
						  [0, 1.0, 0, 0.00784990191459656],
						  [0, 0, 1.0, 0.821733474731445],
						  [0, 0, 0, 1]])

	box.probability_class = 1.0
	box.probability_position = 1.0
	box.probability_rotation = 1.0
	box.semantic_class = 'cube'



	tbox = BASIC_TBOX
	tbox.update(AGENT_TBOX)
	tbox.update(PREDICATE_TBOX)
	reasoner = Reasoner(tbox, {})
	robot = Fetch(sys.argv[1])
	agent = SimpleAgent('Elliot', reasoner, PREDICATES, robot)
	agent.get_data_state().insert_data(StampedData(rospy.Time.now(), coke), coke.id)
	agent.get_data_state().insert_data(StampedData(rospy.Time.now(), table), table.id)
	agent.get_data_state().insert_data(StampedData(rospy.Time.now(), pringles), pringles.id)
	agent.get_data_state().insert_data(StampedData(rospy.Time.now(), box), box.id)
	agent.awake()