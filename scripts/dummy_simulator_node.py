#!/usr/bin/env python
import rospy
import pybullet as pb
import sys
from iai_bullet_sim.basic_simulator import BasicSimulator
from iai_bullet_sim.basic_simulator_node import BasicSimulatorNode
from iai_bullet_sim.dummy_simulator_node import DummyTickSimulator
from iai_bullet_sim.full_state_node import FullStatePublishingNode
from iai_bullet_sim.full_state_interactive_node import FullStateInteractiveNode
from gebsyas.simulator_plugins import FakeGMMObjectPublisher, GMMObjectPublisher
from gebsyas.probabilistic_interactive_node import ProbabilisticInteractiveNode, ProbabilisticSimulator

if __name__ == '__main__':
    rospy.init_node('gebsyas_realtime_simulator')

    mode = 'direct' if len(sys.argv) < 2 else sys.argv[1]
    if mode not in ['direct', 'gui']:
        print('Mode {} not recognized. Options are: "direct" "gui"'.format(mode))
        exit(-1)

    node = DummyTickSimulator(ProbabilisticInteractiveNode, 'gebsyas_bullet_sim', ProbabilisticSimulator)
    #node = FixedTickSimulator(FullStateInteractiveNode, 'gebsyas_bullet_sim', BasicSimulator)
    node.init_from_rosparam('sim_config', mode=mode)
    if not node.sim.has_plugin_of_type(GMMObjectPublisher):
        node.sim.register_plugin(FakeGMMObjectPublisher())

    node.run()

    while not rospy.is_shutdown():
        pass

    node.kill()