#!/usr/bin/env python
import rospy
import pybullet as pb
import sys
from iai_bullet_sim.realtime_simulator_node import FixedTickSimulator
from gebsyas.simulator_plugins import FullPerceptionPublisher
from iai_bullet_sim.probabilistic_interactive_node import ProbabilisticInteractiveNode

if __name__ == '__main__':
    rospy.init_node('gebsyas_realtime_simulator')

    mode = 'direct' if len(sys.argv) < 2 else sys.argv[1]
    if mode not in ['direct', 'gui']:
        print('Mode {} not recognized. Options are: "direct" "gui"'.format(mode))
        exit(-1)

    node = FixedTickSimulator(ProbabilisticInteractiveNode)
    node.init_from_rosparam('sim_config', mode=mode)
    node.run()

    while not rospy.is_shutdown():
        pass

    node.kill()