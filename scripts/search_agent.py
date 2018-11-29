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
from gebsyas.dl_reasoning import Reasoner, BASIC_TBOX, DLAtom, DLRigidObject
from gebsyas.utils import Blank, StampedData, bb
from gebsyas.actions import Context
from gebsyas.generic_motion_action import ACTIONS as MACTIONS
from gebsyas.observation_controller import ACTIONS as OACTIONS
from gebsyas.grasp_action import ACTIONS as GACTIONS
from gebsyas.search_behavior import MultiObjectSearchAndDeliveryAction

from iai_bullet_sim.srv import Empty as EmptySrv
from std_srvs.srv import SetBool as SetBoolSrv

from fetch_giskard.fetch_robot import Fetch as Robot
#from gebsyas.floaty_robot import Floaty as Robot
import symengine as sp

from pprint import pprint

if __name__ == '__main__':
    sphere   = bb(radius=0.2, mass=1.0, pose=sp.eye(4))
    cylinder = bb(radius=0.2, height=0.4, mass=1.0, pose=sp.eye(4))
    box = bb(width=0.2, height=0.2, length=0.2, mass=1.0, pose=sp.eye(4))
    print('Is rigid body:\n  sphere = {}\n  cylinder = {}\n  box = {}'.format(DLRigidObject.is_a(sphere), DLRigidObject.is_a(cylinder), DLRigidObject.is_a(box)))

    if len(sys.argv) < 4:
        print('Usage: <FETCH URDF PATH> <MEMORY FILE> <SEARCHED CLASSES>')
        exit(0)

    rospy.init_node('agent_node')

    try:
        srv_reset = rospy.ServiceProxy('/reset', EmptySrv)
        srv_reset()
    except (rospy.ServiceException, rospy.ROSException), e:
        print('Simulator does not seem to be running...')


    stop_bag = False
    try:
        srv_record = rospy.ServiceProxy('/record', SetBoolSrv)
        if srv_record(True).success:
            stop_bag = True
            print('ROSBag is being recorded.')
    except (rospy.ServiceException, rospy.ROSException), e:
        print('Simulator does not seem to be running...')

    from gebsyas.agent import TBOX as ATBOX
    from gebsyas.sensors import TBOX as STBOX
    from gebsyas.trackers import TBOX as TTBOX

    tbox = BASIC_TBOX + ATBOX + STBOX + TTBOX + PREDICATE_TBOX

    reasoner = Reasoner(tbox, {})
    robot = Robot(sys.argv[1])
    PREDICATES = {p.P: Predicate(p.P, p.fp, tuple([reasoner[dlc] if dlc in reasoner else dlc for dlc in p.dl_args])) for p in PREDICATES.values()}
    capabilities = MACTIONS + GACTIONS + OACTIONS

    #print('{}'.format('\n'.join([str(action) for action in capabilities])))

    agent = BasicAgent('Elliot', reasoner, PREDICATES, robot, capabilities, sys.argv[2])




    agent.awake(MultiObjectSearchAndDeliveryAction([a for a in sys.argv[3:] if ':=' not in a], sp.eye(4), sim_mode=True))

    os.system('rostopic pub /nav_to_pose/cancel actionlib_msgs/GoalID "{stamp: {secs: 0, nsecs: 0}, id: ''}" -1')

    if stop_bag:
        os.system('rosservice call /record "data: false"')        