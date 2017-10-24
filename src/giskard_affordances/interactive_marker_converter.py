#!/usr/bin/env python
import sys
import rospy
from visualization_msgs.msg import InteractiveMarkerInit as IMIMsg
from visualization_msgs.msg import InteractiveMarkerFeedback as IMFMsg
from visualization_msgs.msg import Marker
from giskard_affordances.msg import ProbabilisticObject as POMsg

publisher = None
objects = {}

def full_update_cb(markers):
    print('Received update containing {} markers.'.format(len(markers.markers)))
    for marker in markers.markers:
        msg = POMsg()
        msg.header = marker.header
        msg.id = marker.name
        msg.pose = marker.pose

        control = marker.controls[0]
        if control.markers[0].type == Marker.SPHERE:
            msg.semantic_class = 'ball'
        elif control.markers[0].type == Marker.CYLINDER:
            msg.semantic_class = 'cylinder'
        msg.dimensions = control.markers[0].scale

        msg.probability_class = 1.0
        msg.probability_position = 1.0
        msg.probability_rotation = 1.0

        objects[msg.id] = msg

        publisher.publish(msg)

def partial_update_cb(update):
    if update.event_type == 0 or update.event_type == 1:
        if update.marker_name in objects:
            objects[update.marker_name].pose = update.pose

            publisher.publish(objects[update.marker_name])


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: <MARKER TOPIC PREFIX> <PO TOPIC>')
        exit(0)

    rospy.init_node('interactive_marker_converter')

    subscriber = rospy.Subscriber('/{}/update_full'.format(sys.argv[1]), IMIMsg, full_update_cb, queue_size=10)
    fb_sub     = rospy.Subscriber('/{}/feedback'.format(sys.argv[1]), IMFMsg, partial_update_cb, queue_size=10)
    publisher  = rospy.Publisher(sys.argv[2], POMsg, queue_size=3)

    rospy.spin()