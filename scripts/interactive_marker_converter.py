#!/usr/bin/env python
import sys
import rospy
import tf

from geometry_msgs.msg import PoseStamped
from gebsyas.msg import ProbabilisticObject as POMsg
from std_msgs.msg import Empty
from visualization_msgs.msg import InteractiveMarkerInit as IMIMsg
from visualization_msgs.msg import InteractiveMarkerFeedback as IMFMsg
from visualization_msgs.msg import Marker

publisher = None
objects = {}
listener = None

def full_update_cb(markers):
    print('Received update containing {} markers.'.format(len(markers.markers)))
    for marker in markers.markers:
        msg = POMsg()
        msg.header = marker.header
        msg.header.stamp = rospy.Time.now()
        msg.id = marker.name
        if msg.header.frame_id == 'base_link':
            msg.pose = marker.pose
        else:
            try:
                ps = PoseStamped()
                ps.header = msg.header
                listener.waitForTransform('base_link', ps.header.frame_id, rospy.Time(0), rospy.Duration(0.2))
                newPose = listener.transformPose('base_link', ps)
                msg.pose = ps.pose
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print('Lookup of {} in base_link failed'.format(msg.header.frame_id))

        control = marker.controls[0]
        if control.markers[0].type == Marker.SPHERE:
            msg.semantic_class = 'sphere'
        elif control.markers[0].type == Marker.CYLINDER:
            msg.semantic_class = 'cylinder'
        elif control.markers[0].type == Marker.CUBE:
            msg.semantic_class = 'cube'
        elif control.markers[0].type == Marker.MESH_RESOURCE:
            msg.semantic_class = control.markers[0].mesh_resource[control.markers[0].mesh_resource.rfind('/') + 1:-4]
        msg.dimensions = control.markers[0].scale

        msg.probability_class = 1.0
        msg.probability_position = 1.0
        msg.probability_rotation = 1.0

        objects[msg.id] = msg

        publisher.publish(msg)

def partial_update_cb(update):
    if update.event_type == 0 or update.event_type == 1:
        if update.marker_name in objects:
            objects[update.marker_name].header.stamp = rospy.Time.now()
            if update.header.frame_id == 'base_link':
                objects[update.marker_name].pose = update.pose
            else:
                try:
                    ps = PoseStamped()
                    ps.header = update.header
                    ps.header.stamp = rospy.Time.now()
                    listener.waitForTransform('base_link', ps.header.frame_id, rospy.Time(0), rospy.Duration(0.2))
                    newPose = listener.transformPose('base_link', ps)
                    objects[update.marker_name].pose = ps.pose
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print('Lookup of {} in base_link failed'.format(update.header.frame_id))


            publisher.publish(objects[update.marker_name])

def republish(msg):
    print('I\'ll show you what I got!')
    for obj in objects.values():
        publisher.publish(obj)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: <MARKER TOPIC PREFIX> <PO TOPIC>')
        exit(0)

    rospy.init_node('interactive_marker_converter')

    listener = tf.TransformListener()

    publisher  = rospy.Publisher(sys.argv[2], POMsg, queue_size=12)
    show_me_what_you_got = rospy.Subscriber('/show_me_what_you_got', Empty, republish, queue_size=1)
    subscriber = rospy.Subscriber('/{}/update_full'.format(sys.argv[1]), IMIMsg, full_update_cb, queue_size=10)
    fb_sub     = rospy.Subscriber('/{}/feedback'.format(sys.argv[1]), IMFMsg, partial_update_cb, queue_size=10)

    rospy.spin()