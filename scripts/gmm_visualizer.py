#!/usr/bin/env python
import sys
import rospy
import numpy as np

from iai_bullet_sim.basic_simulator import hsva_to_rgba
from gebsyas.ros_visualizer import ROSVisualizer
from giskardpy.symengine_wrappers import vector3, point3, norm, cross
from gop_gebsyas_msgs.msg import SearchObjectList as SearchObjectListMsg

class SearchObjectVisualizer(object):
    def __init__(self, sub_topic, pub_topic, frame='map'):
        self.visualizer = ROSVisualizer(pub_topic, frame)
        self.sub_sol = rospy.Subscriber(sub_topic, SearchObjectListMsg, self.draw_objects, queue_size=1)

    def draw_objects(self, msg):
        self.visualizer.begin_draw_cycle()
        for so in msg.search_object_list:
            for x in range(len(so.object_pose_gmm)):
                gc = so.object_pose_gmm[x]
                gc_cov = np.array(gc.cov_pose.covariance, dtype=float).reshape((6,6))
                np_pos_cov = gc_cov[:3, :3]
                w, v = np.linalg.eig(np_pos_cov)
                pos_eig = v * w

                position = point3(gc.cov_pose.pose.position.x, gc.cov_pose.pose.position.y, gc.cov_pose.pose.position.z)
                self.visualizer.draw_text('labels', position, '{}_{}[{}]'.format(so.name, so.id, x), height=0.15)

                if np.isrealobj(pos_eig) and np.average(pos_eig) != 0.0:
                    x_vec = vector3(*pos_eig[:, 0])
                    y_vec = vector3(*pos_eig[:, 1])
                    #z_vec = vector3(*pos_eig[:, 2])
                    x_vec *= 1.0 / norm(x_vec)
                    y_vec *= 1.0 / norm(y_vec)
                    #z_vec *= 1.0 / norm(z_vec)
                    M = x_vec.row_join(y_vec).row_join(cross(x_vec, y_vec)).row_join(position)

                    self.visualizer.draw_shape('cov', M, w.astype(float), 2, *hsva_to_rgba((1.0 - gc.weight) * 0.65, 1, 1, 0.7))
        self.visualizer.render()


if __name__ == '__main__':
    print('Usage: [<SUBSCRIPTION TOPIC>] [<PUBLISHING TOPIC>] [<FRAME>]')

    rospy.init_node('gmm_visualizer')

    sub_topic = '/object_metric_states' if len(sys.argv) < 2 else sys.argv[1]
    pub_topic = '/gmm_vis'              if len(sys.argv) < 3 else sys.argv[2]
    frame     = 'map'                   if len(sys.argv) < 4 else sys.argv[3]

    print('Subscribing to: {}\nPublishing at:{}\nIn frame:{}'.format(sub_topic, pub_topic, frame))

    node = SearchObjectVisualizer(sub_topic, pub_topic, frame)

    while not rospy.is_shutdown():
        pass