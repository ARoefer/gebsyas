import rospy
import unittest
import time

from gebsyas.core.data_state import DataState
from gebsyas.core.subs_ds import ks_from_obj
from gebsyas.kinematics.kinematic_state import KinematicState
from gebsyas.kinematics.kinematic_rule  import find_obj_ids_from_so
from gebsyas.utils import bb, visualize_obj
from gebsyas.ros_visualizer import ROSVisualizer

from giskardpy.symengine_wrappers import frame3_rpy, point3, translation3



class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.vis = ROSVisualizer('kso_visuals')
        self.kstate = KinematicState(DataState())

        c_pose   = frame3_rpy(0, 0, 0, point3(0,0,0.25))
        c_length = 0.4
        c_width  = 0.4
        c_height = 0.5
        self.compartment = cube(('compartment',), c_pose, 
                                c_length, c_width, c_height, 
                                self.kstate.data_state.value_table)

        hinge_transform = frame3_rpy(0,0,1.56, point3(-c_length * 0.5,
                                                      -c_width  * 0.5, 0))

        d_pose   = hinge_transform * translation3(-0.005, c_width  * 0.5, 0)
        d_width  = c_width
        d_length = 0.01
        d_height = c_height
        self.door = cube(('door',), c_pose, 
                         d_length, d_width, d_height, 
                         self.kstate.data_state.value_table)


        h_length = 0.04
        h_width  = 0.01
        h_height = 0.15
        h_pose = d_pose * translation3(-0.02, d_width * 0.5 - 0.05, d_height * 0.5 - 0.05 - 0.5 * h_height)

        self.handle = cube(('handle',), h_pose,
                           h_length, h_width, h_height, 
                           self.kstate.data_state.value_table)


    def test_scene_layout(self):
        self.vis.begin_draw_cycle('scene')

        compartment = self.compartment.subs(self.kstate.data_state.value_table)
        door        = self.door.subs(self.kstate.data_state.value_table)
        handle      = self.handle.subs(self.kstate.data_state.value_table)

        visualize_obj(compartment, self.vis, compartment.pose, 'scene', (0.4, 0.1,   0, 1))
        visualize_obj(       door, self.vis,        door.pose, 'scene', (0.8, 0.8, 0.6, 1))
        visualize_obj(     handle, self.vis,      handle.pose, 'scene', (0.6, 0.6, 0.6, 1))

        self.vis.render()


if __name__ == '__main__':
    rospy.init_node('test_kinematics_rule')

    time.sleep(0.3)

    unittest.main()

    time.sleep(0.3)