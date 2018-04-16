from collections import namedtuple
from time import time

from gebsyas.basic_controllers import InEqController
from gebsyas.dl_reasoning import DLSymbolic, SymbolicData, DLManipulator
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.simulator import BulletSimulator, frame_tuple_to_sym_frame
from gebsyas.utils import StampedData, rot3_to_quat
from gebsyas.predicates import IsControlled, IsGrasped
from gebsyas.closest_point_queries import ClosestPointQuery_AnyN, ClosestPointQuery_Specific_SA, ClosestPointQuery_Specific_BA
from gebsyas.numeric_scene_state import visualize_obj
from giskardpy import print_wrapper
from giskardpy.input_system import Point3Input, Vec3Input
from giskardpy.qp_problem_builder import SoftConstraint as SC
from giskardpy.qpcontroller import QPController
from giskardpy.symengine_wrappers import *
import pybullet as pb
import rospy
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF
from gebsyas.dl_urdf_tools import add_dl_object_to_urdf

LinkCDInput = namedtuple('LinkCDInput', ['in_on_link', 'in_in_world', 'in_normal', 'safety_margin'])


class InEqBulletController(InEqController):
    """
    @brief      This controller is a specialization of the inequality controller, but also provides a simple collision avoidance system.
    """
    def __init__(self, context, stamped_objects, ineq_constraints, ppl=3, builder_backend=None, weight=1, logging=print_wrapper):
        """
        Constructor. Uses a context and a map of stamped objects to create collision avoidance scene.
        """
        self.simulator = BulletSimulator(50)
        self.simulator.init()
        self.closest_point_queries = []
        self.controlled_objects = {}
        self.robot_name = context.agent.robot.urdf_robot.name
        self.filter_set = {self.robot_name}
        self.closest_point_queries = []
        self.predicate_state = context.agent.get_predicate_state()

        temp_urdf = URDF.from_xml(context.agent.robot.urdf_robot.to_xml())

        for Id, stamped in stamped_objects:
            # Static objects
            if type(stamped.data) != SymbolicData:
                self.simulator.add_object(stamped.data)
            # Attached objects
            elif context.agent.get_predicate_state().evaluate(context, IsControlled, (Id, )):
                stamped = stamped.data

                # Find the controlling manipulator. This could be standardized
                for manipulator_id in context.agent.get_data_state().dl_iterator(DLManipulator):
                    if context.agent.get_predicate_state().evaluate(context, IsGrasped, (manipulator_id, Id)):
                        manipulator = context.agent.get_predicate_state().map_to_numeric(manipulator_id).data
                        num_object  = context.agent.get_predicate_state().map_to_numeric(Id).data
                        obj_in_manipulator = manipulator.pose.inv() * num_object.pose

                        add_dl_object_to_urdf(temp_urdf, manipulator.link_name, num_object, obj_in_manipulator)

                        logging(stamped.data.pose)
                        self.controlled_objects[Id] = num_object
                        # Avoid the environment
                        # self.closest_point_queries.append((ClosestPointQuery_AnyN(self.robot_name, Id, stamped.data.pose, filter=self.filter_set, n=6), 0.03))
                        # # Avoid the torso
                        # self.closest_point_queries.append((ClosestPointQuery_Specific_BA(self.robot_name, Id,
                        #                                                                 self.robot_name, 'torso_lift_link',
                        #                                                                 stamped.data.pose,
                        #                                                                 context.agent.robot.frames['torso_lift_link']), 0.03))
                        break

        f = open('{}_temp.urdf'.format(self.robot_name), 'w+')
        f.write(temp_urdf.to_xml_string())
        f.close()
        self.simulator.load_robot('{}_temp.urdf'.format(self.robot_name))

        # for link_name, margin in [('forearm_roll_link', 0.05),
        #                         ('wrist_roll_link',   0.01),
        #                         ('gripper_link',      0.005),
        #                         ('r_gripper_finger_link', 0.001),
        #                         ('l_gripper_finger_link', 0.001)]:
        #     self.closest_point_queries.append((ClosestPointQuery_AnyN(self.robot_name, link_name,
        #                                                               context.agent.robot.frames[link_name],
        #                                                               filter=self.filter_set), margin))

        # for link_name1, link_name2, margin in [('forearm_roll_link', 'torso_lift_link', 0.05),
        #                                        ('wrist_roll_link', 'torso_lift_link', 0.05),
        #                                        ('gripper_link', 'torso_lift_link', 0.05)]:
        #     self.closest_point_queries.append((ClosestPointQuery_Specific_BA(self.robot_name, link_name1,
        #                                                                      self.robot_name, link_name2,
        #                                                                      context.agent.robot.frames[link_name1],
        #                                                                      context.agent.robot.frames[link_name2]), margin))

        self.ppl = ppl
        self.link_cd_inputs = {}
        self.visualizer = ROSVisualizer('bullet_motion_controller/vis')
        self.aabb_border = vec3(*([0.2]*3))
        self.dist_exprs = {}

        super(InEqBulletController, self).__init__(context.agent.robot, ineq_constraints, builder_backend, weight, logging)

    # @profile
    def add_inputs(self, robot):
        pass

    # @profile
    def make_constraints(self, robot):
        """Adds inequality constraints to the controller and generates additional constraints for collision avoidance."""
        super(InEqBulletController, self).make_constraints(robot)
        # start_position = pos_of(start_pose)
        for cpp, margin in self.closest_point_queries:
            if isinstance(cpp, ClosestPointQuery_AnyN):
                for x in range(cpp.n):
                    dist = dot(cpp.point_1_expression(x) - cpp.point_2_expression(x), cpp.normal_expression(x))
                    self._soft_constraints['closest_any_{}_{}_{}'.format(cpp.body_name, cpp.link_name, x)] = SC(margin - dist, 100, 100, dist)
            else:
                dist = dot(cpp.point_1_expression() - cpp.point_2_expression(), cpp.normal_expression())
                self._soft_constraints['closest_{}_{}_to_{}_{}'.format(cpp.body_name, cpp.link_name, cpp.other_body, cpp.other_link)] = SC(margin - dist, 100, 100, dist)


    # @profile
    def get_next_command(self):
        """Processes new joint state, updates the simulation and then generates the new command."""
        if self.visualizer:
            self.visualizer.begin_draw_cycle()

        robot_state = self.get_robot().get_state()
        self.simulator.set_joint_positions(self.get_robot().urdf_robot.name, {j: p for j, p in robot_state.items() if j[-10:] != '_cc_weight'})

        for cpp, margin in self.closest_point_queries:
            self.update_observables(cpp.get_update_dict(self.simulator, self.visualizer))

        cmd = super(InEqBulletController, self).get_next_command()
        if self.visualizer:
            for Id, obj in self.controlled_objects.items():
                new_frame = frame_tuple_to_sym_frame(self.simulator.get_link_state(self.robot_name, Id).worldFrame)
                visualize_obj(obj, self.visualizer, new_frame, 'controlled_objects', [0,1,0,1])
                new_numeric = self.predicate_state.map_to_numeric(Id).data
                visualize_obj(obj, self.visualizer, new_numeric.pose, 'controlled_objects', [1,0,0,1])
                # cpp = self.closest_point_queries[0][0]
                # for x in range(cpp.n):
                #     self.visualizer.draw_sphere('ccp', cpp.point_1_expression(x).subs(self.get_state()), 0.01, r=1)
                #     self.visualizer.draw_sphere('ccp', cpp.point_2_expression(x).subs(self.get_state()), 0.01, g=1)
            self.visualizer.render()


        #self.qp_problem_builder.log_jacobian()
        self.qp_problem_builder.log_lbA_ubA()
        return cmd

    def stop(self):
        self.simulator.kill()
