import pybullet as pb
import rospy

from collections import namedtuple
from time import time

from gebsyas.basic_controllers import InEqController
from gebsyas.closest_point_queries import ClosestPointQuery_AnyN, ClosestPointQuery_Specific_SA, ClosestPointQuery_Specific_BA
from gebsyas.dl_reasoning import DLSymbolic, SymbolicData, DLManipulator
from gebsyas.dl_urdf_tools import add_dl_object_to_urdf
from gebsyas.predicates import IsControlled, IsGrasped
from gebsyas.numeric_scene_state import visualize_obj
from gebsyas.simulator import GebsyasSimulator, frame_tuple_to_sym_frame
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.utils import StampedData, rot3_to_quat
from giskardpy import print_wrapper
from giskardpy.qp_problem_builder import SoftConstraint as SC
from giskardpy.symengine_wrappers import *
from iai_bullet_sim.utils import Frame
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF

LinkCDInput = namedtuple('LinkCDInput', ['in_on_link', 'in_in_world', 'in_normal', 'safety_margin'])


class InEqBulletController(InEqController):
    """
    @brief      This controller is a specialization of the inequality controller, but also provides a simple collision avoidance system.
    """
    def __init__(self, context, avoid_collisions, allow_collisions, ppl=3, logging=print_wrapper):
        """
        Constructor. Uses a context and a map of stamped objects to create collision avoidance scene.
        """
        super(InEqBulletController, self).__init__(context.agent.robot, logging)
        self.simulator = GebsyasSimulator(50)
        self.simulator.init()
        self.controlled_objects = {}
        robot_name = context.agent.robot._urdf_robot.name
        self.closest_point_queries = []
        self.predicate_state = context.agent.get_predicate_state()

        temp_urdf = URDF.from_xml(context.agent.robot._urdf_robot.to_xml())

        self.included_objects = []

        for Id, stamped in avoid_collisions:
            # Static objects
            if Id not in allow_collisions:
                if type(stamped.data) != SymbolicData and 'floor' not in Id:
                    self.simulator.add_object(stamped.data)
                    self.included_objects.append(stamped.data)
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
                            # self.closest_point_queries.append((ClosestPointQuery_AnyN(robot_name, Id, stamped.data.pose, filter=self.filter_set, n=6), 0.03))
                            # # Avoid the torso
                            # self.closest_point_queries.append((ClosestPointQuery_Specific_BA(robot_name, Id,
                            #                                                                 robot_name, 'torso_lift_link',
                            #                                                                 stamped.data.pose,
                            #                                                                 context.agent.robot.frames['torso_lift_link']), 0.03))
                            break

        f = open('{}_temp.urdf'.format(robot_name), 'w+')
        f.write(temp_urdf.to_xml_string())
        f.close()
        self.bullet_bot = self.simulator.load_urdf('{}_temp.urdf'.format(robot_name))
        self.filter_set = {self.bullet_bot}

        self.avoidance_constraints = {}
        for link, (margin, blowup) in self.robot.collision_avoidance_links.items():
            cpq = ClosestPointQuery_AnyN(self.bullet_bot, link, self.robot.get_fk_expression('map', link), margin, filter=self.filter_set, n=2, aabb_border=blowup)
            self.closest_point_queries.append(cpq)
            self.avoidance_constraints.update(cpq.generate_constraints())

        for link_a, link_b, margin in self.robot.self_collision_avoidance_pairs:
            cpq = ClosestPointQuery_Specific_BA(self.bullet_bot, link_a, self.bullet_bot, link_b, 
                                                self.robot.get_fk_expression('map', link_a),
                                                self.robot.get_fk_expression('map', link_b),
                                                margin)

        self.ppl = ppl
        self.link_cd_inputs = {}
        self.visualizer = ROSVisualizer('bullet_motion_controller/vis', 'map')
        self.aabb_border = vector3(*([0.2]*3))
        self.dist_exprs = {}
        self.data_state = context.agent.get_data_state()
        self.obstacle_color = (0,1,0,1)

        # start_position = pos_of(start_pose)


    def init(self, soft_constraints):
        soft_constraints = soft_constraints.copy()
        soft_constraints.update(self.avoidance_constraints)
        #self.hard_constraints.update(self.avoidance_constraints)
        super(InEqBulletController, self).init(soft_constraints)

    # @profile
    def get_cmd(self, nWSR=None):
        """Processes new joint state, updates the simulation and then generates the new command."""
        if self.visualizer:
            self.visualizer.begin_draw_cycle()

        self.bullet_bot.set_joint_positions({j: self.current_subs[self.robot.joint_states_input.joint_map[j]] for j in self.robot.get_joint_names()})

        localization = self.data_state['localization'].data
        quat = pb.getQuaternionFromEuler((0,0,localization.az))
        self.bullet_bot.set_pose(Frame((localization.x, localization.y, localization.z), quat))

        for cpp in self.closest_point_queries:
            cpp.update_subs_dict(self.simulator, self.current_subs, self.visualizer)

        cmd = super(InEqBulletController, self).get_cmd()

        if self.visualizer:
            for Id, obj in self.controlled_objects.items():
                new_frame = frame_tuple_to_sym_frame(self.bullet_bot.get_link_state(Id).worldFrame)
                visualize_obj(obj, self.visualizer, new_frame, 'controlled_objects', [0,1,0,1])
                new_numeric = self.predicate_state.map_to_numeric(Id).data
                visualize_obj(obj, self.visualizer, new_numeric.pose, 'controlled_objects', [1,0,0,1])
                # cpp = self.closest_point_queries[0][0]
                # for x in range(cpp.n):
                #     self.visualizer.draw_sphere('ccp', cpp.point_1_expression(x).subs(self.get_state()), 0.01, r=1)
                #     self.visualizer.draw_sphere('ccp', cpp.point_2_expression(x).subs(self.get_state()), 0.01, g=1)

            for obj in self.included_objects:
                visualize_obj(obj, self.visualizer, obj.pose, 'obstacles', self.obstacle_color)
            
            self.draw_tie_in()
            self.visualizer.render()

        #self.print_fn('\n'.join(['{}: {} -> {}'.format(n, c.lower.subs(self.current_subs), c.expression.subs(self.current_subs)) for n, c in self.avoidance_constraints.items()]))



        #self.qp_problem_builder.log_jacobian()
        #self.qp_problem_builder.log_lbA_ubA()
        return cmd

    def stop(self):
        self.simulator.kill()

    def draw_tie_in(self):
        pass
