from giskardpy.controller import Controller
from giskardpy.qp_problem_builder import SoftConstraint
from giskard_affordances.grasp_affordances import *
from giskard_affordances.object_input import ProbabilisticObjectInput as POInput

import sympy as sp

class GraspAnyController(Controller):
    def __init__(self, robot, grippers, candidates):
        self.grippers = grippers
        self.candidates = candidates
        self.object_inputs = {}
        super(GraspAnyController, self).__init__(robot)

    def add_inputs(self, robot):
        for obj in self.candidates:
            obj_input = POInput(obj.id)
            self.update_observables(obj_input.get_update_dict(obj))
            self.object_inputs[obj.id] = obj_input

    def make_constraints(self, robot):
        super(GraspAnyController, self).make_constraints(robot)

        grasp_terms = []
        for gripper in self.grippers:
            for obj in self.candidates:
                if obj.semantic_class == 'cylinder':
                    grasp_terms.append(cylinder_grasp_probabilistic(gripper, self.object_inputs[obj.id]))
                elif obj.semantic_class == 'ball':
                    grasp_terms.append(ball_grasp_probabilistic(gripper, self.object_inputs[obj.id]))
                else:
                    print("Can't grasp object {} of class {}. Skipping it...".format(obj.id, obj.semantic_class))

        sc_expression = sp.Max(*grasp_terms)
        sc_ctrl = 1 - sc_expression

        self._soft_constraints['grasp any'] = SoftConstraint(sc_ctrl, sc_ctrl, 1, sc_expression)

    def update_object(self, p_object):
        if p_object.id in self.object_inputs:
            self.update_observables(self.object_inputs[p_object.id].get_update_dict(p_object))
