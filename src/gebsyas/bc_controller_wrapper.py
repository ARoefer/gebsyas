from giskardpy import print_wrapper
from giskardpy.symengine_controller import SymEngineController
from giskardpy.qp_problem_builder import JointConstraint
from giskardpy.god_map import GodMap
from giskardpy.symengine_wrappers import Symbol
from gebsyas.utils import res_pkg_path

class BCControllerWrapper(SymEngineController):
    def __init__(self, robot, print_fn=print_wrapper, control_localization=False):
        self.path_to_functions = res_pkg_path('package://gebsyas/.controllers/')
        self.controlled_joints = []
        self.hard_constraints = {}
        self.joint_constraints = {}
        self.qp_problem_builder = None
        self.robot = robot
        self.current_subs = {}
        self.print_fn = print_fn
        self.control_localization = control_localization

    def init(self, soft_constraints, dynamic_base_weight=False):
        free_symbols = set()
        for sc in soft_constraints.values():
            for f in sc:
                if hasattr(f, 'free_symbols'):
                    free_symbols = free_symbols.union(f.free_symbols)
        self.set_controlled_joints(free_symbols, dynamic_base_weight)
        for jc in self.joint_constraints.values():
            for f in jc:
                if hasattr(f, 'free_symbols'):
                    free_symbols = free_symbols.union(f.free_symbols)
        for hc in self.hard_constraints.values():
            for f in hc:
                if hasattr(f, 'free_symbols'):
                    free_symbols = free_symbols.union(f.free_symbols)
        #print('  \n'.join([str(s) for s in free_symbols]))
        self.free_symbols = free_symbols
        super(BCControllerWrapper, self).init(soft_constraints, free_symbols, self.print_fn)

    
    def set_controlled_joints(self, free_symbols, dynamic_base_weight=False):
        filter = {'base_linear_joint', 'base_angular_joint'} if self.control_localization else set()

        super(BCControllerWrapper, self).set_controlled_joints([j for j in self.robot.get_joint_names() if self.robot.joint_states_input.joint_map[j] in free_symbols and j not in filter])
        rname = self.robot.get_name()
        if self.control_localization:
            s_lx  = self.robot.joint_states_input.joint_map['localization_x']
            s_ly  = self.robot.joint_states_input.joint_map['localization_y']
            s_laz = self.robot.joint_states_input.joint_map['localization_z_ang']
            if s_lx in free_symbols:
                self.joint_constraints[(rname, 'localization_x')] = JointConstraint(-100, 100, 0.001)
                self.controlled_joints.append('localization_x')
                self.controlled_joint_symbols.append(s_lx)
            if s_ly in free_symbols:
                self.joint_constraints[(rname, 'localization_y')] = JointConstraint(-100, 100, 0.001)
                self.controlled_joints.append('localization_y')
                self.controlled_joint_symbols.append(s_ly)
            if s_laz in free_symbols:
                self.joint_constraints[(rname, 'localization_z_ang')] = JointConstraint(-100, 100, 0.001)
                self.controlled_joints.append('localization_z_ang')
                self.controlled_joint_symbols.append(s_laz)
        elif dynamic_base_weight:
            self.s_base_weight = Symbol('base_weight_control')
            if 'base_angular_joint' in self.robot.joint_constraints:
                oc = self.robot.joint_constraints['base_angular_joint']
                self.joint_constraints[(rname, 'base_angular_joint')] = JointConstraint(oc.lower, oc.upper, self.s_base_weight * oc.weight)
            if 'base_linear_joint' in self.robot.joint_constraints:
                oc = self.robot.joint_constraints['base_linear_joint']
                self.joint_constraints[(rname, 'base_linear_joint')] = JointConstraint(oc.lower, oc.upper, self.s_base_weight * oc.weight)
            self.current_subs[self.s_base_weight] = 1.0

                

    def set_robot_js(self, js):
        for j, s in js.items():
            if j in self.robot.joint_states_input.joint_map:
                self.current_subs[self.robot.joint_states_input.joint_map[j]] = s.position

    def get_cmd(self, nWSR=None):
        return super(BCControllerWrapper, self).get_cmd({str(s): p for s, p in self.current_subs.items()}, nWSR)

    def stop(self):
        pass