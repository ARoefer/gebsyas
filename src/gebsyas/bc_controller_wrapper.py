from giskardpy import print_wrapper
from giskardpy.symengine_controller import SymEngineController
from giskardpy.god_map import GodMap
from gebsyas.utils import res_pkg_path

class BCControllerWrapper(SymEngineController):
    def __init__(self, robot, print_fn=print_wrapper):
        self.path_to_functions = res_pkg_path('package://gebsyas/.controllers/')
        self.controlled_joints = []
        self.hard_constraints = {}
        self.joint_constraints = {}
        self.qp_problem_builder = None
        self.robot = robot
        self.current_subs = {}
        self.print_fn = print_fn


    def init(self, soft_constraints):
        free_symbols = set()
        for sc in soft_constraints.values():
            for f in sc:
                if hasattr(f, 'free_symbols'):
                    free_symbols = free_symbols.union(f.free_symbols)
        self.set_controlled_joints([j for j in self.robot.get_joint_names() if self.robot.joint_states_input.joint_map[j] in free_symbols])
        for jc in self.joint_constraints.values():
            for f in jc:
                if hasattr(f, 'free_symbols'):
                    free_symbols = free_symbols.union(f.free_symbols)
        for hc in self.hard_constraints.values():
            for f in hc:
                if hasattr(f, 'free_symbols'):
                    free_symbols = free_symbols.union(f.free_symbols)
        #print(free_symbols)
        super(BCControllerWrapper, self).init(soft_constraints, free_symbols, self.print_fn)

    def set_robot_js(self, js):
        for j, s in js.items():
            if j in self.robot.joint_states_input.joint_map:
                self.current_subs[self.robot.joint_states_input.joint_map[j]] = s.position

    def get_cmd(self, nWSR=None):
        return super(BCControllerWrapper, self).get_cmd({str(s): p for s, p in self.current_subs.items()}, nWSR)

    def stop(self):
        pass