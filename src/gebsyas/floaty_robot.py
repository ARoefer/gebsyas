import giskardpy.symengine_wrappers as spw
from giskardpy.symengine_robot import Robot, Joint, Camera
from giskardpy.input_system    import JointStatesInput
from giskardpy.qp_problem_builder import JointConstraint
from gebsyas.dl_reasoning import SymbolicData
from gebsyas.utils import JointState, symbol_formatter, deg2rad
from symengine import Symbol


class Floaty(Robot):
    def __init__(self, urdf='floaty.urdf'):
        super(Floaty, self).__init__(urdf, 0.6)


        use_perp = True

        self._joints['base_linear_joint'] = Joint(Symbol('base_linear_joint'),
                                                  0.4,
                                                  -1000,
                                                  1000,
                                                  'prismatic',
                                                  spw.translation3(Symbol('base_linear_joint'), 0,0))
        if use_perp:
          self._joints['base_perp_joint'] = Joint(Symbol('base_perp_joint'),
                                                    0.4,
                                                    -1000,
                                                    1000,
                                                    'prismatic',
                                                    spw.translation3(Symbol('base_perp_joint'), 0,0))
        
        self._joints['base_angular_joint'] = Joint(Symbol('base_angular_joint'),
                                                  0.2,
                                                  None,
                                                  None,
                                                  'continuous',
                                                  spw.rotation3_rpy(0, 0, Symbol('base_angular_joint')))


        self.set_joint_symbol_map(JointStatesInput.prefix_constructor(symbol_formatter, 
                                    self.get_joint_names() + 
                                    ['localization_x', 'localization_y', 'localization_z', 
                                     'localization_z_ang'], 
                                     '', 
                                     'position'))

        sj_lin  = self._joints['base_linear_joint'].symbol
        sj_perp = self._joints['base_perp_joint'].symbol if use_perp else 0
        sj_ang = self._joints['base_angular_joint'].symbol
        s_ang  = self.joint_states_input.joint_map['localization_z_ang']

        self.world_transform = spw.frame3_rpy(0, 0, s_ang + sj_ang, 
                                                [self.joint_states_input.joint_map['localization_x'] + spw.cos(s_ang) * sj_lin + spw.sin(s_ang) * sj_perp, 
                                                 self.joint_states_input.joint_map['localization_y'] + spw.sin(s_ang) * sj_lin + spw.cos(s_ang) * sj_perp, 
                                                 self.joint_states_input.joint_map['localization_z']])# * \
                                                #  .frame


        self.state = SymbolicData({jn: JointState(self._joints[jn].symbol, 0, 0) for jn in self.get_joint_names()}, self.do_js_resolve, ['joint_state'])

  
        self.joint_constraints['base_linear_joint']  = JointConstraint(0, 1, 0.001)
        
        if use_perp:
          self.joint_constraints['base_perp_joint']    = JointConstraint(-1, 1, 0.001)
        self.joint_constraints['base_angular_joint'] = JointConstraint(-0.4, 0.4, 0.001)

        self.camera = Camera(name='camera',
                             pose=self.get_fk_expression('map', 'camera_link'),
                             hfov=54.0 * deg2rad,
                             near=0.35,
                             far=6.0)

        # 'torso_lift_link', 'wrist_roll_link'
        # Link names mapped to safety margin and AABB blow up
        self.collision_avoidance_links = {'camera_link': (0.02, 0.1)}

        self.self_collision_avoidance_pairs = {}


    def get_fk_expression(self, root_link, tip_link):
        if (root_link, tip_link) in self.fks:
            return self.fks[root_link, tip_link]

        if root_link == 'map':
            fk = self.world_transform * super(Floaty, self).get_fk_expression(self._urdf_robot.get_root(), tip_link)
            self.fks[root_link, tip_link] = fk
            return fk
        return super(Floaty, self).get_fk_expression(root_link, tip_link)

    def do_camera_fk(self, joint_state):
        js = {self.joint_states_input.joint_map[name]: state.position for name, state in joint_state.items()}
        return Camera(name=self.camera.name,
                      pose=self.camera.pose.subs(js),
                      hfov=self.camera.hfov,
                      near=self.camera.near,
                      far=self.camera.far)

    def do_js_resolve(self, joint_state):
        return {jname: JointState(joint_state[jname].position, joint_state[jname].velocity, joint_state[jname].effort) for jname in self.state.data}