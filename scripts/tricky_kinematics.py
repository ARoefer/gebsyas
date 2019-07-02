#!/usr/bin/env python
import rospy

from giskardpy.symengine_wrappers      import *

from gebsyas.kinematics.min_qp_builder import HardConstraint, SoftConstraint, ControlledValue
from gebsyas.kinematics.min_qp_builder import MinimalQPBuilder  as MQPB
from gebsyas.kinematics.min_qp_builder import TypedQPBuilder    as TQPB
from gebsyas.kinematics.gradients      import GradientContainer as GC
from gebsyas.kinematics.gradients      import GradientMatrix    as GM
from gebsyas.kinematics.gradients      import *
from gebsyas.kinematics.integrator     import CommandIntegrator, DT_SYM
from gebsyas.plotting                  import split_recorders, draw_recorders
from gebsyas.ros_visualizer            import ROSVisualizer


if __name__ == '__main__':
    rospy.init_node('tricky_kinematics')

    vis = ROSVisualizer('tricky_kinematics_vis', 'map')

    # Baseline
    j1, j2, j3 = sp.symbols(' '.join(['joint_{}_p'.format(x + 1) for x in range(3)]))
    eef = rotation3_axis_angle(unitZ, j1) * frame3_axis_angle(unitZ, j2, [1, 0, 0]) * frame3_axis_angle(unitZ, j3, [0, 1, 0]) * point3(1, -1, 0)
    base_g  = point3(0, 2, 0)
    base_d  = norm(eef - base_g)
    base_sc = SoftConstraint(-base_d, -base_d, 1, base_d)
    base_controlled = {str(j): ControlledValue(-0.6, 0.6, get_diff_symbol(j), 0.01) for j in [j1, j2, j3]}
    base_integrator = CommandIntegrator(TQPB({}, {'goal': base_sc}, base_controlled), recorded_terms={'distance': base_d})

    # Speedo case
    car_v     = sp.symbols('car_v')
    spe_alpha = car_v * 0.01
    spe_d     = 1.2 - spe_alpha
    spe_g     = SoftConstraint(spe_d, spe_d, 1, GC(spe_alpha))
    spe_controlled = {'car_velocity': ControlledValue(-100, 100, get_diff_symbol(car_v), 0.00001)}
    spe_integrator = CommandIntegrator(TQPB({}, {'goal': spe_g}, spe_controlled),
                            recorded_terms={'alpha': spe_alpha})

    # Roomba case
    roo_x, roo_y, roo_v, roo_r, roo_rv = sp.symbols(' '.join(['roo_{}'.format(x) for x in 'x_p y_p l_v r_p r_v'.split(' ')]))
    p = frame3_axis_angle(unitZ, GC(roo_r, {roo_rv: 1}), [GC(roo_x, {roo_v : cos(roo_r)}), 
                                                          GC(roo_y, {roo_v : sin(roo_r)}),0]) * point3(0.1, 0, 0)
    # cos(roo_r) * roo_v
    goal_d = norm(point3(2, -4, 0) - p)
    roo_g  = SoftConstraint(-goal_d, -goal_d, 1, goal_d)
    roo_controlled = {str(c.symbol) : c for c in [ControlledValue(-1, 1, roo_v,  0.01),
                                                  ControlledValue(-0.6, 0.6, roo_rv, 0.01)]}
    roo_integrator = CommandIntegrator(TQPB({}, {'goal': roo_g}, roo_controlled),
                                       {roo_x: roo_x + cos(roo_r + roo_rv * DT_SYM) * roo_v * DT_SYM,
                                        roo_y: roo_y + sin(roo_r + roo_rv * DT_SYM) * roo_v * DT_SYM,
                                        roo_r: roo_r + roo_rv * DT_SYM,
                                        roo_v: 0 * DT_SYM,
                                        roo_rv: 0 * DT_SYM},
                                        recorded_terms={'distance': goal_d.expr})

    # RC case
    car_x   = sp.symbols('car_x_p')
    stick_a = sp.symbols('stick_a_p')
    stick_a_limit = HardConstraint(-1 - stick_a, 1 - stick_a, stick_a)

    car_p = GC(car_x, {stick_a: 1})
    car_d = 2 - car_p
    car_e_i = Symbol('error_i')
    car_e_d = Symbol('error_d')
    car_u = 1 * car_d.expr #+ 0 * car_e_i + 1.5 * car_e_d
    car_g = SoftConstraint(car_u, car_u, 1, car_p)
    car_controlled = {'stick_alpha': ControlledValue(-1, 1, stick_a, 0.01)}

    car_integrator = CommandIntegrator(TQPB({'stick_limit': stick_a_limit}, 
                                            {'goal'       : car_g},
                                            car_controlled),
                                       {car_x: car_x + stick_a * DT_SYM,
                                        stick_a: stick_a,
                                        car_e_i: car_d * DT_SYM, # BROKEN
                                        car_e_d: -stick_a},
                                        recorded_terms={'distance': car_d.expr,
                                                        'u': car_u})

    # Dependant Kinematics 
    door_a   = sp.symbols('door_a_p')
    handle_a = sp.symbols('handle_a_p')
    door_a_limit = HardConstraint(-door_a, 2 - door_a, door_a)    
    
    door_opening = sin(door_a)
    door_l_u = 0.7 - door_opening
    door_u_u = None
    door_ul  = 0.4 - 0.4 * sigmoid(0.8 - handle_a) * sigmoid(0.1 - door_a)
    door_g       = SoftConstraint(     door_l_u, door_u_u, 1, door_a)
    door_release = SoftConstraint(0.4 - door_ul,     None, 1, handle_a)
    door_controlled = {'door_alpha'  : ControlledValue(-0.4, door_ul, door_a, 0.01),
                       'handle_alpha': ControlledValue(-0.4, 0.4, handle_a, 0.01)}
    door_integrator = CommandIntegrator(TQPB({'door_limit'  : door_a_limit},
                                             {'door_release': door_release,
                                              'goal'        : door_g},
                                             door_controlled),
                                             recorded_terms={'opening': door_opening})

    base_integrator.restart('3Dof Arm Example')
    spe_integrator.restart('Speedo Example')
    roo_integrator.restart('Roomba Example')
    car_integrator.restart('RC-Car Example')
    door_integrator.restart('Door Example')

    base_integrator.run()
    vis.begin_draw_cycle('trajectory', 'goal')
    points = [eef.subs({j1: v1, j2: v2, j3: v3}) for v1, v2, v3 in zip(base_integrator.recorder.data[str(j1)],
                                                     base_integrator.recorder.data[str(j2)],
                                                     base_integrator.recorder.data[str(j3)])]
    vis.draw_strip('trajectory', sp.eye(4), 0.01, points)
    vis.draw_sphere('goal', base_g, 0.025, r=0, g=1)
    vis.render()
    #print(base_integrator.qp_builder.last_matrix_str())
    spe_integrator.run()
    roo_integrator.run(0.1)
    car_integrator.run(0.2)
    door_integrator.run(0.1)

    draw_recorders(split_recorders([base_integrator.recorder,
                    base_integrator.sym_recorder,
                    spe_integrator.recorder,
                    spe_integrator.sym_recorder, 
                    roo_integrator.recorder,
                    roo_integrator.sym_recorder, 
                    car_integrator.recorder,
                    car_integrator.sym_recorder #,door_integrator.sym_recorder
                    #door_integrator.recorder,
                    ]), 4.0/9.0, 4).savefig('tricky_kinematics.png')
    rospy.sleep(1)
