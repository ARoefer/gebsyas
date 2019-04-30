#!/usr/bin/env python
import rospy
from gebsyas.core.dl_types   import DLPhysicalThing
from gebsyas.core.data_state import DataState
from gebsyas.utils           import visualize_obj
from gebsyas.ros_visualizer  import ROSVisualizer
from gebsyas.test.demo_builder import compartment
from giskardpy.symengine_wrappers import frame3_rpy, point3

if __name__ == '__main__':
    rospy.init_node('debug_vis_data_state')

    vis = ROSVisualizer('ds_vis', 'map')

    vis.begin_draw_cycle('scene')
    
    ds = DataState()

    configs = []
    h_fac = {'left': 1, 'center': 0, 'right': -1}
    v_fac = {'bottom': -1, 'center': 0, 'top': 1}
    o_fac = {'horizontal': 0, 'vertical': 1}

    for h in h_fac.keys():
        for v in v_fac.keys():
            for o in o_fac.keys():
                configs.append((h, v, o))


    for x in range(len(configs)):
        h, v, o = configs[x]
        hf = h_fac[h]
        vf = v_fac[v]
        of = o_fac[o]
        compartment(x, frame3_rpy(0,0,0, point3(0, 0.45 * hf + of * 0.45 * len(h_fac), 0.8 + vf * 0.55)), 
                    0.4, 0.4, 0.5, 0.1, (h,v), o, ds)

    for Id, o in ds.dl_data_iterator(DLPhysicalThing):
        o = o.subs(ds.value_table)
        visualize_obj(o, vis, o.pose, 'scene')

    vis.render()

    rospy.sleep(0.3)
