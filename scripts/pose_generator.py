#!/usr/bin/env python
import rospy
import math
import sys

from urdf_parser_py.urdf import URDF

from kineverse.gradients.diff_logic        import create_pos
from kineverse.gradients.gradient_math     import translation3, rotation3_axis_angle, vector3, spw
from kineverse.model.paths                 import Path
from kineverse.model.geometry_model        import GeometryModel
from kineverse.operations.basic_operations import CreateComplexObject
from kineverse.operations.urdf_operations  import load_urdf
from kineverse.urdf_fix                    import urdf_filler
from kineverse.utils                       import res_pkg_path
from gebsyas.view_pose_generator           import ViewPoseGenerator
from gebsyas.gaussian_observer             import Camera
from gebsyas.sdf_loader                    import load_world_from_xml

if __name__ == '__main__':
    rospy.init_node('view_pose_generator')

    km = GeometryModel()
    
    if len(sys.argv) >= 2:
        world_path = res_pkg_path(sys.argv[1])
        print('Loading world file {}'.format(world_path))
        load_world_from_xml(km, world_path)
    
    urdf_fetch = URDF.from_xml_file(res_pkg_path('package://gebsyas/robots/fetch_armless.urdf'))

    load_urdf(km, 'fetch', urdf_fetch, 'odom')

    print('\n'.join(km.list_constraints()))
    print('\n'.join(['{}:\n  {}'.format(k, v) for k, v in km.get_constraints_by_symbols({spw.Symbol('fetch__torso_lift_joint_v')}).items()]))

    sym_x = create_pos('localization_x')
    sym_y = create_pos('localization_y')
    sym_a = create_pos('localization_a')

    to_map_transform = translation3(sym_x, sym_y, 0) * rotation3_axis_angle(vector3(0,0,1), sym_a)
    connection_op    = CreateComplexObject(Path('fetch/links/base_link/pose'), to_map_transform)
    km.apply_operation_after(connection_op, 'set base_link transform', 'create fetch/base_link')
    km.clean_structure()
    km.dispatch_events()

    # Fetch the static collision model

    camera = Camera(km.get_data('fetch/links/head_camera_link/pose'), 54 * (math.pi / 180), 4.0/3.0)

    generator = ViewPoseGenerator(km, camera, sym_x, sym_y, sym_a, '/get_view_poses', [Path('fetch/links/base_collision_link')])

    while not rospy.is_shutdown():
        rospy.sleep(1000)
