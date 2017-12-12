from giskardpy.symengine_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint as SC
from giskard_affordances.utils import *
from giskard_affordances.dl_reasoning import *
import symengine as sp

class BasicGraspAffordances(object):


    @classmethod
    def point_grasp(cls, gripper, point):
        return 1 - norm(pos_of(gripper.frame) - point)

    @classmethod
    def sphere_grasp(cls, gripper, center, width):
        return cls.point_grasp(gripper, center) * saturate(gripper.opening / width) * sp.heaviside(gripper.max_opening - width)

    @classmethod
    def partial_sphere_grasp(cls, gripper, center, up, width, angular_limit=1.56):
        return cls.sphere_grasp(gripper, center, width) * sp.Max(0, sp.cos(angular_limit) + dot(gripper.frame[:4, :1], up))

    @classmethod
    def axis_grasp(cls, gripper, point, dir):
        gripper_frame = gripper.frame
        gripper_z = gripper_frame[:4, 2:3]
        gripper_pos = pos_of(gripper_frame)
        p2g = gripper_pos - point
        #      Align z axis of gripper and axis; Align gripper position to axis
        return abs(dot(dir, gripper_z)) * (1 - norm(cross(dir, p2g)))

    @classmethod
    def line_grasp(cls, gripper, center, dir, length):
        gripper_frame = gripper.frame
        gripper_pos = pos_of(gripper_frame)
        normed_z_dist = abs(dot(gripper_pos - center, dir)) / (length * 0.5)
        distance_scale = 1 - max(normed_z_dist - 1, 0)
        return cls.axis_grasp(gripper, center, dir) * distance_scale

    @classmethod
    def column_grasp(cls, gripper, point, dir, width):
        return cls.axis_grasp(gripper, point, dir) * saturate(gripper.opening / width) * sp.heaviside(gripper.max_opening - width)

    @classmethod
    def rod_grasp(cls, gripper, center, dir, length, width):
        return cls.line_grasp(gripper, center, dir, length) * saturate(gripper.opening / width) * sp.heaviside(gripper.max_opening - width)

    @classmethod
    def edge_grasp(cls, gripper, point, normal, dir):
        return cls.axis_grasp(gripper, point, dir) * -dot(gripper.frame[:4, :1], normal)

    @classmethod
    def rim_grasp(cls, gripper, point, normal, dir, width):
        return cls.edge_grasp(gripper, point, normal, dir) * saturate(gripper.opening / width) * sp.heaviside(gripper.max_opening - width)

    @classmethod
    def circular_edge_grasp(cls, gripper, center, axis, radius):
        gripper_pos = pos_of(gripper.frame)
        gripper_y = gripper.frame[:4, 1:2]
        gripper_x = gripper.frame[:4,  :1]
        p2g = gripper_pos - center
        gripper_plane_dist = dot(axis, p2g)
        gripper_pos_in_plane = gripper_pos - gripper_plane_dist * axis
        radial_dist = norm(center - gripper_pos_in_plane)
        radius_dist = radius - radial_dist
        # careful about div by zero!
        radial_dir  = 1 / radial_dist * (gripper_pos_in_plane - gripper_pos_in_plane)
        alignment = abs(dot(axis, gripper_y)) * -dot(radial_dir, gripper_x)
        return (1 - abs(radius_dist) - abs(gripper_plane_dist)) * alignment

    @classmethod
    def circular_rim_grasp(cls, gripper, center, axis, radius, width):
        return cls.circular_edge_grasp(gripper, center, axis, radius) * saturate(gripper.opening / width) * sp.heaviside(gripper.max_opening - width)

    @classmethod
    def box_grasp(cls, gripper, frame, dx, dy, dz):
        gx = x_col(gripper.frame)
        gz = z_col(gripper.frame)
        # print(str(gx))
        # print(str(gz))

        bx = x_col(frame)
        by = y_col(frame)
        bz = z_col(frame)
        # print(str(bx))
        # print(str(by))
        # print(str(bz))

        xx_dot = dot(gx, bx)
        zz_dot = dot(gz, bz)

        xy_dot = dot(gx, by)
        xz_dot = dot(gx, bz)
        zx_dot = dot(gz, bx)
        zy_dot = dot(gz, by)

        # Distances in box-coords
        b2g = pos_of(gripper.frame) - pos_of(frame)
        distx  = dot(b2g, bx)
        disty  = dot(b2g, by)
        distz  = dot(b2g, bz)

        # Normed distances
        distx_n = distx / (dx * 0.5)
        disty_n = disty / (dy * 0.5)
        distz_n = distz / (dz * 0.5)

        dx_sm = dx + 0.02
        dy_sm = dy + 0.02
        dz_sm = dz + 0.02

        inx = max(distx_n**2 - 1, 0)
        iny = max(disty_n**2 - 1, 0)
        inz = max(distz_n**2 - 1, 0)

        # Grasp along x
        #                        1             1                           1

        xz_align_and_open = abs(xx_dot) * abs(zz_dot) * saturate(gripper.opening / dy_sm) * sp.heaviside(gripper.max_opening - dy_sm)
        xy_align_and_open = abs(xx_dot) * abs(zy_dot) * saturate(gripper.opening / dz_sm) * sp.heaviside(gripper.max_opening - dz_sm)
        x_grasp_pos = tip_at_one(distx_n * -xx_dot) - max(xz_align_and_open - inz - abs(disty_n), xy_align_and_open - iny - abs(distz_n))

        # Grasp along y
        yz_align_and_open = abs(xy_dot) * abs(zz_dot) * saturate(gripper.opening / dx_sm) * sp.heaviside(gripper.max_opening - dx_sm)
        yx_align_and_open = abs(xy_dot) * abs(zx_dot) * saturate(gripper.opening / dz_sm) * sp.heaviside(gripper.max_opening - dz_sm)
        y_grasp_pos = (-xy_dot * disty_n) * max(inz * (-distx_n**2 + 1) * yz_align_and_open, inx * (-distz_n**2 + 1) * yx_align_and_open)

        # Grasp along z
        zx_align_and_open = abs(xz_dot) * abs(zx_dot) * saturate(gripper.opening / dy_sm) * sp.heaviside(gripper.max_opening - dy_sm)
        zy_align_and_open = abs(xz_dot) * abs(zx_dot) * saturate(gripper.opening / dx_sm) * sp.heaviside(gripper.max_opening - dx_sm)
        z_grasp_pos = (-xz_dot * distz_n) * max(inx * (-disty_n**2 + 1) * zx_align_and_open, iny * (-distx_n**2 + 1) * zy_align_and_open)

        return x_grasp_pos #- (inz + abs(disty_n)) - abs(distx_n) #* abs(xx_dot) * abs(zz_dot) * (inz - abs(disty_n))   # max(x_grasp_pos, y_grasp_pos, z_grasp_pos)

    @classmethod
    def combine_expressions_additive(cls, *args):
        out = 0
        for a in args:
            if a != 0:
                out = out + a
        return out

    @classmethod
    def combine_expressions_subtractive(cls, *args):
        out = 0
        for a in args:
            if a != 0:
                out = out - a
        return out

    @classmethod
    def combine_expressions_multiplicative(cls, *args):
        out = 1
        for a in args:
            if a == 0:
                return 0
            out = out * a
        return out

    @classmethod
    def combine_expressions_divisive(cls, *args):
        out = 1
        for a in args:
            if a == 0:
                return 0
            out = out / a
        return out

    @classmethod
    def combine_expressions_max(cls, filter_zero, args):
        if filter_zero:
            args = [a for a in args if a != 0]
        if len(args) == 1:
            return args[0]
        elif len(args) > 1:
            return max(*args)
        else:
            return 0

    @classmethod
    def combine_expressions_min(cls, filter_zero, args):
        if filter_zero:
            args = [a for a in args if a != 0]
        if len(args) == 1:
            return args[0]
        elif len(args) > 1:
            return min(*args)
        else:
            return 0

    @classmethod
    def capsule_grasp(cls, gripper, frame, height, diameter):
        capsule_z = frame[:4, 2:3]
        return cls.rod_grasp(gripper, pos_of(frame), capsule_z, 0.5*height, diameter)
        # #cls.combine_expressions_max(True, [cls.rod_grasp(gripper, pos_of(frame), capsule_z, 0.5*height, diameter),
        #                                    cls.partial_sphere_grasp(gripper, frame * point3(0,0,0.5*height), capsule_z, diameter, 0.5*pi),
        #                                    cls.partial_sphere_grasp(gripper, frame * point3(0,0,-0.5*height), -capsule_z, diameter, 0.5*pi)])

    @classmethod
    def object_grasp(cls, gripper, obj):
        if DLRigidObject.is_a(obj):
            if DLCompoundObject.is_a(obj):
                print('Grasp object was classified as compound object')
                if type(obj.subObject) == list:
                    return cls.combine_expressions_max(True, [cls.object_grasp(gripper, so) for so in obj.subObject])
                else:
                    return cls.object_grasp(gripper, obj.subObject)
            else:
                if DLCube.is_a(obj):
                    print('Grasp object was classified as cube')
                    return cls.box_grasp(gripper, obj.pose, obj.length, obj.width, obj.height)
                elif DLPartialSphere.is_a(obj):
                    print('Grasp object was classified as partial sphere')
                    return cls.partial_sphere_grasp(gripper, pos_of(obj.pose), z_col(obj.pose), 2*obj.radius, obj.angle)
                elif DLCylinder.is_a(obj):
                    print('Grasp object was classified as cylinder')
                    return cls.rod_grasp(gripper, pos_of(obj.pose), z_col(obj.pose), obj.height, 2*obj.radius) #obj.height, 2*obj.radius
                elif DLSphere.is_a(obj):
                    print('Grasp object was classified as sphere')
                    return cls.sphere_grasp(gripper, pos_of(obj.pose), 2*obj.radius)
        else:
            print("Can't grasp {}, because it's not a rigid object.".format(str(obj)))

        return 0