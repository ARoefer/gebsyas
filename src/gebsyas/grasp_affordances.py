import symengine as sp

from giskardpy.symengine_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint as SC
from gebsyas.utils import *
from gebsyas.dl_reasoning import *

class BasicGraspAffordances(object):
    """
    @brief      Class containing generator functions which generate inequality constraints for grasping objects.
    """
    @classmethod
    def __gen_grasp_expr(cls, gripper, width, safety_margin=0.015):
        """Generates an expression models the opening of a gripper to a given width as a value between 0 and 1."""
        return saturate(gripper.opening / (width + safety_margin)) * sp.heaviside(gripper.max_opening - width - safety_margin)

    @classmethod
    def __gen_open_gripper(cls, gripper, width, maximum=10, safety_margin=0.015):
        """Soft constraint which opens a gripper."""
        return SC(width + safety_margin - gripper.opening, maximum, 1, gripper.opening)

    @classmethod
    def point_grasp(cls, gripper, point):
        """Generates a set of constraints to grasp a point."""
        expr = norm(pos_of(gripper.pose) - point)
        return {'point_grasp': SC(-expr, -expr, 1, expr)}

    @classmethod
    def sphere_grasp(cls, gripper, center, width):
        """Generates a set of constraints to grasp a ball."""
        pg_sc = cls.point_grasp(gripper, center).values()[0]
        pg_sc = SC(pg_sc.lower * cls.__gen_grasp_expr(gripper, width),
                   pg_sc.upper * cls.__gen_grasp_expr(gripper, width),
                   pg_sc.weight, pg_sc.expression)
        return {'sphere_grasp': pg_sc, 'open_gripper': cls.__gen_open_gripper(gripper, width)}

    # @classmethod
    # def partial_sphere_grasp(cls, gripper, center, up, width, angular_limit=1.56):
    #     return cls.sphere_grasp(gripper, center, width) * sp.Max(0, sp.cos(angular_limit) + dot(gripper.pose[:4, :1], up))

    @classmethod
    def axis_grasp(cls, gripper, point, dir):
        """Generates a set of constraints to grasp an infinite axis."""
        gripper_frame = gripper.pose
        gripper_z = gripper_frame[:4, 2:3]
        gripper_pos = pos_of(gripper_frame)
        p2g = gripper_pos - point
        #      Align z axis of gripper and axis; Align gripper position to axis
        pos_alignment_expr = norm(cross(dir, p2g))
        rot_alignment_expr = abs(dot(dir, gripper_z))
        pa_sc = SC(-pos_alignment_expr * rot_alignment_expr, 0.01-pos_alignment_expr * rot_alignment_expr, 1, pos_alignment_expr)
        ra_sc = SC(1 - rot_alignment_expr, 1 - rot_alignment_expr, 1, rot_alignment_expr)
        return {'axis_position_alignment': pa_sc, 'axis_rotation_alignment': ra_sc}

    @classmethod
    def line_grasp(cls, gripper, center, dir, length):
        """Generates a set of constraints to grasp a finite axis."""
        length = length - 2*gripper.height
        gripper_frame = gripper.pose
        gripper_pos = pos_of(gripper_frame)
        z_dist = abs(dot(gripper_pos - center, dir))
        agdict = cls.axis_grasp(gripper, center, dir)
        pa_sc = agdict['axis_position_alignment']
        ra_sc = agdict['axis_rotation_alignment']
        return {'axis_position_alignment': pa_sc, 'line_rotation_alignment': ra_sc, 'line_z_alignment': SC(-z_dist * ra_sc.expression, (0.5*length-z_dist) * ra_sc.expression, 1, z_dist)}

    @classmethod
    def column_grasp(cls, gripper, point, dir, width):
        """Generates a set of constraints to grasp an infinite cylinder."""
        updated = {n: SC(sc.lower * cls.__gen_grasp_expr(gripper, width), sc.upper * cls.__gen_grasp_expr(gripper, width), sc.weight, sc.expression) for n, sc in cls.axis_grasp(gripper, point, dir).items()}
        updated['open_gripper'] = cls.__gen_open_gripper(gripper, width)
        return updated

    @classmethod
    def rod_grasp(cls, gripper, center, dir, length, width):
        """Generates a set of constraints to grasp a cylinder."""
        updated = {n: SC(sc.lower * cls.__gen_grasp_expr(gripper, width), sc.upper * cls.__gen_grasp_expr(gripper, width), sc.weight, sc.expression) for n, sc in cls.line_grasp(gripper, center, dir, length).items()}
        updated['open_gripper'] = cls.__gen_open_gripper(gripper, width)
        return updated

    @classmethod
    def edge_grasp(cls, gripper, point, normal, dir):
        """Generates a set of constraints to grasp an edge of infinite length."""
        agdict = cls.axis_grasp(gripper, point, dir)
        ndot  = -dot(x_col(gripper.pose), normal)
        ra_sc = agdict['axis_rotation_alignment']
        ra_expr = ra_sc.expression + ndot
        agdict['axis_rotation_alignment'] = SC(2 - ra_expr, 2 - ra_expr, ra_sc.weight, ra_expr)
        return agdict

    @classmethod
    def rim_grasp(cls, gripper, point, normal, dir, width):
        """Generates a set of constraints to grasp an infinite rim."""
        updated = {n: SC(sc.lower * cls.__gen_grasp_expr(gripper, width), sc.upper * cls.__gen_grasp_expr(gripper, width), sc.weight, sc.expression) for n, sc in cls.edge_grasp(gripper, point, normal, dir).items()}
        updated['open_gripper'] = cls.__gen_open_gripper(gripper, width)
        return updated

    @classmethod
    def circular_edge_grasp(cls, gripper, center, axis, radius):
        """Generates a set of constraints to grasp an edge which is spun around an axis."""
        raise Exception('Needs to be update to use inequality constraints')

        gripper_pos = pos_of(gripper.pose)
        gripper_y = gripper.pose[:4, 1:2]
        gripper_x = gripper.pose[:4,  :1]
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
        """Generates a set of constraints to grasp an rim which is spun around an axis."""
        return cls.circular_edge_grasp(gripper, center, axis, radius) * saturate(gripper.opening / width) * sp.heaviside(gripper.max_opening - width)

    @classmethod
    def box_grasp(cls, gripper, frame, dx, dy, dz):
        """Generates a set of constraints to grasp a box."""
        gx = x_col(gripper.pose)
        gz = z_col(gripper.pose)

        bx = x_col(frame)
        by = y_col(frame)
        bz = z_col(frame)

        raxx = abs(dot(gx, bx))
        raxy = abs(dot(gx, by))
        raxz = abs(dot(gx, bz))

        razx = abs(dot(gz, bx))
        razy = abs(dot(gz, by))
        razz = abs(dot(gz, bz))

        x_grasp_depth = max(dx * 0.5 - 0.04, 0) # TODO: Add finger length
        y_grasp_depth = max(dy * 0.5 - 0.04, 0) # TODO: Add finger length
        z_grasp_depth = max(dz * 0.5 - 0.04, 0) # TODO: Add finger length

        dx_sm = dx * 0.5 + 0.015 - gripper.height
        dy_sm = dy * 0.5 + 0.015 - gripper.height
        dz_sm = dz * 0.5 + 0.015 - gripper.height

        x_grasp_feasibility = sp.heaviside(gripper.max_opening - dx_sm)
        y_grasp_feasibility = sp.heaviside(gripper.max_opening - dy_sm)
        z_grasp_feasibility = sp.heaviside(gripper.max_opening - dz_sm)

        # Distances in box-coords
        b2g = pos_of(gripper.pose) - pos_of(frame)
        distx  = abs(dot(b2g, bx))
        disty  = abs(dot(b2g, by))
        distz  = abs(dot(b2g, bz))


        rot_goal_expr = max(raxx*razz*y_grasp_feasibility,
                            raxx*razy*z_grasp_feasibility,
                            raxy*razz*x_grasp_feasibility,
                            raxy*razx*z_grasp_feasibility,
                            raxz*razx*y_grasp_feasibility,
                            raxz*razy*x_grasp_feasibility)


        rot_goal_sc = SC(1 - rot_goal_expr, 1 - rot_goal_expr, 1, rot_goal_expr)
        x_coord_sc = SC(x_grasp_depth * raxx - distx, max(raxy*razz, raxz*razy) * -distx + (dx_sm - distx) * razx, 1, distx)
        y_coord_sc = SC(y_grasp_depth * raxy - disty, max(raxx*razz, raxz*razx) * -disty + (dy_sm - disty) * razy, 1, disty)
        z_coord_sc = SC(z_grasp_depth * raxz - distz, max(raxy*razx, raxx*razy) * -distz + (dz_sm - distz) * razz, 1, distz)

        return {'rot_goal': rot_goal_sc, 'x_coord': x_coord_sc, 'y_coord': y_coord_sc, 'z_coord': z_coord_sc, 'open_gripper': cls.__gen_open_gripper(gripper, gripper.max_opening, safety_margin=0)}


    @classmethod
    def combine_expressions_additive(cls, *args):
        """Combines a list of expressions into a sum."""
        out = 0
        for a in args:
            if a != 0:
                out = out + a
        return out

    @classmethod
    def combine_expressions_subtractive(cls, *args):
        """Combines a list of expressions into a subtraction."""
        out = 0
        for a in args:
            if a != 0:
                out = out - a
        return out

    @classmethod
    def combine_expressions_multiplicative(cls, *args):
        """Combines a list of expressions into a product."""
        out = 1
        for a in args:
            if a == 0:
                return 0
            out = out * a
        return out

    @classmethod
    def combine_expressions_divisive(cls, *args):
        """Combines a list of expression using a division."""
        out = 1
        for a in args:
            if a == 0:
                return 0
            out = out / a
        return out

    @classmethod
    def combine_expressions_max(cls, filter_zero, args):
        """Combines a list of expressions into a max-expression."""
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
        """Combines a list of expressions into a min-expression."""
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
        """Generates a set of constraints to grasp a cylinder along its axis or over its caps."""
        capsule_z = frame[:4, 2:3]
        return cls.rod_grasp(gripper, pos_of(frame), capsule_z, 0.5*height, diameter)
        # #cls.combine_expressions_max(True, [cls.rod_grasp(gripper, pos_of(frame), capsule_z, 0.5*height, diameter),
        #                                    cls.partial_sphere_grasp(gripper, frame * point3(0,0,0.5*height), capsule_z, diameter, 0.5*pi),
        #                                    cls.partial_sphere_grasp(gripper, frame * point3(0,0,-0.5*height), -capsule_z, diameter, 0.5*pi)])

    @classmethod
    def object_grasp(cls, context, gripper, obj):
        """Generates a set of constraints to grasp a generic object."""
        if DLRigidObject.is_a(obj):
            if DLCompoundObject.is_a(obj):
                if type(obj.subObject) == list:
                    raise Exception('Needs to be updated to use inequality constraints')
                    return cls.combine_expressions_max(True, [cls.object_grasp(gripper, so) for so in obj.subObject])
                else:
                    return cls.object_grasp(gripper, obj.subObject)
            else:
                if DLCube.is_a(obj):
                    return cls.box_grasp(gripper, obj.pose, obj.length, obj.width, obj.height)
                elif DLPartialSphere.is_a(obj):
                    return cls.partial_sphere_grasp(gripper, pos_of(obj.pose), z_col(obj.pose), 2*obj.radius, obj.angle)
                elif DLCylinder.is_a(obj):
                    return cls.rod_grasp(gripper, pos_of(obj.pose), z_col(obj.pose), obj.height, 2*obj.radius) #obj.height, 2*obj.radius
                elif DLSphere.is_a(obj):
                    return cls.sphere_grasp(gripper, pos_of(obj.pose), 2*obj.radius)
        else:
            context.log("Can't grasp {}, because it's not a rigid object.".format(str(obj)))

        return 0