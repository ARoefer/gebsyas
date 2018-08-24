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
        return saturate(gripper.opening / (width + safety_margin)) * fake_heaviside(gripper.max_opening - width - safety_margin)

    @classmethod
    def __gen_open_gripper(cls, gripper, width, maximum=10, safety_margin=0.015):
        """Soft constraint which opens a gripper."""
        return SC(width + safety_margin - gripper.opening, maximum, 1, gripper.opening)

    @classmethod
    def __gen_reachability(cls, gripper, distance_terms, constraints_to_update={}):
        dist = distance_terms[0]
        if len(distance_terms) > 1:
            for t in distance_terms[1:]:
                dist = fake_Max(dist, t)

        reach_scale  = dist / gripper.reach 
        scaling_expr = if_greater_zero(1 - reach_scale, 1, 0)
        out = {'reachability': SC(-reach_scale, 0.8 - reach_scale, Min(1, reach_scale), reach_scale)}
        for sn, s in constraints_to_update.items():
            out[sn] = SC(s.lower, 
                         s.upper * scaling_expr, 
                         s.weight * scaling_expr, 
                         s.expression)
        return out


    @classmethod
    def point_grasp(cls, gripper, point):
        """Generates a set of constraints to grasp a point."""
        expr = norm(pos_of(gripper.pose) - point)
        r_dist = norm(sp.diag(1,1,0,1) * (gripper.pivot_position - point))
        return cls.__gen_reachability(gripper, [r_dist], {'point_grasp': SC(-expr, -expr, 1, expr)})
        #return {'point_grasp': SC(-expr, -expr, 1, expr)}

    @classmethod
    def sphere_grasp(cls, gripper, center, width):
        """Generates a set of constraints to grasp a ball."""
        pg = cls.point_grasp(gripper, center)
        pg_sc = pg['point_grasp']
        del pg['point_grasp']
        pg['sphere_grasp'] = SC(pg_sc.lower * cls.__gen_grasp_expr(gripper, width),
                               pg_sc.upper * cls.__gen_grasp_expr(gripper, width),
                               pg_sc.weight, pg_sc.expression)
         
        pg['open_gripper'] = cls.__gen_open_gripper(gripper, width)
        return pg

    # @classmethod
    # def partial_sphere_grasp(cls, gripper, center, up, width, angular_limit=1.56):
    #     return cls.sphere_grasp(gripper, center, width) * sp.Max(0, sp.cos(angular_limit) + dot(gripper.pose[:4, :1], up))

    @classmethod
    def axis_grasp(cls, gripper, point, dir):
        """Generates a set of constraints to grasp an infinite axis."""
        gripper_frame = gripper.pose
        gripper_z = z_of(gripper_frame)
        gripper_pos = pos_of(gripper_frame)
        p2g = gripper_pos - point
        #      Align z axis of gripper and axis; Align gripper position to axis
        pos_alignment_expr = norm(cross(dir, p2g))
        axis_cross = cross(gripper_z, dir)
        rot_expr   = norm(axis_cross)
        rot_alignment_expr = 1 - rot_expr
        reach_dist = norm(cross(dir, gripper.pivot_position - point))

        pa_sc = SC(-pos_alignment_expr * rot_alignment_expr, 
                   0.01-pos_alignment_expr * rot_alignment_expr, 
                   1, 
                   pos_alignment_expr)
        ra_sc = SC(1 - rot_alignment_expr, 
                   1 - rot_alignment_expr, 
                   1, 
                   rot_alignment_expr)
        out = cls.__gen_reachability(gripper, [reach_dist], {'axis_position_alignment': pa_sc})
        out['axis_rotation_alignment'] = ra_sc
        return out

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
        del agdict['axis_rotation_alignment']
        agdict.update({'axis_position_alignment': pa_sc, 
                       'line_rotation_alignment': ra_sc, 
                       'line_z_alignment': SC(-z_dist * ra_sc.expression, 
                                       (0.5*length-z_dist) * ra_sc.expression, 
                                       1, 
                                       z_dist)})
        return agdict

    @classmethod
    def column_grasp(cls, gripper, point, dir, width):
        """Generates a set of constraints to grasp an infinite cylinder."""
        updated = {n: SC(sc.lower * cls.__gen_grasp_expr(gripper, width), 
                         sc.upper * cls.__gen_grasp_expr(gripper, width), 
                         sc.weight, 
                         sc.expression) 
                    for n, sc in cls.axis_grasp(gripper, point, dir).items()}
        updated['open_gripper'] = cls.__gen_open_gripper(gripper, width)
        return updated

    @classmethod
    def rod_grasp(cls, gripper, center, dir, length, width):
        """Generates a set of constraints to grasp a cylinder."""
        updated = {n: SC(sc.lower * cls.__gen_grasp_expr(gripper, width), 
                         sc.upper * cls.__gen_grasp_expr(gripper, width), 
                         sc.weight, 
                         sc.expression) 
                    for n, sc in cls.line_grasp(gripper, center, dir, length).items()}
        updated['open_gripper'] = cls.__gen_open_gripper(gripper, width)
        return updated

    @classmethod
    def edge_grasp(cls, gripper, point, normal, dir):
        """Generates a set of constraints to grasp an edge of infinite length."""
        agdict = cls.axis_grasp(gripper, point, dir)
        normal_align = -norm(cross(x_of(gripper.pose), normal))
        ra_sc = agdict['axis_rotation_alignment']
        ra_expr = ra_sc.expression + normal_align
        agdict['axis_rotation_alignment'] = SC(2 - ra_expr, 
                                               2 - ra_expr, 
                                               ra_sc.weight, 
                                               ra_expr)
        return agdict

    @classmethod
    def rim_grasp(cls, gripper, point, normal, dir, width):
        """Generates a set of constraints to grasp an infinite rim."""
        updated = {n: SC(sc.lower * cls.__gen_grasp_expr(gripper, width), 
                         sc.upper * cls.__gen_grasp_expr(gripper, width), 
                         sc.weight, 
                         sc.expression) 
                    for n, sc in cls.edge_grasp(gripper, point, normal, dir).items()}
        updated['open_gripper'] = cls.__gen_open_gripper(gripper, width)
        return updated

    @classmethod
    def circular_edge_grasp(cls, gripper, center, axis, radius):
        """Generates a set of constraints to grasp an edge which is spun around an axis."""
        raise Exception('Needs to be update to use inequality constraints')

        gripper_pos = pos_of(gripper.pose)
        gripper_y = y_of(gripper.pose)
        gripper_x = x_of(gripper.pose)
        p2g = gripper_pos - center
        gripper_plane_dist = dot(axis, p2g)
        gripper_pos_in_plane = gripper_pos - gripper_plane_dist * axis
        radial_dist = norm(center - gripper_pos_in_plane)
        radius_dist = radius - radial_dist
        # careful about div by zero!
        radial_dir  = 1 / radial_dist * (gripper_pos_in_plane - gripper_pos_in_plane)
        rot_alignment = fake_Abs(dot(axis, gripper_y)) * norm(cross(radial_dir, gripper_x))
        return (1 - fake_Abs(radius_dist) - fake_Abs(gripper_plane_dist)) * alignment

    @classmethod
    def circular_rim_grasp(cls, gripper, center, axis, radius, width):
        """Generates a set of constraints to grasp an rim which is spun around an axis."""
        return cls.circular_edge_grasp(gripper, center, axis, radius) * saturate(gripper.opening / width) * fake_heaviside(gripper.max_opening - width)

    @classmethod
    def box_grasp(cls, gripper, frame, dx, dy, dz):
        """Generates a set of constraints to grasp a box."""
        gx = x_of(gripper.pose)
        gz = z_of(gripper.pose)

        bx = x_of(frame)
        by = y_of(frame)
        bz = z_of(frame)

        raxx = 1 - norm(cross(gx, bx))
        raxy = 1 - norm(cross(gx, by))
        raxz = 1 - norm(cross(gx, bz))

        razx = 1 - norm(cross(gz, bx))
        razy = 1 - norm(cross(gz, by))
        razz = 1 - norm(cross(gz, bz))

        x_grasp_depth = Max(dx * 0.5 - 0.04, 0) # TODO: Add finger length
        y_grasp_depth = Max(dy * 0.5 - 0.04, 0) # TODO: Add finger length
        z_grasp_depth = Max(dz * 0.5 - 0.04, 0) # TODO: Add finger length

        dx_sm = dx * 0.5 + 0.015 - gripper.height
        dy_sm = dy * 0.5 + 0.015 - gripper.height
        dz_sm = dz * 0.5 + 0.015 - gripper.height

        x_grasp_feasibility = fake_heaviside(gripper.max_opening - dx_sm)
        y_grasp_feasibility = fake_heaviside(gripper.max_opening - dy_sm)
        z_grasp_feasibility = fake_heaviside(gripper.max_opening - dz_sm)

        # Distances in box-coords
        b2g = pos_of(gripper.pose) - pos_of(frame)
        distx  = fake_Abs(dot(b2g, bx))
        disty  = fake_Abs(dot(b2g, by))
        distz  = fake_Abs(dot(b2g, bz))


        rot_goal_expr = fake_Max(raxx*razz*y_grasp_feasibility,
                            raxx*razy*z_grasp_feasibility,
                            raxy*razz*x_grasp_feasibility,
                            raxy*razx*z_grasp_feasibility,
                            raxz*razx*y_grasp_feasibility,
                            raxz*razy*x_grasp_feasibility)


        rot_goal_sc = SC(1 - rot_goal_expr, 1 - rot_goal_expr, 1, rot_goal_expr)
        x_coord_sc  = SC(x_grasp_depth * raxx - distx, Max(raxy*razz, raxz*razy) * -distx + (dx_sm - distx) * razx, 1, distx)
        y_coord_sc  = SC(y_grasp_depth * raxy - disty, Max(raxx*razz, raxz*razx) * -disty + (dy_sm - disty) * razy, 1, disty)
        z_coord_sc  = SC(z_grasp_depth * raxz - distz, Max(raxy*razx, raxx*razy) * -distz + (dz_sm - distz) * razz, 1, distz)

        return {'rot_goal': rot_goal_sc, 
                'x_coord' : x_coord_sc, 
                'y_coord' : y_coord_sc, 
                'z_coord' : z_coord_sc, 
                'open_gripper': cls.__gen_open_gripper(gripper, gripper.max_opening, safety_margin=0)}


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
        if DLRigidGMMObject.is_a(obj):
            mlgc = sorted(obj.gmm)[0]
            if abs(mlgc.cov[0,0]) > 0.02 and abs(mlgc.cov[1,1]) > 0.02 and abs(mlgc.cov[2,2]) > 0.02:
                return {'clearly_perceived': SC(1, 0, 1, 0)}
            else:
                pose = mlgc.pose

        if DLRigidObject.is_a(obj):
            pose = obj.pose

        if DLCompoundObject.is_a(obj):
            if type(obj.subObject) == list:
                raise Exception('Needs to be updated to use inequality constraints')
                return cls.combine_expressions_max(True, [cls.object_grasp(gripper, so) for so in obj.subObject])
            else:
                return cls.object_grasp(gripper, obj.subObject)
        else:
            if DLCube.is_a(obj):
                return cls.box_grasp(gripper, pose, obj.length, obj.width, obj.height)
            elif DLPartialSphere.is_a(obj):
                return cls.partial_sphere_grasp(gripper, pos_of(pose), z_of(pose), 2*obj.radius, obj.angle)
            elif DLCylinder.is_a(obj):
                return cls.rod_grasp(gripper, pos_of(pose), z_of(pose), obj.height, 2*obj.radius) #obj.height, 2*obj.radius
            elif DLSphere.is_a(obj):
                return cls.sphere_grasp(gripper, pos_of(pose), 2*obj.radius)
            else:
                context.log("Can't grasp {}, because it's not a rigid object.".format(str(obj)))

        return 0