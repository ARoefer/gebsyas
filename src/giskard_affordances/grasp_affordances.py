from giskardpy.sympy_wrappers import *
import symengine as sp


def ball_grasp_affordance(gripper, obj_input):
    return (1 - norm(pos_of(gripper.frame) - pos_of(obj_input.get_frame()))) * (gripper.opening / obj_input.get_dimensions()[0])

def ball_grasp_probabilistic(gripper, obj_input):
    return ball_grasp_affordance(gripper, obj_input) * obj_input.get_class_probability()

def cylinder_grasp_affordance(gripper, obj_input):
    frame = obj_input.get_frame()
    shape = obj_input.get_dimensions()
    cylinder_z = frame.col(2)
    cylinder_pos = pos_of(frame)

    gripper_x = gripper.frame.col(0)
    gripper_z = gripper.frame.col(2)
    gripper_pos = pos_of(gripper.frame)
    c_to_g = gripper_pos - cylinder_pos

    zz_align = sp.Abs(gripper_z.dot(cylinder_z))
    xz_align = gripper_x.dot(cylinder_z)
    dist_z = cylinder_z.dot(c_to_g)
    border_z = (shape[2] - gripper.height) * 0.5
    cap_dist_normalized_signed = dist_z / border_z
    cap_dist_normalized = sp.Abs(cap_dist_normalized_signed)

    cap_top_grasp = 1 - sp.Max(-xz_align * sp.Min(cap_dist_normalized_signed, 1), 0)
    cap_bottom_grasp = 1 - sp.Min(xz_align * sp.Max(cap_dist_normalized_signed, -1), 0)

    dist_z_center_normalized = sp.Max(1 - cap_dist_normalized, 0)
    dist_ax = sp.sqrt(frame.col(0).dot(c_to_g)**2 + frame.col(1).dot(c_to_g)**2)

    center_grasp = (1 - dist_z_center_normalized - dist_ax) * zz_align

    return sp.Max(center_grasp, cap_top_grasp, cap_bottom_grasp)

def cylinder_grasp_probabilistic(gripper, obj_input):
    return cylinder_grasp_affordance(gripper, obj_input) * obj_input.get_class_probability()