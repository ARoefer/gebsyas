from giskardpy.input_system import ControllerInputArray, FrameInput, Vec3Input, ScalarInput

def vector_to_tuple(vec):
    return (vec.x, vec.y, vec.z)

def quaternion_to_tuple(quat):
    return (quat.x, quat.y, quat.z, quat.w)

def pose_to_tuple(pose):
    return quaternion_to_tuple(pose.orientation) + vector_to_tuple(pose.position)

class ObjectInput(object):
    def __init__(self, name):
        super(ObjectInput, self).__init__()

        self.frame = FrameInput('{}{}frame'.format(name, ControllerInputArray.separator))
        self.dimensions = Vec3Input('{}{}dimensions'.format(name, ControllerInputArray.separator))

    def get_update_dict(self, frame, dimensions):
        frame_dict = self.frame.get_update_dict(*frame)
        #frame_dict.update(self.dimensions.get_update_dict(*dimensions))
        return frame_dict

    def get_frame(self):
        return self.frame.get_expression()

    def get_dimensions(self):
        return self.dimensions #.get_expression()


class ProbabilisticObjectInput(ObjectInput):
    def __init__(self, name):
        super(ProbabilisticObjectInput, self).__init__(name)

        self.probability_class = ScalarInput('P_class')
        self.probability_pos = ScalarInput('P_trans')
        self.probability_rot   = ScalarInput('P_rot')

    def get_update_dict(self, p_object):
        out = super(ProbabilisticObjectInput, self).get_update_dict(pose_to_tuple(p_object.pose), vector_to_tuple(p_object.dimensions))
        out.update(self.probability_class.get_update_dict(p_object.probability_class))
        out.update(self.probability_pos.get_update_dict(p_object.probability_position))
        out.update(self.probability_rot.get_update_dict(p_object.probability_rotation))
        return out

    def get_class_probability(self):
        return self.probability_class.get_expression()

