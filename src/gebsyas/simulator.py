from gebsyas.dl_reasoning import DLCompoundObject, \
                                 DLRigidObject,    \
                                 DLRigidGMMObject, \
                                 DLIded,           \
                                 DLSphere,         \
                                 DLCylinder,       \
                                 DLCube
from gebsyas.utils import rot3_to_quat
from giskardpy.symengine_wrappers import *
from iai_bullet_sim.basic_simulator import BasicSimulator, AABB, vec_add, vec_sub, vec3_to_list, invert_transform


def frame_tuple_to_sym_frame(frame_tuple):
    return frame3_quaternion(frame_tuple.quaterion.x,
                             frame_tuple.quaterion.y,
                             frame_tuple.quaterion.z,
                             frame_tuple.quaterion.w,
                             point3(*frame_tuple.position))


class GebsyasSimulator(BasicSimulator):

    def add_object(self, dl_object):
        if DLCompoundObject.is_a(dl_object):
            raise Exception('Compund objects are not supported at this time.')
        elif DLRigidObject.is_a(dl_object):
          pose = dl_object.pose
        elif DLRigidGMMObject.is_a(dl_object):
          pose = sorted(dl_object.gmm)[-1].pose

        Id = None
        if DLIded.is_a(dl_object):
            Id = dl_object.id
            
            if DLCube.is_a(dl_object):
                return self.create_box([dl_object.length, dl_object.width, dl_object.height],
                                       vec3_to_list(pos_of(pose)),
                                       list(quaternion_from_matrix(pose)),
                                       dl_object.mass, name_override=Id)
            elif DLCylinder.is_a(dl_object):
                return self.create_cylinder(dl_object.radius, dl_object.height,
                                       vec3_to_list(pos_of(pose)),
                                       list(quaternion_from_matrix(pose)),
                                       dl_object.mass, name_override=Id)
            elif DLSphere.is_a(dl_object):
                return self.create_sphere(dl_object.radius,
                                       vec3_to_list(pos_of(pose)),
                                       list(quaternion_from_matrix(pose)),
                                       dl_object.mass, name_override=Id)
            else:
                Exception('Cannot generate Bullet-body for object which is not a sphere, box, or cylinder.\nObject: {}'.format(str(dl_object)))  
        raise Exception('Cannot generate Bullet-body for object which is not rigid. Object: {}'.format(str(dl_object)))