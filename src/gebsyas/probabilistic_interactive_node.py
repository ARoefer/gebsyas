import rospy
import pybullet as pb

from math import cos
from iai_bullet_sim.basic_simulator import BasicSimulator, vec3_to_list
from iai_bullet_sim.multibody import JointDriver
from iai_bullet_sim.full_state_interactive_node import FullStateInteractiveNode, pose_msg_to_frame
from iai_bullet_sim.utils import Frame
from gebsyas.data_structures import GaussianPoseComponent
from gebsyas.srv import *
from giskardpy.symengine_wrappers import eye, Matrix, frame3_quaternion, zeros, diag, pos_of, point3, norm, rot_of, x_of, dot

from visualization_msgs.msg import InteractiveMarkerFeedback as InteractiveMarkerFeedbackMsg


class ProbabilisticSimulator(BasicSimulator):
    def __init__(self, tick_rate=50, gravity=[0,0,-9.81]):
        super(ProbabilisticSimulator, self).__init__(tick_rate, gravity)

        self.gpcs = {}
        self.gmm_objects = set()
        self.initial_gpc_weights = {}
        self.std_variance = 1
        self.observing_body = None
        self.camera_link = None
        self.fov    = 1.5
        self.near   = 0.3
        self.far    = 10
        self.d_gain = 0.2
        self.h_gain = 0.4

    def post_update(self):
        if self.observing_body is not None:
            client_id = self.client_id()
            deltaT = self.time_step
            cf_tuple = self.observing_body.get_link_state(self.camera_link).worldFrame
            camera_frame = frame3_quaternion(cf_tuple.position.x, cf_tuple.position.y, cf_tuple.position.z, *cf_tuple.quaternion)
            cov_proj = rot_of(camera_frame)[:3, :3]
            ray_cast_camera = vec3_to_list(camera_frame * point3(self.near, 0, 0))
            inv_cov_proj = cov_proj.T

            for name, gmm in self.gpcs.items():
                body = self.bodies[name]
                gmm[0].pose = body.pose()
                for x in range(len(gmm)):
                    gc = gmm[x]
                    gc_pos = point3(*gc.pose.position)
                    gc_cov = gc.cov
                    c2o  = gc_pos - pos_of(camera_frame)
                    dist = norm(c2o) 
                    if dist < self.far and dist > self.near and dot(c2o, x_of(camera_frame)) > cos(self.fov * 0.5) * dist:
                        
                        hit_obj = pb.rayTest(ray_cast_camera, vec3_to_list(gc_pos), physicsClientId=client_id)[0][0]

                        #print('Object {} gmm {} hit result {}'.format(name, x, hit_obj))

                        if x == 0 and (hit_obj == body.bId() or hit_obj == -1):
                            s_h = min(1, max(0.01, 1 - self.h_gain / dist * deltaT))
                            s_d = min(1, max(0.01, 1 - self.d_gain / dist * deltaT))
                            S_pos = diag(s_d, s_h, s_h)
                            S_rot = diag(s_h, s_d, s_d)
                            new_pos_cov = cov_proj * S_pos * inv_cov_proj * gc_cov[:3, :3]
                            new_rot_cov = cov_proj * S_rot * inv_cov_proj * gc_cov[3:, 3:]
                            for x in range(3):
                                new_pos_cov[x,x] = max(0.0001, new_pos_cov[x, x])

                            gc_cov = new_pos_cov.col_join(zeros(3)).row_join(zeros(3).col_join(new_rot_cov))

                            #print(gc_cov)

                            weight_diff = (1.0 - gc.weight) * 0.5 * deltaT
                            gc.weight  += weight_diff
                            for y in range(1, len(gmm)):
                                gmm[y].weight -= gmm[y].weight * weight_diff

                            gc.cov = gc_cov
                        elif hit_obj == -1:
                            # Decrease weight of empty covariance
                            weight_diff = gc.weight - gc.weight * (max(1 - 0.2 * deltaT, 0))
                            gc.weight  -= weight_diff
                            for y in [z for z in range(len(gmm)) if z != x]:
                                gmm[y].weight += gmm[y].weight * weight_diff

        super(ProbabilisticSimulator, self).post_update()

    def reset(self):
        super(ProbabilisticSimulator, self).reset()
        for name, gmm in self.gpcs.items():
            for x in range(len(gmm)):
                gmm[x].weight = self.initial_gpc_weights[name][x]
                gmm[x].cov = eye(6) * self.std_variance


    def create_object(self, geom_type, extents=[1,1,1], radius=0.5, height=1, pos=[0,0,0], rot=[0,0,0,1], mass=1, color=None, name_override=None):
        body = super(ProbabilisticSimulator, self).create_object(geom_type, extents, radius, height, pos, rot, mass, color, name_override)
        bodyId = self.get_body_id(body.bId())

        if bodyId in self.gmm_objects:
            self.gpcs[bodyId] = [GaussianPoseComponent(1.0, body.pose(), eye(6) * self.std_variance)]
            self.initial_gpc_weights[bodyId] = [1.0]

        return body


    def load_urdf(self, urdf_path, pos=[0,0,0], rot=[0,0,0,1], joint_driver=JointDriver(), useFixedBase=0, name_override=None):
        body = super(ProbabilisticSimulator, self).load_urdf(urdf_path, pos, rot, joint_driver, useFixedBase, name_override)
        bodyId = self.get_body_id(body.bId())

        if bodyId in self.gmm_objects:
            self.gpcs[bodyId] = [GaussianPoseComponent(1.0, body.pose(), eye(6) * self.std_variance)]
            self.initial_gpc_weights[bodyId] = [1.0]

        return body

    def set_gmmc_pose(self, body, index, new_pose):
        bodyId = self.get_body_id(body.bId())
        self.gpcs[bodyId][index].pose = new_pose


    def add_gmmc(self, body, new_gc):
        if bodyId in self.gmm_objects:
            bodyId = self.get_body_id(body.bId())
            for x in range(len(self.gpcs[bodyId])):
                gc = self.gpcs[bodyId][x]
                gc.weight *= 1.0 - new_gc.weight
                self.initial_gpc_weights[bodyId][x] *= 1.0 - new_gc.weight
            self.gpcs[bodyId].append(new_gc)
            self.initial_gpc_weights[bodyId].append(new_gc.weight)

    def update_gmmc(self, body, idx, weight):
        if bodyId in self.gmm_objects:
            bodyId = self.get_body_id(body.bId())
            weights = self.initial_gpc_weights[body]
            if idx >= len(weights):
                raise Exception('Can not update gaussian component {} of {}. There are only {} components.'.format(idx, body, len(weights)))
            else:
                weight = max(0.0, min(req.weight, 1.0))
                for x in range(len(weights)):
                    weights[x] += (1.0 - weight) * weights[x]
                weights[idx] = weight
                gmm = self.gpcs[body]
                for x in range(len(gmm)):
                    gmm[x].weight = weights[x]


    def load_world(self, world_dict):
        super(ProbabilisticSimulator, self).load_world(world_dict)
        if 'objects' in world_dict:
            for od in world_dict['objects']:
                if 'gmm' in od and od['name'] in self.gmm_objects:
                    name = od['name']
                    if not type(od['gmm']) == list:
                        raise Exception('Field "gmm" in object dictionary needs to be a list.')
                    self.gpcs[name] = []
                    self.initial_gpc_weights[name] = []
                    for gc in od['gmm']:
                        i_pos = gc['pose']['position']
                        i_rot = gc['pose']['rotation']
                        self.gpcs[name].append(GaussianPoseComponent(gc['weight'], 
                                               Frame(i_pos, i_rot), 
                                               Matrix([gc['cov'][x * 6: x * 6 + 6] for x in range(6)])))
                        self.initial_gpc_weights[name].append(gc['weight'])
                else:
                    i_pos = od['initial_pose']['position']
                    i_rot = od['initial_pose']['rotation']
                    if od['name'] in self.gmm_objects:
                        self.gpcs[od['name']] = [GaussianPoseComponent(1, Frame(i_pos, i_rot), eye(6) * self.std_variance)]
                        self.initial_gpc_weights[od['name']] = [1.0]

    def save_world(self, use_current_state_as_init=False):
        out = super(ProbabilisticSimulator, self).save_world(use_current_state_as_init)
        for od in out['objects']:
            gmm = []

            if od['name'] in self.gmm_objects:
                gpcs = self.gpcs[od['name']]
                in_w = self.initial_gpc_weights[od['name']]
                
                for x in range(len(gpcs)):
                    pose   = {'position': list(gpcs[x].pose.position),
                              'rotation': list(gpcs[x].pose.quaternion)} if x > 0 else od['initial_pose'].copy()
                    weight = gpcs[x].weight if use_current_state_as_init else in_w[x]
                    cov    = gpcs[x].cov.tolist() if use_current_state_as_init else ([self.std_variance, 0, 0, 0, 0, 0, 0]*6)[:36]
                    gmm.append({'pose':   pose,
                                'cov':    cov,
                                'weight': weight})
                od['gmm'] = gmm
        return out
    
    def load_simulator(self, config_dict):
        if 'gmm_objects' in config_dict:
            self.gmm_objects = set(config_dict['gmm_objects'])

        super(ProbabilisticSimulator, self).load_simulator(config_dict)
        if 'camera_config' in config_dict:
            d_cf = config_dict['camera_config']
            self.observing_body = self.bodies[d_cf['body']]
            self.camera_link = d_cf['camera_link']
            self.fov    = d_cf['fov']
            self.near   = d_cf['near']
            self.far    = d_cf['far']
            self.d_gain = d_cf['d_gain']
            self.h_gain = d_cf['h_gain']

    def save_simulator(self, use_current_state_as_init):
        out = super(ProbabilisticSimulator, self).save_simulator(use_current_state_as_init)
        out['gmm_objects'] = list(self.gmm_objects)
        if self.observing_body != None:
            out['camera_config'] = {'body': self.get_body_id(self.observing_body.bId()),
                                    'camera_link': self.camera_link,
                                    'fov':    self.fov,
                                    'near':   self.near,
                                    'far':    self.far,
                                    'd_gain': self.d_gain,
                                    'h_gain': self.h_gain}
        return out


class ProbabilisticInteractiveNode(FullStateInteractiveNode):
        def __init__(self, server_name='iai_bullet_sim', simulator_class=ProbabilisticSimulator):
            super(ProbabilisticInteractiveNode, self).__init__(server_name, simulator_class)
            if not issubclass(simulator_class, ProbabilisticSimulator):
                raise Exception('Simulator class must be subclassed from ProbabilisticSimulator. Given class: {}'.format(str(simulator_class)))

            self.services.extend([
                rospy.Service('add_gmmc', AddGMMC, self.srv_add_gmmc),
                rospy.Service('update_gmmc', UpdateGMMC, self.srv_reweight_gmmc)
                ])

        def init(self, config_dict=None, mode='direct'):
            super(ProbabilisticInteractiveNode, self).init(config_dict, mode)

            for name, gmm in self.sim.gpcs.items():
                body = self.sim.bodies[name]
                for x in range(1, len(gmm)):
                    self.add_new_marker('{}_gc[{}]'.format(name, x), 
                                        body,
                                        self.process_gmm_update, 
                                        False, True, 
                                        frame='map', pose=gmm[x].pose)
            self.marker_server.applyChanges()


        def srv_add_gmmc(self, req):
            with self.lock:
                res = AddGMMCResponse()
                if req.object_id in self.sim.bodies and req.object_id in self.sim.gmm_objects:
                    body = self.sim.bodies[req.object_id]
                    weight = max(0.0, min(req.weight, 1.0))
                    self.sim.add_gmmc(body, GaussianPoseComponent(weight, body.pose(), eye(6) * self.sim.std_variance))

                    self.add_new_marker('{}_gc[{}]'.format(req.object_id, len(self.sim.gpcs[req.object_id]) -1),
                                        body,
                                        self.process_gmm_update, frame='map', pose=self.sim.gpcs[req.object_id][len(self.sim.gpcs[req.object_id])-1].pose)
                    res.success = True
                    return res
                else:
                    res.error_msg = '{} is not a known gmm object.'.format(req.object_id)
                res.success = False
                return res


        def srv_reweight_gmmc(self, req):
            with self.lock:
                res = UpdateGMMCResponse()
                res.success = False
                if req.object_id in self.sim.bodies and req.object_id in self.sim.gmm_objects:
                    try:
                        self.sim.update_gmmc(self.sim.bodies[req.object_id], req.index, req.weight)
                        res.success = True
                    except e:
                        res.error_msg = str(e)
                else:
                    res.error_msg = '{} is not a known gmm object.'.format(req.object_id)
                return res


        def process_gmm_update(self, feedback):
            if feedback.event_type == InteractiveMarkerFeedbackMsg.MOUSE_DOWN and feedback.marker_name != self._selected_object:
                self.select_marker(feedback.marker_name)
            
            if feedback.event_type == InteractiveMarkerFeedbackMsg.POSE_UPDATE:
                if feedback.marker_name != self._selected_object:
                    self.select_marker(feedback.marker_name)    
                body  = self.sim.bodies[self._selected_object.split('_gc[')[0]]
                index = int(self._selected_object.split('[')[1][:-1])
                intMarker, cb = self.markers[self._selected_object]
                intMarker.pose = feedback.pose

                self.sim.set_gmmc_pose(body, index, pose_msg_to_frame(feedback.pose))