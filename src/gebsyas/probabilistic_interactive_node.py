import numpy as np

from iai_bullet_sim.basic_simulator import SimulatorPlugin, hsva_to_rgba
from iai_bullet_sim.full_state_interactive_node import FullStateInteractiveNode
from gebsyas.data_structures import GaussianPoseComponent
from giskardpy.symengine_wrappers import eye
from gop_gebsyas_msgs.msg import SearchObject as SearchObjectMsg
from gop_gebsyas_msgs.msg import ObjectPoseGaussianComponent as OPGCMsg


def create_search_object_message(body, name):
    msg = SearchObjectMsg()
    msg.id = body.bId()
    msg.name = name.split('.')[0]
    opgc = OPGCMsg()
    opgc.weight = 1.0
    msg.object_pose_gmm.append(opgc)
    return msg


class ProbabilisticInteractiveNode(FullStateInteractiveNode):
        def __init__(self, server_name='gebsyas_sim'):
            super(ProbabilisticInteractiveNode, self).__init__(server_name)

            self.gpcs = {}
            self.std_variance = 1
            self.observing_body = None
            self.camera_link = None
            self.fov    = 0
            self.near   = 0
            self.far    = 0
            self.d_gain = 0
            self.h_gain = 0
            self.message_templates = {}
            self.pub_so = None

        def init(self, config_dict=None, mode='direct'):
            super(ProbabilisticInteractiveNode, self).init(config_dict, mode)
            if 'observer' not in config_dict:
                raise Exception('Required attribute "observer" not found in configuration.')

            if 'camera_config' not in config_dict:
                raise Exception('Required attribute "camera_config" not found in configuration.')

            self.observing_body = self.sim.bodies[config_dict['camera_config']['body']]
            self.camera_link = config_dict['camera_config']['camera_link']
            self.fov    = config_dict['camera_config']['fov']
            self.near   = config_dict['camera_config']['near']
            self.far    = config_dict['camera_config']['far']
            self.d_gain = config_dict['camera_config']['d_gain']
            self.h_gain = config_dict['camera_config']['h_gain']
            self.pub_so = rospy.Publisher('{}/perceived_prob_objects'.format(config_dict['camera_config']['topic_prefix']), SearchObjectMsg, queue_size=1, tcp_nodelay=True)

            self.sim.register_plugin(Hook(post_cb=self.post_physics_update))

            for name, body in self.sim.bodies.items():
                self.gpcs[name] = [GaussianPoseComponent(1.0, body.pose(), eye(6) * self.std_variance)]
                self.message_templates[name] = create_search_object_message(body, name)

        def srv_add_urdf(self, req):
            res = super(ProbabilisticInteractiveNode, self).srv_add_urdf(req)
            if res.success:
                self.gpcs[res.object_id] = [GaussianPoseComponent(1.0, self.sim.bodies[res.object_id].pose(), eye(6) * self.std_variance)]
                self.message_templates[res.object_id] = create_search_object_message(self.sim.bodies[res.object_id], res.object_id)
            return res


        def srv_add_rigid_body(self, req):
            res = super(ProbabilisticInteractiveNode, self).srv_add_rigid_body(req)
            if res.success:
                self.gpcs[res.object_id] = [GaussianPoseComponent(1.0, self.sim.bodies[res.object_id].pose(), eye(6) * self.std_variance)]
                self.message_templates[res.object_id] = create_search_object_message(self.sim.bodies[res.object_id], res.object_id)
            return res


        def post_physics_update(self, simulator, deltaT):
            if self.is_running():
                cf_tuple = self.body.get_link_state(self.camera_link).worldFrame
                camera_frame = frame3_quaternion(cf_tuple.position.x, cf_tuple.position.y, cf_tuple.position.z, *cf_tuple.quaternion)
                cov_proj = rot_of(camera_frame)[:3, :3]
                inv_cov_proj = cov_proj.T

                for name, gmm in self.gpcs.items():
                    body = self.sim.bodies[name]
                    gmm[0].pose = body.pose()
                    msg = self.message_templates[name]
                    for x in range(len(gmm)):
                        gc = gmm[x]
                        gc_pos = point3(*gc.pose.position)
                        gc_cov = gc.cov
                        c2o  = gc_pos - pos_of(camera_frame)
                        dist = norm(c2o) 
                        if dist < self.far and dist > self.near and dot(c2o, x_of(camera_frame)) > cos(self.fov * 0.5) * dist:
                            
                            if x == 0:
                                s_h = min(1, max(0.01, 1 - self.camera_h_gain / dist * deltaT))
                                s_d = min(1, max(0.01, 1 - self.camera_d_gain / dist * deltaT))
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
                            else:
                                # Decrease weight of empty covariance
                                weight_diff = gc.weight - gc.weight * (max(1 - 0.2 * deltaT, 0))
                                gc.weight  -= weight_diff
                                for y in [z for z in range(len(gmm)) if z != x]:
                                    gmm[y].weight += gmm[y].weight * weight_diff


                            #print(gc_cov)

                        np_pos_cov = np.array(gc_cov[:3, :3].tolist(), dtype=float).reshape((3,3))
                        w, v = np.linalg.eig(np_pos_cov)
                        pos_eig = v * w


                        if np.isrealobj(pos_eig):
                            x_vec = vector3(*pos_eig[:, 0])
                            y_vec = vector3(*pos_eig[:, 1])
                            #z_vec = vector3(*pos_eig[:, 2])
                            x_vec *= 1.0 / norm(x_vec)
                            y_vec *= 1.0 / norm(y_vec)
                            #z_vec *= 1.0 / norm(z_vec)
                            M = x_vec.row_join(y_vec).row_join(cross(x_vec, y_vec)).row_join(gc_pos)

                            self.visualizer.draw_shape('cov', M, w.astype(float), 2, *hsva_to_rgba((1.0 - gc.weight) * 0.65, 1, 1, 0.7))

                        msg.object_pose_gmm[x].cov_pose.pose.position  = expr_to_rosmsg(gc.pose.position)
                        msg.object_pose_gmm[x].cov_pose.pose.orientation.x = gc.pose.quaternion[0]
                        msg.object_pose_gmm[x].cov_pose.pose.orientation.y = gc.pose.quaternion[1]
                        msg.object_pose_gmm[x].cov_pose.pose.orientation.z = gc.pose.quaternion[2]
                        msg.object_pose_gmm[x].cov_pose.pose.orientation.w = gc.pose.quaternion[3]
                        msg.object_pose_gmm[x].cov_pose.covariance = list(gc_cov)
                    msg.header.stamp = rospy.Time.now()
                    self.


class Hook(SimulatorPlugin):
    """Class to hook other classes into the simulator's update cycle, which are not actually plugins."""
    def __init__(self, pre_cb=None, post_cb=None, reset_cb=None):
        self.pre_cb  = pre_cb
        self.post_cb = post_cb

    def pre_physics_update(self, simulator, deltaT):
        if self.pre_cb is not None:
            self.pre_cb(simulator, deltaT)

    def post_physics_update(self, simulator, deltaT):
        if self.post_cb is not None:
            self.post_cb(simulator, deltaT)

    def reset(self, simulator):
        if self.reset_cb is not None:
            self.reset_cb(simulator)