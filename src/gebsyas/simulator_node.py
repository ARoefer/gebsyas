#!/usr/bin/env python
import rospy
import pybullet as pb
from time import time
from sensor_msgs.msg import JointState as JointStateMsg
from iai_naive_kinematics_sim.srv import SetJointState, SetJointStateResponse
from gebsyas.msg import ProbabilisticObject as PObject
from gebsyas.numeric_scene_state import visualize_obj
from gebsyas.ros_visualizer import ROSVisualizer
from gebsyas.simulator import BulletSimulator, BodyData, AABB
from gebsyas.utils import Blank, ros_msg_to_expr
from gebsyas.dl_reasoning import DLIded
from giskardpy.symengine_wrappers import translation3, norm, point3, vec3
import symengine as sp

class Watchdog(object):
	def __init__(self, timeout):
		self.last_tick = 0
		self.timeout = timeout

	def tick(self):
		self.last_tick = time()

	def barks(self):
		return time() - self.last_tick > self.timeout


class SimulatorNode(BulletSimulator):
	def __init__(self, vis_topic='~visual'):
		super(SimulatorNode, self).__init__(rospy.get_param('~sim_frequency', 50))
		self.init()
		self.visualizer = ROSVisualizer(vis_topic)
		robot_urdf = rospy.get_param('~robot_urdf')
		self.watchdog_period = rospy.get_param('~watchdog_period', 0.1)
		self.controlled_robot  = self.load_robot(robot_urdf)
		self.controlled_joints = set(rospy.get_param('~controlled_joints', []))
		self.simulated_joints  = set(rospy.get_param('~simulated_joints', []))
		robot_data = self.bodies[self.controlled_robot]
		self.next_cmd  = {}
		self.dl_bodies = {}
		for cj in self.controlled_joints:
			if cj not in robot_data.joints:
				raise Exception('Supposedly controlled joint "{}" is not part of the loaded robot.'.format(cj))
			self.watchdogs[cj] = Watchdog(self.watchdog_period)
			if not cj in self.simulated_joints:
				self.simulated_joints.append(cj)
				print('Added controlled joint "{}" to the list of simulated joints'.format(cj))
			self.next_cmd[cj] = 0.0

		robot_data.initial_joint_state = {sj: rospy.get_param('~start_config/{}'.format(sj), 0.0) for sj in self.simulated_joints}
		self.reset_body(self.controlled_robot)
		self.sim_indices = [robot_data.joints[sj].jointIndex for sj in self.simulated_joints]

		self.set_js_srv = rospy.Service("~set_joint_states", SetJointState, self.set_js_srv)
		self.js_pub = rospy.Publisher('~joint_states', JointStateMsg, queue_size=5)
		self.js_sub = rospy.Subscriber('~commands', JointStateMsg, callback=self.js_callback, queue_size=5)
		self.obj_sub = rospy.Subscriber('/perceived_objects', PObject, callback=self.receive_objects, queue_size=5)
		self.timer = rospy.Timer(rospy.Duration(1.0 / self.tick_rate), self.tick)
		self.aabb_border = vec3(*([0.2]*3))

	def start(self):
		self.timer.run()

	def pause(self):
		self.timer.shutdown()

	def kill(self):
		super(SimulatorNode, self).kill()
		self.js_sub.unregister()
		self.js_pub.unregister()
		self.timer.shutdown()


	def tick(self, timer_event):
		self.visualizer.begin_draw_cycle()
		for jname in self.next_cmd.keys():
			if jname in self.watchdogs and 	self.watchdogs[jname].barks():
				self.next_cmd[jname] = 0

		self.apply_joint_vel_cmds(self.controlled_robot, self.next_cmd)
		self.update()

		new_frames = self.get_all_body_frames()
		for Id, obj in self.dl_bodies.items():
			visualize_obj(obj, self.visualizer, new_frames[Id], 'bullet_objects', self.bodies[Id].color)

		# minAABB = point3(1000,1000,1000)
		# maxAABB = point3(-1000, -1000, -1000)
		# # 	minAABB = point3(min(minAABB[0], aabb.min[0]), min(minAABB[1], aabb.min[1]), min(minAABB[2], aabb.min[2]))
		# # 	maxAABB = point3(max(maxAABB[0], aabb.max[0]), max(maxAABB[1], aabb.max[1]), max(maxAABB[2], aabb.max[2]))
		# # self.visualizer.draw_cube('aabb', translation3((minAABB + maxAABB) * 0.5), maxAABB - minAABB, a=0.5)
		# filter = {self.controlled_robot}
		# for sj in self.simulated_joints:
		# 	robot_data = self.bodies[self.controlled_robot]
		# 	sl = robot_data.joints[sj].linkName
		# 	aabb = self.get_AABB(self.controlled_robot, sl)
		# 	blownup_aabb = AABB(aabb.min - self.aabb_border, aabb.max + self.aabb_border)
		# 	self.visualizer.draw_mesh('aabb', translation3((aabb.min + aabb.max) * 0.5), aabb.max - aabb.min, 'package://gebsyas/meshes/bounding_box.dae', r=1.0, a=1.0, use_mat=False)
		# 	self.visualizer.draw_mesh('aabb_bigger', translation3((blownup_aabb.min + blownup_aabb.max) * 0.5), blownup_aabb.max - blownup_aabb.min, 'package://gebsyas/meshes/bounding_box.dae', r=0.01, g=1.0, b=1.0, a=1.0, use_mat=False)
		# 	overlapping = self.get_overlapping(blownup_aabb, filter)
		# 	for bodyId, linkId in overlapping:
		# 		bodyAABB = self.get_AABB(bodyId, linkId)
		# 		self.visualizer.draw_mesh('overlapping', translation3((bodyAABB.min + bodyAABB.max) * 0.5), bodyAABB.max - bodyAABB.min, 'package://gebsyas/meshes/bounding_box.dae', r=1.0, g=1.0, b=0.0, a=1.0, use_mat=False)
		# 		for cp in self.get_closest_points(self.controlled_robot, bodyId, sl):
		# 			self.visualizer.draw_arrow('closest_points', cp.posOnB, cp.posOnA, g=0, b=0)


		for c in self.get_contacts(self.controlled_robot):
			self.visualizer.draw_sphere('contacts', point3(*c.posOnA), 0.2*c.normalForce, a=0.6)

		new_js = self.get_joint_state(self.controlled_robot, self.sim_indices)
		new_js_msg = JointStateMsg()
		new_js_msg.header.stamp = rospy.Time.now()
		for name, state in new_js.items():
			new_js_msg.name.append(name)
			new_js_msg.position.append(state.position)
			new_js_msg.velocity.append(state.velocity)
			new_js_msg.effort.append(state.appliedMotorTorque)
		self.js_pub.publish(new_js_msg)
		self.visualizer.render()

	def js_callback(self, js_msg):
		for x in range(len(js_msg.name)):
			jname = js_msg.name[x]
			if jname in self.controlled_joints:
				self.next_cmd[jname] = js_msg.velocity[x]
				self.watchdogs[jname].tick()

	def set_js_srv(self, req):
		robot_data = self.bodies[self.controlled_robot]
		new_initial = robot_data.initial_joint_state
		new_initial = {req.state.name[x]: req.state.position[x] for x in range(len(req.state.name)) if req.state.name[x] in self.simulated_joints}
		self.reset()
		res = SetJointStateResponse()
		res.success = True
		return res

	def receive_objects(self, msg):
		dl_object = ros_msg_to_expr(msg)
		if DLIded.is_a(dl_object):
			if 'table' in dl_object.id or 'floor' in dl_object.id:
				dl_object.mass = 0.0
				print('Detected static object {}'.format(dl_object.id))
		self.add_object(dl_object)

	def add_object(self, dl_object):
		Id = super(SimulatorNode, self).add_object(dl_object)
		self.dl_bodies[Id] = dl_object


if __name__ == '__main__':
	rospy.init_node('simulator')
	s = SimulatorNode()

	while not rospy.is_shutdown():
		pass
	s.kill()