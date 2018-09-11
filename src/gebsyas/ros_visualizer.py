import rospy
from visualization_msgs.msg import Marker, MarkerArray
from gebsyas.utils import expr_to_rosmsg
from gebsyas.msg import FloatTable
from urdf_parser_py.urdf import Sphere, Cylinder, Box, Mesh

def del_marker(Id, namespace):
	out = Marker()
	out.ns = namespace
	out.header.stamp = rospy.Time.now()
	out.id = Id
	out.action = Marker.DELETE
	return out

def blank_marker(Id, namespace, r, g, b, a, frame):
	out = Marker()
	out.ns = namespace
	out.header.stamp = rospy.Time.now()
	out.header.frame_id = frame
	out.id = Id
	out.action  = Marker.ADD
	out.color.r = r
	out.color.g = g
	out.color.b = b
	out.color.a = a
	return out



class ROSVisualizer():
	def __init__(self, vis_topic, base_frame='base_link', plot_topic='plot'):
		self.ids     = {}
		self.lastIds = {}
		self.publisher = rospy.Publisher(vis_topic, MarkerArray, queue_size=1, tcp_nodelay=1)
		self.plot_publisher = rospy.Publisher(plot_topic, FloatTable, queue_size=1, tcp_nodelay=1)
		self.base_frame = base_frame
		self.current_msg = {}

	def begin_draw_cycle(self, *layers):
		if len(layers) == 0:
			layers = self.ids.keys()

		for layer in layers:
			if layer not in self.ids:
				self.ids[layer] = 0
				print('Added new layer {}'.format(layer))
			self.lastIds[layer] = self.ids[layer]
			self.ids[layer] = 0
			self.current_msg[layer] = MarkerArray()

	def consume_id(self, namespace):
		if not namespace in self.ids:
			self.ids[namespace] = 0
			self.lastIds[namespace] = 0
			self.current_msg[namespace] = MarkerArray()

		self.ids[namespace] += 1
		return self.ids[namespace] - 1


	def render(self, *layers):
		if len(layers) == 0:
			layers = self.ids.keys()

		for namespace in layers:
			Id = self.ids[namespace]
			self.current_msg[namespace].markers.extend([del_marker(x, namespace) for x in range(Id, self.lastIds[namespace])])

			self.publisher.publish(self.current_msg[namespace])

	def __resframe(self, frame):
		if frame == None:
			return self.base_frame
		return frame

	def draw_sphere(self, namespace, position, radius, r=1, g=0, b=0, a=1, frame=None):
		marker = blank_marker(self.consume_id(namespace), namespace, r, g, b, a, self.__resframe(frame))
		marker.type = Marker.SPHERE
		marker.pose.position = expr_to_rosmsg(position)
		marker.scale = expr_to_rosmsg([radius * 2] * 3)
		self.current_msg[namespace].markers.append(marker)

	def draw_cube(self, namespace, pose, scale, r=0, g=0, b=1, a=1, frame=None):
		self.draw_shape(namespace, pose, scale, Marker.CUBE, r, g, b, a, frame)

	def draw_cube_batch(self, namespace, pose, size, positions, r=1, g=0, b=0, a=1, frame=None):
		marker = blank_marker(self.consume_id(namespace), namespace, r, g, b, a, self.__resframe(frame))
		marker.type = Marker.CUBE_LIST
		marker.points = [expr_to_rosmsg(p) for p in positions]
		marker.scale.x = size
		marker.scale.y = size
		marker.scale.z = size
		self.current_msg[namespace].markers.append(marker)

	def draw_cylinder(self, namespace, pose, length, radius, r=0, g=0, b=1, a=1, frame=None):
		self.draw_shape(namespace, pose, (radius * 2, radius * 2, length), Marker.CYLINDER, r, g, b, a, frame)

	def draw_arrow(self, namespace, start, end, r=1, g=1, b=1, a=1, width=0.01, frame=None):
		marker = blank_marker(self.consume_id(namespace), namespace, r, g, b, a, self.__resframe(frame))
		marker.type = Marker.ARROW
		marker.scale.x = width
		marker.scale.y = 2 * width
		marker.points.extend([expr_to_rosmsg(start), expr_to_rosmsg(end)])
		self.current_msg[namespace].markers.append(marker)

	def draw_vector(self, namespace, position, vector, r=1, g=1, b=1, a=1, width=0.01, frame=None):
		self.draw_arrow(namespace, position, position + vector, r, g, b, a, width, frame)

	def draw_text(self, namespace, position, text, r=1, g=1, b=1, a=1, height=0.08, frame=None):
		marker = blank_marker(self.consume_id(namespace), namespace, r, g, b, a, self.__resframe(frame))
		marker.type = Marker.TEXT_VIEW_FACING
		marker.pose.position = expr_to_rosmsg(position)
		marker.scale.z = height
		marker.text = text
		self.current_msg[namespace].markers.append(marker)

	def draw_shape(self, namespace, pose, scale, shape, r=1, g=1, b=1, a=1, frame=None):
		marker = blank_marker(self.consume_id(namespace), namespace, r, g, b, a, self.__resframe(frame))
		marker.type = shape
		marker.pose = expr_to_rosmsg(pose)
		marker.scale = expr_to_rosmsg(scale)
		self.current_msg[namespace].markers.append(marker)

	def draw_mesh(self, namespace, pose, scale, resource, frame=None, r=0, g=0, b=0, a=0, use_mat=True):
		marker = blank_marker(self.consume_id(namespace), namespace, r, g, b, a, self.__resframe(frame))
		marker.type = Marker.MESH_RESOURCE
		marker.pose = expr_to_rosmsg(pose)
		marker.scale = expr_to_rosmsg(scale)
		marker.mesh_resource = resource
		marker.mesh_use_embedded_materials = use_mat
		self.current_msg[namespace].markers.append(marker)

	def draw_robot_pose(self, namespace, robot, joint_pos_dict, frame=None, tint=(1,1,1,1)):
		if robot._urdf_robot == None:
			return

		joint_pos_dict = {robot.joint_states_input.joint_map[j]: p for j, p in joint_pos_dict.items() if j in robot.joint_states_input.joint_map}

		urdf = robot._urdf_robot
		for link_name in robot.get_controllable_links(urdf.get_root()):
			if link_name in urdf.link_map and urdf.link_map[link_name].visual != None:
				visual = urdf.link_map[link_name].visual
				t_vis  = type(visual.geometry)
				pose   = robot.get_fk_expression(self.base_frame, link_name).subs(joint_pos_dict)
				color  = tint
				if visual.material != None and visual.material.color != None:
				 	color = (visual.material.color.rgba[0] * tint[0], visual.material.color.rgba[1] * tint[1], visual.material.color.rgba[2] * tint[2], visual.material.color.rgba[3] * tint[3])

				if len(pose.free_symbols) == 0:
					if t_vis is Sphere:
						self.draw_shape(namespace, pose, (visual.geometry.radius) * 3, Marker.SPHERE, color[0], color[1], color[2], color[3], frame)
					elif t_vis is Box:
						self.draw_cube(namespace, pose, visual.geometry.visual, color[0], color[1], color[2], color[3], frame)
					elif t_vis is Cylinder:
						self.draw_cylinder(namespace, pose, visual.geometry.length, visual.geometry.radius, color[0], color[1], color[2], color[3], frame)
					elif t_vis is Mesh:
						scale = (1,1,1)
						if visual.geometry.scale != None:
							scale = visual.geometry.scale
						self.draw_mesh(namespace, pose, scale, visual.geometry.filename, frame, color[0], color[1], color[2], color[3], False)

	def draw_robot_trajectory(self, namespace, robot, trajectory, frame=None, steps_per_sec=3, tint=(1,1,1,1)):
		if len(trajectory) == 0 or steps_per_sec == 0:
			return

		last_stamp = rospy.Time(0)
		interval = rospy.Duration(1.0 / steps_per_sec)

		points = 0

		for stamped_state in trajectory:
			if stamped_state.stamp - last_stamp >= interval:
				js_dict = {name: state.position for name, state in stamped_state.data.items()}
				self.draw_robot_pose(namespace, robot, js_dict, frame, tint)
				last_stamp = stamped_state.stamp
				points += 1
				#print('\n   '.join(['{:25}: {}'.format(name, pos) for name, pos in js_dict.items()]))

		#print('Rendered {} trajectory points. Original count: {}'.format(points, len(trajectory)))

	#def draw_plot_point(self, name, value):

