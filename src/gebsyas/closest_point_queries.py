from giskardpy.symengine_wrappers import point3
from giskardpy.input_system import Point3Input, Vec3Input
from gebsyas.simulator import AABB, vec_add, vec_sub, vec3_to_list, frame_tuple_to_sym_frame, invert_transform

class ClosestPointQuery(object):
	def point_1_expression(self):
		raise (NotImplementedError)

	def point_2_expression(self):
		raise (NotImplementedError)

	def normal_expression(self):
		raise (NotImplementedError)

	# Returns dict with new values
	def get_update_dict(self, simulator, visualizer=None):
		raise (NotImplementedError)


class ClosestPointQuery_AnyN(ClosestPointQuery):
	def __init__(self, body_name, link_name, active_frame, n=3, aabb_border=0.2, filter=set()):
		self.body_name = body_name
		self.link_name = link_name
		self.active_frame = active_frame
		self.n = n
		self.aabb_border = [aabb_border]*3
		self.point1 = []
		self.point2 = []
		self.normal = []
		self.filter = filter.union({body_name})

		for x in range(self.n):
			self.point1.append(Point3Input('any_point_{}_{}_on_link1_{}'.format(body_name, link_name, self.n)))
			self.point2.append(Point3Input('any_point_{}_{}_on_link2_{}'.format(body_name, link_name, self.n)))
			self.normal.append(Vec3Input('any_point_{}_{}_normal_{}'.format(body_name, link_name, self.n)))

	def point_1_expression(self, index=0):
		return self.active_frame * self.point1[index].get_expression()

	def point_2_expression(self, index=0):
		return self.point2[index].get_expression()

	def normal_expression(self, index=0):
		return self.normal[index].get_expression()

	# Returns dict with new values
	def get_update_dict(self, simulator, visualizer=None):
		aabb = simulator.get_AABB(self.body_name, self.link_name)
		blownup_aabb = AABB(vec_sub(aabb.min, self.aabb_border), vec_add(aabb.max, self.aabb_border))

		# Get overlapping objects to minimize computation
		overlapping = simulator.get_overlapping(blownup_aabb, self.filter)
		closest = []
		for bodyId, linkId in overlapping:
			bodyAABB = simulator.get_AABB(bodyId, linkId)
			closest += simulator.get_closest_points(self.body_name, bodyId, self.link_name)

		closest = sorted(closest)
		link_frame_inv = frame_tuple_to_sym_frame(invert_transform(simulator.get_link_state(self.body_name, self.link_name).worldFrame))
		obs = {}
		for x in range(self.n):
			if x < len(closest):
				if visualizer != None:
					visualizer.draw_arrow('cpq', point3(*closest[x].posOnB), point3(*closest[x].posOnA))
				obs.update(self.point1[x].get_update_dict(*vec3_to_list(link_frame_inv * point3(*closest[x].posOnA))))
				obs.update(self.point2[x].get_update_dict(*vec3_to_list(closest[x].posOnB)))
				obs.update(self.normal[x].get_update_dict(*vec3_to_list(closest[x].normalOnB)))
			else:
				obs.update(self.point1[x].get_update_dict(0,0,0))
				obs.update(self.point2[x].get_update_dict(0,0,0))
				obs.update(self.normal[x].get_update_dict(0,0,1))
		return obs

class ClosestPointQuery_Any(ClosestPointQuery_AnyN):
	def __init__(self, body_name, link_name, active_frame, aabb_border=0.2):
		super(ClosestPointQuery_Any, self).__init__(body_name, link_name, active_frame, 1, aabb_border)


class ClosestPointQuery_Specific(ClosestPointQuery):
	def __init__(self, body_name, link_name, other_body, other_link):
		self.body_name  = body_name
		self.link_name  = link_name
		self.other_body = other_body
		self.other_link = other_link
		self.point1 = Point3Input('closest_on_{}_{}_to_{}_{}'.format(body_name, link_name, other_body, other_link))
		self.point2 = Point3Input('closest_on_{}_{}_to_{}_{}'.format(other_body, other_link, body_name, link_name))
		self.normal = Point3Input('normal_from_{}_{}_to_{}_{}'.format(other_body, other_link, body_name, link_name))


class ClosestPointQuery_Specific_SA(ClosestPointQuery_Specific):
	def __init__(self, body_name, link_name, other_body, other_link, active_frame):
		super(ClosestPointQuery_Specific_SA, self).__init__(body_name, link_name, other_body, other_link)
		self.active_frame = active_frame

	def point_1_expression(self):
		return self.active_frame * self.point1.get_expression()

	def point_2_expression(self):
		return self.point2.get_expression()

	def normal_expression(self):
		return self.normal.get_expression()

	# Returns dict with new values
	def get_update_dict(self, simulator, visualizer=None):
		closest = simulator.get_closest_points(self.body_name, self.other_body, self.link_name, self.other_link)
		link_frame_inv = frame_tuple_to_sym_frame(invert_transform(simulator.get_link_state(self.body_name, self.link_name).worldFrame))
		obs = {}
		if len(closest) > 0:
			if visualizer != None:
				visualizer.draw_arrow('cpq', point3(*closest[0].posOnB), point3(*closest[0].posOnA))
			obs.update(self.point1.get_update_dict(*vec3_to_list(link_frame_inv * point3(*closest[0].posOnA))))
			obs.update(self.point2.get_update_dict(*closest[0].posOnB))
			obs.update(self.normal.get_update_dict(*closest[0].normalOnB))
		# else:
		#     obs.update(self.normal.get_update_dict(*vec3_to_list()))
		return obs


class ClosestPointQuery_Specific_BA(ClosestPointQuery_Specific):
	def __init__(self, body_name, link_name, other_body, other_link, active_frame, other_active_frame):
		super(ClosestPointQuery_Specific_BA, self).__init__(body_name, link_name, other_body, other_link)
		self.active_frame = active_frame
		self.other_active_frame = other_active_frame

	def point_1_expression(self):
		return self.active_frame * self.point1.get_expression()

	def point_2_expression(self):
		return self.other_active_frame * self.point2.get_expression()

	def normal_expression(self):
		return self.normal.get_expression()

	# Returns dict with new values
	def get_update_dict(self, simulator, visualizer=None):
		closest = simulator.get_closest_points(self.body_name, self.other_body, self.link_name, self.other_link)
		link_frame_inv = frame_tuple_to_sym_frame(invert_transform(simulator.get_link_state(self.body_name, self.link_name).worldFrame))
		other_frame_inv = frame_tuple_to_sym_frame(invert_transform(simulator.get_link_state(self.other_body, self.other_link).worldFrame))

		obs = {}
		if len(closest) > 0:
			if visualizer != None:
				visualizer.draw_arrow('cpq', point3(*closest[0].posOnB), point3(*closest[0].posOnA))
			obs.update(self.point1.get_update_dict(*vec3_to_list(link_frame_inv * point3(*closest[0].posOnA))))
			obs.update(self.point2.get_update_dict(*vec3_to_list(other_frame_inv * point3(*closest[0].posOnB))))
			obs.update(self.normal.get_update_dict(*closest[0].normalOnB))
		else:
			obs.update(self.point1.get_update_dict(0,0,0))
			obs.update(self.point2.get_update_dict(0,0,0))
			obs.update(self.normal.get_update_dict(0,0,1))
		return obs