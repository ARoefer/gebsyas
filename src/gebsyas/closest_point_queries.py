from giskardpy.symengine_wrappers import point3, norm, dot, translation3
from giskardpy.input_system import Point3Input, Vector3Input
from gebsyas.simulator import frame_tuple_to_sym_frame, invert_transform
from iai_bullet_sim.basic_simulator import transform_point, vec_add, vec_sub, vec3_to_list, vec_scale
from iai_bullet_sim.utils import Frame, AABB
from gebsyas.utils import symbol_formatter
from math import sqrt
from giskardpy.qp_problem_builder import SoftConstraint as SC

class ClosestPointQuery(object):
	"""
	@brief      Superclass for closest point queries.
	All closest point queries consist of a point relative to a link and a point relative to the world.
	Additionally, every query provides a normal direction from point the world-point to the link-point.
	"""
	def point_1_expression(self):
		"""Point relative to robot link."""
		raise (NotImplementedError)

	def point_2_expression(self):
		"""Point relative to world."""
		raise (NotImplementedError)

	def normal_expression(self):
		"""Normal from world- to body-point."""
		raise (NotImplementedError)

	def update_subs_dict(self, simulator, subs, visualizer=None):
		raise (NotImplementedError)

	def reset_subs_dict(self, subs):
		raise (NotImplementedError)


class ClosestPointQuery_AnyN(ClosestPointQuery):
	"""
	@brief      Point query which returns the closest N points.
	"""
	def __init__(self, bullet_body, link_name, active_frame, margin, n=3, aabb_border=0.2, filter=set()):
		"""Constructor.
		Requires the name of the body that the link belongs to,
		the name of the link and
		an expression for the link's frame.
		Additionally, n can be supplied, as well as the margin of the search box and a set of bodies' names to ignore in the queries.
		"""
		self.body = bullet_body
		self.link_name = link_name
		self.active_frame = active_frame
		self.n = n
		self.aabb_border = [aabb_border]*3
		self.search_dist = aabb_border
		self.point1 = []
		self.point2 = []
		self.normal = []
		self.filter = filter.union({self.body})
		self.margin = margin

		for x in range(self.n):
			self.point1.append(Point3Input.prefix(symbol_formatter, 'any_point_{}_{}_on_link_{}'.format(self.body.bId(), link_name, x)))
			self.point2.append(Point3Input.prefix(symbol_formatter, 'any_point_{}_{}_in_world_{}'.format(self.body.bId(), link_name, x)))
			self.normal.append(Vector3Input.prefix(symbol_formatter, 'any_point_{}_{}_normal_{}'.format(self.body.bId(), link_name, x)))

	def generate_constraints(self):
		out = {}
		for x in range(self.n):
			dist = dot((self.active_frame * self.point1[x].get_expression()) - self.point2[x].get_expression(), self.normal[x].get_expression())
			#dist = norm((self.active_frame * self.point1[x].get_expression()) - self.point2[x].get_expression())
			out['closest_any_{}_{}_{}'.format(self.body.bId(), self.link_name, x)] = SC(
					self.margin - dist, 
					1000,
					100,
					dist)

		return out


	# Returns dict with new values
	def update_subs_dict(self, simulator, subs, visualizer=None):
		aabb = self.body.get_AABB(self.link_name)
		blownup_aabb = AABB(vec_sub(aabb.min, self.aabb_border), vec_add(aabb.max, self.aabb_border))

		if visualizer != None:
			visualizer.draw_mesh('aabb', 
								 Frame(vec_scale(vec_add(blownup_aabb.min, blownup_aabb.max), 0.5), (0,0,0,1)),
								 vec_sub(blownup_aabb.max, blownup_aabb.min),
								 'package://gebsyas/meshes/bounding_box.dae')

		# Get overlapping objects to minimize computation
		overlapping = simulator.get_overlapping(blownup_aabb, self.filter)
		closest = []
		for body, linkId in overlapping:
			closest += self.body.get_closest_points(body, self.link_name, linkId, self.search_dist)

		closest = sorted(closest)
		link_frame_inv = invert_transform(self.body.get_link_state(self.link_name).worldFrame)

		for x in range(self.n):
			if x < len(closest):
				#raise Exception('FIXME! A and B are reversed for external objects.')
				onA = transform_point(link_frame_inv, closest[x].posOnA)
				if visualizer != None:
					visualizer.draw_arrow('cpq', closest[x].posOnB, closest[x].posOnA, g=0, b=0)
				subs[self.point1[x].x] = onA[0]
				subs[self.point2[x].x] = closest[x].posOnB[0]
				subs[self.normal[x].x] = closest[x].normalOnB[0]
				subs[self.point1[x].y] = onA[1]
				subs[self.point2[x].y] = closest[x].posOnB[1]
				subs[self.normal[x].y] = closest[x].normalOnB[1]
				subs[self.point1[x].z] = onA[2]
				subs[self.point2[x].z] = closest[x].posOnB[2]
				subs[self.normal[x].z] = closest[x].normalOnB[2]
			else:
				subs[self.point1[x].x] = 0
				subs[self.point2[x].x] = 0
				subs[self.normal[x].x] = 0
				subs[self.point1[x].y] = 0
				subs[self.point2[x].y] = 0
				subs[self.normal[x].y] = 0
				subs[self.point1[x].z] = 0
				subs[self.point2[x].z] = -10000
				subs[self.normal[x].z] = 1

	def reset_subs_dict(self, subs):
		for x in range(self.n):
			subs[self.point1[x].x] = 0
			subs[self.point2[x].x] = 0
			subs[self.normal[x].x] = 0
			subs[self.point1[x].y] = 0
			subs[self.point2[x].y] = 0
			subs[self.normal[x].y] = 0
			subs[self.point1[x].z] = 0
			subs[self.point2[x].z] = -10000
			subs[self.normal[x].z] = 1		


class ClosestPointQuery_Any(ClosestPointQuery_AnyN):
	"""
	@brief      Override of ClosestPointQuery_AnyN, where n is a constant 1.
	"""
	def __init__(self, body_name, link_name, active_frame, aabb_border=0.2):
		super(ClosestPointQuery_Any, self).__init__(body_name, link_name, active_frame, 1, aabb_border)


class ClosestPointQuery_Specific(ClosestPointQuery):
	"""
	@brief      Superclass for closest point queries between two specific bodies.
	"""
	def __init__(self, body, link_name, other_body, other_link, margin, dist=0.2):
		"""Constructor. Requires the names of the two bodies and their links."""
		self.body  = body
		self.link_name  = link_name
		self.other_body = other_body
		self.other_link = other_link
		self.point1 = Point3Input.prefix(symbol_formatter, 'closest_on_{}_{}_to_{}_{}'.format(body.bId(), link_name, other_body.bId(), other_link))
		self.point2 = Point3Input.prefix(symbol_formatter, 'closest_on_{}_{}_to_{}_{}'.format(other_body.bId(), other_link, body.bId(), link_name))
		self.normal = Vector3Input.prefix(symbol_formatter, 'normal_from_{}_{}_to_{}_{}'.format(other_body.bId(), other_link, body.bId(), link_name))
		self.margin = margin
		self.dist   = dist


class ClosestPointQuery_Specific_SA(ClosestPointQuery_Specific):
	"""
	@brief      Subclass for a specific single point query. In this implementation, only the first body has an active frame.
	"""
	def __init__(self, body, link_name, other_body, other_link, active_frame, margin, dist=0.2):
		super(ClosestPointQuery_Specific_SA, self).__init__(body, link_name, other_body, other_link, margin, dist)
		self.active_frame = active_frame

	def generate_constraints(self):
		dist = (self.active_frame * self.point1.get_expression()) - self.point2.get_expression()
		return {'closest_between_{}_{}_and_{}_{}'.format(self.body.bId(), self.link_name, self.other_body.bId(), self.other_link):
				SC(self.margin - dist, 1000, 100, dist)}

	# Returns dict with new values
	def update_subs_dict(self, simulator, subs, visualizer=None):
		closest = self.body.get_closest_points(self.other_body, self.link_name, self.other_link, self.dist)
		if len(closest) > 0:
			link_frame_inv = invert_transform(self.body.get_link_state(self.link_name).worldFrame)
			closest = closest[0]
			if visualizer != None:
				visualizer.draw_arrow('cpq', closest.posOnB, closest.posOnA, r=0, b=0)

			onA = transform_point(link_frame_inv, closest.posOnA)

			subs[self.point1.x] = onA[0]
			subs[self.point2.x] = closest.posOnB[0]
			#subs[self.normal.x] = closest.normalOnB[0]
			subs[self.point1.y] = onA[1]
			subs[self.point2.y] = closest.posOnB[1]
			#subs[self.normal.y] = closest.normalOnB[1]
			subs[self.point1.z] = onA[2]
			subs[self.point2.z] = closest.posOnB[2]
			#subs[self.normal.z] = closest.normalOnB[2]
		# else:
		#     obs.update(self.normal.get_update_dict(*vec3_to_list()))

	def reset_subs_dict(self, subs):
		subs[self.point1.x] = 0
		subs[self.point2.x] = 0
		#subs[self.normal.x] = closest.normalOnB[0]
		subs[self.point1.y] = 0
		subs[self.point2.y] = 0
		#subs[self.normal.y] = closest.normalOnB[1]
		subs[self.point1.z] = 0
		subs[self.point2.z] = 0


class ClosestPointQuery_Specific_BA(ClosestPointQuery_Specific):
	"""
	@brief      Subclass for a specific single point query. In this implementation, both bodies have an active frame.
	"""
	def __init__(self, body, link_name, other_body, other_link, active_frame, other_active_frame, margin, dist=0.2):
		super(ClosestPointQuery_Specific_BA, self).__init__(body, link_name, other_body, other_link, margin, dist)
		self.active_frame = active_frame
		self.other_active_frame = other_active_frame

	# Returns dict with new values
	def generate_constraints(self):
		dist = (self.active_frame * self.point1.get_expression()) - (self.other_active_frame * self.point2.get_expression())
		return {'closest_between_{}_{}_and_{}_{}'.format(self.body.bId(), self.link_name, self.other_body.bId(), self.other_link):
				SC(self.margin - dist, 1000, 100, dist)}

	# Returns dict with new values
	def update_subs_dict(self, simulator, subs, visualizer=None):
		closest = self.body.get_closest_points(self.other_body, self.link_name, self.other_link, self.dist)
		if len(closest) > 0:
			link_frame_inv  = invert_transform(self.body.get_link_state(self.link_name).worldFrame)
			other_frame_inv = invert_transform(self.other_body.get_link_state(self.other_link_name).worldFrame)
			closest = closest[0]
			if visualizer != None:
				visualizer.draw_arrow('cpq', closest.posOnB, closest.posOnA, r=0, b=0)

			onA = transform_point(link_frame_inv, closest.posOnA)
			onB = transform_point(other_frame_inv, closest.posOnB)

			subs[self.point1.x] = onA[0]
			subs[self.point2.x] = onB[0]
			#subs[self.normal.x] = closest.normalOnB[0]
			subs[self.point1.y] = onA[1]
			subs[self.point2.y] = onB[1]
			#subs[self.normal.y] = closest.normalOnB[1]
			subs[self.point1.z] = onA[2]
			subs[self.point2.z] = onB[2]
			#subs[self.normal.z] = closest.normalOnB[2]

	def reset_subs_dict(self, subs):
		subs[self.point1.x] = 0
		subs[self.point2.x] = 0
		#subs[self.normal.x] = closest.normalOnB[0]
		subs[self.point1.y] = 0
		subs[self.point2.y] = 0
		#subs[self.normal.y] = closest.normalOnB[1]
		subs[self.point1.z] = 0
		subs[self.point2.z] = 0