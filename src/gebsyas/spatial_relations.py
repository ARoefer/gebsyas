import symengine as sp

from giskardpy.symengine_wrappers import *
from giskardpy.qp_problem_builder import SoftConstraint as SC
from gebsyas.dl_reasoning import *
from gebsyas.utils import *

# Only works for x > 0 so beware
def crappyAtan2(x, y):
	return sp.atan(y/x)

def max_aabb_expr(cls, obj, frame):
	if DLCylinder.is_a(obj):
		cz = z_of(obj.pose)
		cylinder_up_if = vec3(dot(cz, x_of(frame)), dot(cz, y_of(frame)), dot(cz, z_of(frame)))
		alpha = sp.acos(fake_Abs(cylinder_up_if[2]))
		beta  = crappyAtan2(fake_Abs(cylinder_up_if[0]) + 0.0001, fake_Abs(cylinder_up_if[1]))
		cosA  = sp.cos(alpha)
		icosA = sp.sin(alpha)
		cosB  = sp.cos(beta)
		icosB = sp.sin(beta)
		z = cosA * 0.5 * obj.height + icosA * obj.radius
		y = cosB *  (icosA * 0.5 * obj.height + cosA * obj.radius) + icosB * obj.radius
		x = icosB * (icosA * 0.5 * obj.height + cosA * obj.radius) + cosB * obj.radius
		return vec3(x,y,z)
	elif DLCube.is_a(obj):
		bz = z_of(obj.pose)
		box_up_if = vec3(dot(bz, x_of(frame)), dot(bz, y_of(frame)), dot(bz, z_of(frame)))
		alpha = sp.acos(fake_Abs(box_up_if[2]))
		beta  = crappyAtan2(fake_Abs(box_up_if[0]) + 0.0001, fake_Abs(box_up_if[1]))
		gamma = sp.acos(fake_Abs(dot(z_of(frame), y_of(obj.pose))))
		cosA  = sp.cos(alpha)
		icosA = sp.sin(alpha)
		cosB  = sp.cos(beta)
		icosB = sp.sin(beta)
		cosC  = sp.cos(gamma)
		icosC = 1.0 - cosC
		up_extend    = cosC * obj.width  + icosC * obj.length
		depth_extend = cosC * obj.length + icosC * obj.width
		z = cosA * 0.5 * obj.height + icosA * obj.width
		y = icosB * (icosA * 0.5 * obj.height + cosA * depth_extend) + cosB * up_extend
		x = cosB  * (icosA * 0.5 * obj.height + cosA * up_extend) + icosB * depth_extend
		return vec3(x,y,z)
	elif DLSphere.is_a(obj):
		return vec3(obj.radius, obj.radius, obj.radius)
	raise Exception('Failed to construct AABB for object')


class DirectedSpatialRelations(object):
	@classmethod
	def at(cls, context, obj, location):
		if location.at != None:
			return location.at(obj)
		else:
			return {}

	@classmethod
	def on(cls, context, obj, location):
		if not DLRigidObject.is_a(obj):
			print('"on" expression can only be generated for things which are RigidObject.')
			return {}

		if not DLPhysicalThing.is_a(location):
			print('"on" expression can only be generated for locations which are PhysicalThing.')
			return {}

		if DLCylinder.is_a(obj) or DLCube.is_a(obj):
			if DLCube.is_a(location):
				l2o = pos_of(obj.pose) - pos_of(location.pose)
				z_align = obj.height * 0.5 + location.height * 0.5 - l2o[2]
				inxy = fake_Max(fake_Abs(dot(l2o, x_of(location.pose))) - location.length * 0.5, 0) + fake_Max(fake_Abs(dot(l2o, y_of(location.pose))) - location.width * 0.5, 0)
				in_sc = SC(-inxy, -inxy, 1, inxy)
				z_sc = SC(-z_align, -z_align, 1, z_align)
				return {'inside_rect': in_sc, 'above_cube': z_sc}
			elif DLCylinder.is_a(location):
				raise Exception('Needs update to produce inequality constraints')
				return {}
			elif DLSphere.is_a(location):
				expr = pos_of(location.pose)[2] + location.radius - pos_of(obj.pose)[2] - obj.height * 0.5
				return {'on_sphere': SC(-expr, -expr, 1, expr)}
		elif DLSphere.is_a(obj):
			if DLCube.is_a(location):
				l2o = pos_of(obj.pose) - pos_of(location.pose)
				z_align = obj.radius + location.height * 0.5 - l2o[2]
				inxy = fake_Max(fake_Abs(dot(l2o, x_of(location.pose))) - location.length * 0.5, 0) + fake_Max(fake_Abs(dot(l2o, y_of(location.pose))) - location.width * 0.5, 0)
				in_sc = SC(-inxy, -inxy, 1, inxy)
				z_sc = SC(-z_align, -z_align, 1, z_align)
				return {'inside_rect': in_sc, 'above_cube': z_sc}
			elif DLCylinder.is_a(location):
				return {}
			elif DLSphere.is_a(location):
				expr = pos_of(location.pose)[2] + location.radius - pos_of(obj.pose)[2] - obj.radius
				return {'on_sphere': SC(-expr, -expr, 1, expr)}

		print('Classification of {} or {} for "on" expression generator has failed'.format(str(obj), str(location)))
		return {}
	
	@classmethod
	def inside(cls, context, obj, volume):
		if volume.inside != None:
			return volume.inside(obj)
		else:
			return {}

	@classmethod
	def above(cls, context, obj_a, obj_b, observer):
		if not DLRigidObject.is_a(obj_a):
			raise Exception('NextTo relation is only defined for rigid objects. However object A is not one.')

		if not DLRigidObject.is_a(obj_b):
			raise Exception('NextTo relation is only defined for rigid objects. However object B is not one.')

		obs_frame = observer.frame_of_reference
		b_hdim_in_obs = max_aabb_expr(obj_b, obs_frame)

		b2a = pos_of(obj_a.pose) - pos_of(obj_b.pose)
		b2a_in_obs = vec3(dot(b2a, x_of(obs_frame)), dot(b2a, y_of(obs_frame)), dot(b2a, z_of(obs_frame)))

		max_depth = b_hdim_in_obs[0]
		min_depth = -max_depth

		max_width = b_hdim_in_obs[1]
		min_width = -max_width

		out = cls.next_to(context, obj_a, obj_b, observer)
		out.update({'above':SC(b_hdim_in_obs[2] - b2a_in_obs[2], 
							   1000, 1, b2a_in_obs[2]),
					'same_depth': SC(min_depth        - b2a_in_obs[0], max_depth - b2a_in_obs[0], 1, b2a_in_obs[0]),
					'same_width': SC(min_width        - b2a_in_obs[1], max_width - b2a_in_obs[1], 1, b2a_in_obs[1])})
		return out

	@classmethod
	def below(cls, context, obj_a, obj_b, observer):
		if not DLRigidObject.is_a(obj_a):
			raise Exception('NextTo relation is only defined for rigid objects. However object A is not one.')

		if not DLRigidObject.is_a(obj_b):
			raise Exception('NextTo relation is only defined for rigid objects. However object B is not one.')

		obs_frame = observer.frame_of_reference
		b_hdim_in_obs = max_aabb_expr(obj_b, obs_frame)

		b2a = pos_of(obj_a.pose) - pos_of(obj_b.pose)
		b2a_in_obs = vec3(dot(b2a, x_of(obs_frame)), dot(b2a, y_of(obs_frame)), dot(b2a, z_of(obs_frame)))

		max_depth = b_hdim_in_obs[0]
		min_depth = -max_depth

		max_width = b_hdim_in_obs[1]
		min_width = -max_width

		out = cls.next_to(context, obj_a, obj_b, observer)
		out.update({'below':      SC(                    -1000, b_hdim_in_obs[2] - b2a_in_obs[2], 1, b2a_in_obs[2]),
					'same_depth': SC(min_depth - b2a_in_obs[0],        max_depth - b2a_in_obs[0], 1, b2a_in_obs[0]),
					'same_width': SC(min_width - b2a_in_obs[1],        max_width - b2a_in_obs[1], 1, b2a_in_obs[1])})
		return out

	@classmethod
	def right_of(cls, context, obj_a, obj_b, observer):
		if not DLRigidObject.is_a(obj_b):
			raise Exception('RightOf relation is only defined for rigid objects')

		obs_frame = observer.frame_of_reference
		b_hdim_in_obs = max_aabb_expr(obj_b, obs_frame)

		b2a = pos_of(obj_a.pose) - pos_of(obj_b.pose)
		b2a_in_obs = vec3(dot(b2a, x_of(obs_frame)), dot(b2a, y_of(obs_frame)), dot(b2a, z_of(obs_frame)))

		max_depth = b_hdim_in_obs[0]
		min_depth = -max_depth

		max_height = b_hdim_in_obs[2]
		min_height = -max_height

		out = cls.next_to(context, obj_a, obj_b, observer)
		out.update({'right_of':    SC(                     -1000, -b_hdim_in_obs[1] - b2a_in_obs[1], 1, b2a_in_obs[1]),
					'same_depth':  SC(min_depth  - b2a_in_obs[0],         max_depth - b2a_in_obs[0], 1, b2a_in_obs[0]),
					'same_height': SC(min_height - b2a_in_obs[2],        max_height - b2a_in_obs[2], 1, b2a_in_obs[2])})
		return out

	@classmethod
	def left_of(cls, context, obj_a, obj_b, observer):
		if not DLRigidObject.is_a(obj_b):
			raise Exception('LeftOf relation is only defined for rigid objects')

		obs_frame = observer.frame_of_reference
		b_hdim_in_obs = max_aabb_expr(obj_b, obs_frame)

		b2a = pos_of(obj_a.pose) - pos_of(obj_b.pose)
		b2a_in_obs = vec3(dot(b2a, x_of(obs_frame)), dot(b2a, y_of(obs_frame)), dot(b2a, z_of(obs_frame)))

		max_depth = b_hdim_in_obs[0]
		min_depth = -max_depth

		max_height = b_hdim_in_obs[2]
		min_height = -max_height

		out = cls.next_to(context, obj_a, obj_b, observer)
		out.update({'left_of':     SC(b_hdim_in_obs[1] - b2a_in_obs[1],                       1000, 1, b2a_in_obs[1]),
					'same_depth':  SC(min_depth        - b2a_in_obs[0],  max_depth - b2a_in_obs[0], 1, b2a_in_obs[0]),
					'same_height': SC(min_height       - b2a_in_obs[2], max_height - b2a_in_obs[2], 1, b2a_in_obs[2])})
		return out

	@classmethod
	def in_front_of(cls, context, obj_a, obj_b, observer):
		obs_frame = observer.frame_of_reference
		b_hdim_in_obs = max_aabb_expr(obj_b, obs_frame)

		b2a = pos_of(obj_a.pose) - pos_of(obj_b.pose)
		b2a_in_obs = vec3(dot(b2a, x_of(obs_frame)), dot(b2a, y_of(obs_frame)), dot(b2a, z_of(obs_frame)))

		max_width = b_hdim_in_obs[1]
		min_width = -max_width

		max_height = b_hdim_in_obs[2]
		min_height = -max_height

		out = cls.next_to(context, obj_a, obj_b, observer)
		out.update({'in_front_of': SC(-1000, -b_hdim_in_obs[0] -b2a_in_obs[0], 1, b2a_in_obs[0]),
					'same_height': SC(min_height       - b2a_in_obs[2], max_height - b2a_in_obs[2], 1, b2a_in_obs[2]),
					'same_width':  SC(min_width        - b2a_in_obs[1], max_width  - b2a_in_obs[1], 1, b2a_in_obs[1])})
		return out

	@classmethod
	def behind(cls, context, obj_a, obj_b, observer):
		obs_frame = observer.frame_of_reference
		b_hdim_in_obs = max_aabb_expr(obj_b, obs_frame)

		b2a = pos_of(obj_a.pose) - pos_of(obj_b.pose)
		b2a_in_obs = vec3(dot(b2a, x_of(obs_frame)), dot(b2a, y_of(obs_frame)), dot(b2a, z_of(obs_frame)))

		max_width = b_hdim_in_obs[1]
		min_width = -max_width

		max_height = b_hdim_in_obs[2]
		min_height = -max_height

		out = cls.next_to(context, obj_a, obj_b, observer)
		out.update({'behind_of':   SC(b_hdim_in_obs[0] - b2a_in_obs[0], 1000, 1, b2a_in_obs[0]),
					'same_height': SC(min_height       - b2a_in_obs[2], max_height - b2a_in_obs[2], 1, b2a_in_obs[2]),
					'same_width':  SC(min_width        - b2a_in_obs[1], max_width  - b2a_in_obs[1], 1, b2a_in_obs[1])})
		return out

	@classmethod
	def pointing_at(cls, context, thing_a, thing_b):
		goal_vec = pos_of(thing_b.pose) - pos_of(thing_a.pose)
		ang  = acos(dot(x_of(thing_a.pose), goal_vec) / norm(goal_vec))
		ctrl = -ang
		return {'pointing_at': SC(ctrl, ctrl, 1, ang)}


	@classmethod
	def observe(cls, context, camera, obj):
		if DLProbabilisticThing.is_a(obj):
			x_variance = x_of(obj.pose_cov)
			y_variance = y_of(obj.pose_cov)
			z_variance = z_of(obj.pose_cov)


class NonDirectedSpatialRelations(object):

	@classmethod
	def clear_pos(cls, context, obj, other_objs):
		if len(other_objs) > 0:
			return 1 - sp.fake_Max(*[cls.on(o, obj) for o in other_objs])
		else:
			return 1



	@classmethod
	def uniform_space_projection(cls, context, obj, x, z):
		raise Exception('Needs update to produce inequality constraints')
		frame = x.row_join(cross(x,z)).row_join(z).row_join(pos_of(obj.frame)).inv()
		obj_x = frame * (x_of(obj.frame) * obj.dimensions[0])
		obj_y = frame * (y_of(obj.frame) * obj.dimensions[1])
		obj_z = frame * (z_of(obj.frame) * obj.dimensions[2])
		return   sp.diag(0.5 / sp.fake_Max(fake_Abs(obj_x[0]), fake_Abs(obj_y[0]), fake_Abs(obj_z[0])),
					  	 0.5 / sp.fake_Max(fake_Abs(obj_x[1]), fake_Abs(obj_y[1]), fake_Abs(obj_z[1])),
					  	 0.5 / sp.fake_Max(fake_Abs(obj_x[2]), fake_Abs(obj_y[2]), fake_Abs(obj_z[2])), 1) * frame

	@classmethod
	def next_to(cls, context, obj_a, obj_b, observer):
		if not DLRigidObject.is_a(obj_a):
			raise Exception('NextTo relation is only defined for rigid objects. However object A is not one.')

		if not DLRigidObject.is_a(obj_b):
			raise Exception('NextTo relation is only defined for rigid objects. However object B is not one.')

		if DLCube.is_a(obj_a):
			max_extent_a = fake_Max(obj_a.width, obj_b.height, obj_a.length)
		elif DLCylinder.is_a(obj_a):
			max_extent_a = fake_Max(obj_b.height, obj_a.radius)
		elif DLSphere.is_a(obj_a):
			max_extent_a = obj_a.radius

		if DLCube.is_a(obj_b):
			max_extent_b = fake_Max(obj_b.width, obj_b.height, obj_b.length)
		elif DLCylinder.is_a(obj_b):
			max_extent_b = fake_Max(obj_b.height, obj_b.radius)
		elif DLSphere.is_a(obj_b):
			max_extent_b = obj_b.radius

		d = norm(pos_of(obj_a.pose) - pos_of(obj_b.pose))
		return {'next_to': SC(-d, fake_Max(max_extent_a, max_extent_b) - d, 1, d)}



	@classmethod
	def upright(cls, context, obj):
		expr = 1 - dot(unitZ, z_of(obj.pose))
		return {'upright': SC(-expr, -expr, 1, expr)}