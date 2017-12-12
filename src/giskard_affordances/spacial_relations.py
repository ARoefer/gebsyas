from giskardpy.symengine_wrappers import *
import symengine as sp
from giskard_affordances.dl_reasoning import *
from giskard_affordances.utils import *

class SpacialRelations(object):

	@classmethod
	def at(cls, obj, location):
		if location.at != None:
			return location.at(obj)
		else:
			return 0

	@classmethod
	def clear_pos(cls, obj, other_objs):
		if len(other_objs) > 0:
			return 1 - sp.Max(*[cls.on(o, obj) for o in other_objs])
		else:
			return 1

	@classmethod
	def on(cls, obj, location):
		if not DLRigidObject.is_a(obj):
			print('"on" expression can only be generated for things which are RigidObject.')
			return 0

		if not DLPhysicalThing.is_a(location):
			print('"on" expression can only be generated for locations which are PhysicalThing.')
			return 0

		if DLCylinder.is_a(obj) or DLCube.is_a(obj):
			if DLCube.is_a(location):
				l2o = pos_of(obj.pose) - pos_of(location.pose)
				z_align = abs(l2o[2] - obj.height * 0.5 - location.height * 0.5) / 0.2
				inside  = saturate(abs(dot(l2o, x_col(location.pose)) / location.width - 1) + abs(dot(l2o, y_col(location.pose)) / location.length - 1))
				return 1 - inside - z_align
			elif DLCylinder.is_a(location):
				return 0
			elif DLSphere.is_a(location):
				return 1 - abs(pos_of(obj.pose)[2] - obj.height * 0.5 - pos_of(location.pose)[2] - location.radius) / 0.2
		elif DLSphere.is_a(obj):
			if DLCube.is_a(location):
				l2o = pos_of(obj.pose) - pos_of(location.pose)
				z_align = abs(l2o[2] - obj.radius - location.height * 0.5) / 0.2
				inside  = saturate(abs(dot(l2o, x_col(location.pose)) / location.width - 1) + abs(dot(l2o, y_col(location.pose)) / location.length - 1))
				return 1 -  norm(l2o) #inside - z_align
			elif DLCylinder.is_a(location):
				return 0
			elif DLSphere.is_a(location):
				return 1 - abs(pos_of(obj.pose)[2] - obj.radius - pos_of(location.pose)[2] - location.radius) / 0.2

		print('Classification of {} or {} for "on" expression generator has failed'.format(str(obj), str(location)))
		return 0

	@classmethod
	def below(cls, obj, location):
		if location.below != None:
			return location.below(obj)
		else:
			return 0

	@classmethod
	def inside(cls, obj, volume):
		if volume.inside != None:
			return volume.inside(obj)
		else:
			return 0

	@classmethod
	def uniform_space_projection(cls, obj, x, z):
		frame = x.row_join(cross(x,z)).row_join(z).row_join(pos_of(obj.frame)).inv()
		obj_x = frame * (x_col(obj.frame) * obj.dimensions[0])
		obj_y = frame * (y_col(obj.frame) * obj.dimensions[1])
		obj_z = frame * (z_col(obj.frame) * obj.dimensions[2])
		return   sp.diag(0.5 / sp.Max(abs(obj_x[0]), abs(obj_y[0]), abs(obj_z[0])),
					  	 0.5 / sp.Max(abs(obj_x[1]), abs(obj_y[1]), abs(obj_z[1])),
					  	 0.5 / sp.Max(abs(obj_x[2]), abs(obj_y[2]), abs(obj_z[2])), 1) * frame


	@classmethod
	def right_of(cls, obj_a, obj_b, forward=unitX, up=unitZ):
		projection = cls.uniform_space_projection(obj_b, forward, z)
		obj_p_pos = projection * pos_of(obj_a.frame)
		right_rating = -(obj_p_pos[1]**2) - 3 * obj_p_pos[1] - 1.25
		closeness    = saturate(-(obj_p_pos[0]**2 * obj_p_pos[2]**2) + 1)
		return right_rating * closeness

	@classmethod
	def left_of(cls, obj_a, obj_b, forward=unitX, up=unitZ):
		projection = cls.uniform_space_projection(obj_b, forward, z)
		obj_p_pos = projection * pos_of(obj_a.frame)
		left_rating = -(obj_p_pos[1]**2) + 3 * obj_p_pos[1] - 1.25
		closeness    = saturate(-(obj_p_pos[0]**2 * obj_p_pos[2]**2) + 1)
		return left_rating * closeness

	@classmethod
	def in_front_of(cls, obj_a, obj_b, forward=unitX, up=unitZ):
		projection = cls.uniform_space_projection(obj_b, forward, z)
		obj_p_pos = projection * pos_of(obj_a.frame)
		front_rating = -(obj_p_pos[0]**2) - 3 * obj_p_pos[0] - 1.25
		closeness    = saturate(-(obj_p_pos[1]**2 * obj_p_pos[2]**2) + 1)
		return front_rating * closeness

	@classmethod
	def behind(cls, obj_a, obj_b, forward=unitX, up=unitZ):
		projection = cls.uniform_space_projection(obj_b, forward, z)
		obj_p_pos = projection * pos_of(obj_a.frame)
		back_rating = -(obj_p_pos[0]**2) + 3 * obj_p_pos[0] - 1.25
		closeness    = saturate(-(obj_p_pos[1]**2 * obj_p_pos[2]**2) + 1)
		return back_rating * closeness



