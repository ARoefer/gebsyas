from giskardpy.symengine_wrappers import *


class Thing(object):
	def __init__(self, name):
		self.name = name


class PhysicalThing(Thing):
	def __init__(self, name, frame):
		super(PhysicalThing, self).__init__(name)
		self.frame = frame


class RigidObject(PhysicalThing):
	"""docstring for RigidObject"""
	def __init__(self, name, frame, dimensions):
		super(RigidObject, self).__init__(name, frame)
		self.dimensions = dimensions


class Sphere(RigidObject):
	"""docstring for Sphere"""
	def __init__(self, name, frame, radius):
		super(Sphere, self).__init__(name, frame, vec3(*([radius]*3)))

	def on(self, obj, up=unitZ):
		goal = (self.dimensions[2] + abs((obj.frame * obj.dimensions)[2])) * 0.5
		s2o = pos_of(obj.frame) - pos_of(self.frame)
		return -dot(up, s2o)**2 + (goal - s2o[2])**2 + 1


class Cube(RigidObject):

	def on(self, obj, up=unitZ):
		pass


