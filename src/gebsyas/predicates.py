from gebsyas.dl_reasoning import *
from gebsyas.spatial_relations import SpatialRelations as SR
from gebsyas.grasp_affordances import BasicGraspAffordances as BGA
from giskardpy.qp_problem_builder import SoftConstraint as SC
from giskardpy.symengine_wrappers import norm
from gebsyas.trackers import SymbolicObjectPoseTracker

import symengine as spw

def const_zero(*args):
	return 0

def const_one(*args):
	return 1

class Predicate(object):
	def __init__(self, P, fp, dl_args):
		self.P       = P
		self.fp      = fp
		self.dl_args = dl_args

	def __hash__(self):
		return hash(self.P)

	def __eq__(self, other):
		return type(other) == Predicate and self.P == other.P and self.fp == other.fp

PInstance = namedtuple('PInstance', ['predicate', 'args', 'value'])

class IntrospectiveFunctions(object):
	@classmethod
	def __const_constraint_met(cls, sc):
		return sc.lower <= 0 and sc.upper >= 0

	@classmethod
	def is_grasped(cls, context, gripper, obj):
		tracker = context.agent.get_tracker(obj.id)
		if tracker != None and isinstance(tracker, SymbolicObjectPoseTracker) and tracker.anchor_id == gripper.name:
			return {'is_grasped': SC(0, 0, 1, 0)}
		else:
			return {'is_grasped': SC(1, 0, 1, 0)}

	@classmethod
	def is_controlled(cls, context, obj):
		if DLManipulator.is_a(obj):
			return {'is_controlled': SC(0, 0, 1, 0)}
		if DLCamera().is_a(obj):
			return {'is_controlled': SC(0, 0, 1, 0)}
		for Id, gripper in context.agent.get_data_state().dl_data_iterator(DLManipulator):
			if type(gripper.data) == SymbolicData:
				gripper = gripper.data
			c_dict = cls.is_grasped(context, gripper.data, obj)
			if min([cls.__const_constraint_met(sc) for sc in c_dict.values()]):
				return c_dict
		return {'is_controlled': SC(1, 0, 1, 0)}

	@classmethod
	def is_free(cls, context, gripper):
		for tracker in context.agent.trackers.values():
			if isinstance(tracker, SymbolicObjectPoseTracker) and tracker.anchor_id == gripper.name:
				return {'is_free': SC(1, 0, 1, 0)}

		return {'is_free': SC(0, 0, 1, 0)}

	@classmethod
	def is_in_posture(cls, context, joint_state, posture):
		return {jn: SC(state.position - joint_state[jn].position, state.position - joint_state[jn].position, 1, joint_state[jn].position) for jn, state in posture.items() if jn in joint_state}


class DLPredicate(DLAtom):
	def __init__(self):
		super(DLPredicate, self).__init__('Predicate')

	def is_a(self, obj):
		return type(obj) is Predicate

class DLSpatialFunction(DLAtom):
	def __init__(self):
		super(DLSpatialFunction, self).__init__('SpatialFunction')

	def is_a(self, obj):
		return type(obj) == type(self.is_a) and (obj.im_self is DSR or obj.im_self is NDSR)

class DLDirectedSpatialFunction(DLAtom):
	def __init__(self):
		super(DLDirectedSpatialFunction, self).__init__('DirectedSpatialFunction')

	def is_a(self, obj):
		return type(obj) == type(self.is_a) and obj.im_self is DSR

class DLNonDirectedSpatialFunction(DLAtom):
	def __init__(self):
		super(DLNonDirectedSpatialFunction, self).__init__('NonDirectedSpatialFunction')

	def is_a(self, obj):
		return type(obj) == type(self.is_a) and obj.im_self is NDSR

class DLGraspFunction(DLAtom):
	def __init__(self):
		super(DLGraspFunction, self).__init__('GraspFunction')

	def is_a(self, obj):
		return type(obj) == type(self.is_a) and obj.im_self is BGA

class DLConstFunction(DLAtom):
	def __init__(self):
		super(DLConstFunction, self).__init__('ConstFunction')

	def is_a(self, obj):
		return obj == const_one or obj == const_zero

class DLStateAssessment(DLAtom):
	def __init__(self):
		super(DLStateAssessment, self).__init__('StateAssessment')

	def is_a(self, obj):
		return type(obj) == type(self.is_a) and obj.im_self is IntrospectiveFunctions

DLGraspPredicate     = DLConjunction(DLPredicate(), DLExistsRA('fp', DLGraspFunction()))
DLSpatialPredicate   = DLConjunction(DLPredicate(), DLExistsRA('fp', DLSpatialFunction()))
DLAxiomaticPredicate = DLConjunction(DLPredicate(), DLExistsRA('fp', DLConstFunction()))
DLSymbolicPredicate  = DLConjunction(DLPredicate(), DLExistsRA('fp', DLStateAssessment()))

OnTop     = Predicate('OnTop',     SR.on,          (DLRigidObject, DLRigidObject))
At        = Predicate('At',        SR.at,          (DLRigidObject, DLRigidObject))
Inside    = Predicate('Inside',    SR.inside,      (DLRigidObject, DLRigidObject))
Below     = Predicate('Below',     SR.below,       (DLRigidObject, DLRigidObject, DLObserver))
Above     = Predicate('Above',     SR.above,       (DLRigidObject, DLRigidObject, DLObserver))
RightOf   = Predicate('RightOf',   SR.right_of,    (DLRigidObject, DLRigidObject, DLObserver))
LeftOf    = Predicate('LeftOf',    SR.left_of,     (DLRigidObject, DLRigidObject, DLObserver))
InFrontOf = Predicate('InFrontOf', SR.in_front_of, (DLRigidObject, DLRigidObject, DLObserver))
Behind    = Predicate('Behind',    SR.behind,      (DLRigidObject, DLRigidObject, DLObserver))
Upright   = Predicate('Upright',   SR.upright,     (DLRigidObject,))
PointingAt= Predicate('PointingAt',SR.pointing_at, (DLPhysicalThing, DLPhysicalThing))

Free      = Predicate('Free',        IntrospectiveFunctions.is_free, (DLManipulator,))
Graspable = Predicate('Graspable',     BGA.object_grasp, (DLManipulator, DLRigidObject))

IsGrasped    = Predicate('IsGrasped', IntrospectiveFunctions.is_grasped, (DLManipulator, DLRigidObject))
IsControlled = Predicate('IsControlled', IntrospectiveFunctions.is_controlled, (DLDisjunction(DLRigidObject, DLManipulator, DLCamera()),))

InPosture = Predicate('InPosture', IntrospectiveFunctions.is_in_posture, (DLBodyPosture(), DLBodyPosture()))

PREDICATE_TBOX_LIST = [DLPredicate(), DLSpatialFunction(), DLConstFunction(), DLStateAssessment(), DLSpatialPredicate, DLAxiomaticPredicate, DLGraspPredicate, DLSymbolicPredicate]
PREDICATE_TBOX = PREDICATE_TBOX_LIST + [('SpatialPredicate', DLSpatialPredicate),
										('AxiomaticPredicate', DLAxiomaticPredicate),
										('GraspPredicate', DLGraspPredicate),
										('SymbolicPredicate', DLSymbolicPredicate)]

PREDICATES_LIST = [OnTop, Below, At, Inside, Above, RightOf, LeftOf, InFrontOf, Behind, Upright, PointingAt, Free, Graspable, IsGrasped, IsControlled, InPosture]
PREDICATES = dict([(p.P, p) for p in PREDICATES_LIST])

