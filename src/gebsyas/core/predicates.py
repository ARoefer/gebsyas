from gebsyas.dl_reasoning import DLAtom, DLConcept

def const_zero(*args):
	return 0

def const_one(*args):
	return 1

class Predicate(object):
	def __init__(self, P, fp, *args):
		self.P       = P
		self.fp      = fp
		if min([isinstance(a, DLConcept) for a in args]):
			raise Exception('At least one of the type arguments passed to predicate {} is not a DL concept!\n  Types: {}'.format(P, ', '.join([str(type(a)) for a in args])))
		self.dl_args = args

	def __hash__(self):
		return hash(self.P)

	def __eq__(self, other):
		return type(other) == Predicate and self.P == other.P and self.fp == other.fp

PInstance = namedtuple('PInstance', ['predicate', 'args', 'value'])


class DLPredicate(DLAtom):
	def __init__(self):
		super(DLPredicate, self).__init__('Predicate')

	def is_a(self, obj):
		return type(obj) is Predicate

DLPredicate = DLPredicate()

DLAxiomaticPredicate = DLConjunction(DLPredicate, DLExistsRA('fp', DLConstFunction))

# PREDICATE_TBOX_LIST = [DLPredicate, DLSpatialFunction, DLConstFunction, DLStateAssessment, DLSpatialPredicate, DLAxiomaticPredicate, DLGraspPredicate, DLSymbolicPredicate]
# PREDICATE_TBOX = PREDICATE_TBOX_LIST + [('SpatialPredicate', DLSpatialPredicate),
# 										('AxiomaticPredicate', DLAxiomaticPredicate),
# 										('GraspPredicate', DLGraspPredicate),
# 										('SymbolicPredicate', DLSymbolicPredicate)]

# PREDICATES_LIST = [OnTop, NextTo, Below, At, Inside, Above, RightOf, LeftOf, InFrontOf, Behind, Upright, PointingAt, Free, Graspable, IsGrasped, IsControlled, InPosture, Exists]
# PREDICATES = dict([(p.P, p) for p in PREDICATES_LIST])

