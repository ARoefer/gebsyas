from giskard_affordances.numeric_scene_state import Predicate
from giskard_affordances.dl_reasoning import *
from giskard_affordances.spacial_relations import SpacialRelations as SR
from giskard_affordances.grasp_affordances import BasicGraspAffordances as BGA
from giskardpy.symengine_wrappers import norm
import symengine as spw

def const_zero(*args):
	return 0

def const_one(*args):
	return 1

class DLPredicate(DLAtom):
	def __init__(self):
		super(DLPredicate, self).__init__('Predicate')

	def is_a(self, obj):
		return type(obj) is Predicate

class DLSpacialFunction(DLAtom):
	def __init__(self):
		super(DLSpacialFunction, self).__init__('SpacialFunction')

	def is_a(self, obj):
		return type(obj) == type(self.is_a) and obj.im_self is SR

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

DLGraspPredicate   = DLConjunction(DLPredicate(), DLExistsRA('fp', DLGraspFunction()))
DLSpacialPredicate   = DLConjunction(DLPredicate(), DLExistsRA('fp', DLSpacialFunction()))
DLAxiomaticPredicate = DLConjunction(DLPredicate(), DLExistsRA('fp', DLConstFunction()))

OnTop     = Predicate('OnTop', SR.on , 0.8)
Below     = Predicate('Below', SR.below, 0.8)
At        = Predicate('At', SR.at, 0.8)
Inside    = Predicate('Inside', SR.inside, 0.8)
RightOf   = Predicate('RightOf', SR.right_of, 0.8)
LeftOf    = Predicate('LeftOf', SR.left_of, 0.8)
InFrontOf = Predicate('InFrontOf', SR.in_front_of, 0.8)
Behind    = Predicate('Behind', SR.behind, 0.8)

Free    = Predicate('Free', const_one, 0.5)
Grasped = Predicate('Grasped', BGA.object_grasp, 1.5)

PREDICATE_TBOX_LIST = [DLPredicate(), DLSpacialFunction(), DLConstFunction(), DLSpacialPredicate, DLAxiomaticPredicate]
PREDICATE_TBOX = dict([(str(c), c) for c in PREDICATE_TBOX_LIST])
PREDICATE_TBOX.update({'SpacialPredicate': DLSpacialPredicate, 'AxiomaticPredicate': DLAxiomaticPredicate})

PREDICATES_LIST = [OnTop, Below, At, Inside, RightOf, LeftOf, InFrontOf, Behind, Free, Grasped]
PREDICATES = dict([(p.P, p) for p in PREDICATES_LIST])

