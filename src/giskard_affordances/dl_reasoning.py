from collections import namedtuple
import symengine as spw
from giskardpy.robot import Robot
from fetch_giskard.fetch_robot import Gripper
from utils import StampedData


SymbolicData = namedtuple('SymbolicData', ['data', 'f_convert', 'args'])

DLInstance = namedtuple('DLInstance', ['thing', 'label', 'concepts'])

class DLConcept(object):
	def is_a(self, obj):
		raise (NotImplementedError)

	def __hash__(self):
		return hash(self.__str__())

	def __str__(self):
		raise (NotImplementedError)

class DLTop(DLConcept):
	def is_a(self, obj):
		return True

	def __str__(self):
		return 'T'

	def __eq__(self, other):
		return type(other) == DLTop

class DLBottom(DLConcept):
	def is_a(self, obj):
		return False

	def __str__(self):
		return '_'

	def __eq__(self, other):
		return type(other) == DLBottom

class DLAtom(DLConcept):
	def __init__(self, name):
		if name == 'T' or name == '_':
			raise Exception('A DL-atom can not be named {}. This name is reserved by the top and bottom concepts.'.format(name))
		self.name = name

	def is_a(self, obj):
		return self.name in obj.concepts

	def __str__(self):
		return self.name

	def __eq__(self, other):
		return type(other) == DLAtom and other.name == self.name


class DLNegation(DLConcept):
	def __init__(self, concept):
		self.concept = concept

	def is_a(self, obj):
		return not self.concept.is_a(obj)

	def __str__(self):
		return '-{}'.format(str(self.concept))

	def __eq__(self, other):
		return type(other) == DLNegation and other.concept == self.concept

class DLConjunction(DLConcept):
	def __init__(self, *args):
		self.concepts = args

	def is_a(self, obj):
		for c in self.concepts:
			if not c.is_a(obj):
				return False
		return True

	def __str__(self):
		sub_str = []
		for c in self.concepts:
			if type(c) == DLDisjunction:
				sub_str.append('({})'.format(str(c)))
			else:
				sub_str.append(str(c))
		return ' & '.join(sub_str)

	def __eq__(self, other):
		return type(other) == DLConjunction and other.concepts == self.concepts

class DLDisjunction(DLConcept):
	def __init__(self, *args):
		self.concepts = args

	def is_a(self, obj):
		for c in self.concepts:
			if c.is_a(obj):
				return True
		return False

	def __str__(self):
		sub_str = []
		for c in self.concepts:
			if type(c) == DLConjunction:
				sub_str.append('({})'.format(str(c)))
			else:
				sub_str.append(str(c))
		return ' | '.join(sub_str)

	def __eq__(self, other):
		return type(other) == DLDisjunction and other.concepts == self.concepts

class DLExistsRA(DLConcept):
	def __init__(self, relation, concept=DLTop()):
		self.concept = concept
		self.relation = relation

	def is_a(self, obj):
		try:
			rs = getattr(obj, self.relation)
			if type(rs) == SymbolicData:
				rs = rs.data

			if type(rs) == list:
				for r in rs:
					if self.concept.is_a(r):
						return True
			else:
				return self.concept.is_a(rs)
		except AttributeError:
			return False

	def __str__(self):
		if type(self.concept) == DLConjunction or type(self.concept) == DLDisjunction:
			return 'E.{}.({})'.format(self.relation, str(self.concept))
		else:
			return 'E.{}.{}'.format(self.relation, str(self.concept))

	def __eq__(self, other):
		return type(other) == DLExistsRA and other.relation == self.relation and other.concept == self.concept

class DLAllRA(DLConcept):
	def __init__(self, relation, concept=DLTop()):
		self.concept = concept
		self.relation = relation

	def is_a(self, obj):
		try:
			rs = getattr(obj, self.relation)
			if type(rs) == SymbolicData:
				rs = rs.data

			if type(rs) == list:
				for r in rs:
					if not self.concept.is_a(r):
						return False
			else:
				return self.concept.is_a(rs)
		except AttributeError:
			return True

	def __str__(self):
		if type(self.concept) == DLConjunction or type(self.concept) == DLDisjunction:
			return 'A.{}.({})'.format(self.relation, str(self.concept))
		else:
			return 'A.{}.{}'.format(self.relation, str(self.concept))

	def __eq__(self, other):
		return type(other) == DLAllRA and other.relation == self.relation and other.concept == self.concept

class DLInclusion(DLConcept):
	def __init__(self, concept_a, concept_b):
		self.concept_a = concept_a
		self.concept_b = concept_b

	def is_a(self, obj):
		return not self.concept_a.is_a(obj) or self.concept_b.is_a(obj)

	def __str__(self):
		return '{} < {}'.format(str(self.concept_a), str(self.concept_b))

	def __eq__(self, other):
		return type(other) == DLInclusion and other.concept_a == self.concept_a and other.concept_b == self.concept_b

class DLEquivalence(DLConcept):
	def __init__(self, concept_a, concept_b):
		self.concept_a = concept_a
		self.concept_b = concept_b

	def is_a(self, obj):
		return self.concept_a.is_a(obj) == self.concept_b.is_a(obj)

	def __str__(self):
		return '{} = {}'.format(str(self.concept_a), str(self.concept_b))

	def __eq__(self, other):
		return type(other) == DLEquivalence and other.relation == self.relation and other.concept == self.concept

class Reasoner(object):
	def __init__(self, tbox, abox):
		self.tbox = tbox
		self.abox = abox
		self.__bottom = self.tbox['_']

	def add_to_abox(self, *args):
		for Id, concept in args:
			self.abox[Id] = concept

	def remove_from_abox(self, *args):
		for Id in args:
			if Id in self.abox:
				del self.abox[Id]

	def get_class(self, Id):
		if Id in self.abox:
			return self.abox[Id]
		else:
			return DLBottom()

	def add_concept(self, concept):
		self.tbox[str(concept)] = concept

	def get_concept(self, name):
		if name in self.tbox:
			return self.tbox[name]
		return self.__bottom


class DLScalar(DLAtom):
	def __init__(self):
		super(DLScalar, self).__init__('Scalar')

	def is_a(self, obj):
		return type(obj) == float

class DLString(DLAtom):
	def __init__(self):
		super(DLString, self).__init__('String')

	def is_a(self, obj):
		return type(obj) == str

class DLVector(DLAtom):
	def __init__(self):
		super(DLVector, self).__init__('Vector')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 1 and obj[obj.nrows() - 1] == 0

class DLPoint(DLAtom):
	def __init__(self):
		super(DLPoint, self).__init__('Point')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 1 and obj[obj.nrows() - 1] == 1

class DLRotation(DLAtom):
	def __init__(self):
		super(DLRotation, self).__init__('Rotation')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 4 and obj.nrows() == 4 and norm(obj[:4, 3:]) == 0

class DLTranslation(DLAtom):
	def __init__(self):
		super(DLTranslation, self).__init__('Translation')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 4 and obj.nrows() == 4 and obj[:3, :3] == spw.eye(3)

class DLTransform(DLAtom):
	def __init__(self):
		super(DLTransform, self).__init__('Transform')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 4 and obj.nrows() == 4

class DLStamp(DLAtom):
	def __init__(self):
		super(DLStamp, self).__init__('Stamp')

	def is_a(self, obj):
		return type(obj) == StampedData


class DLSymbolic(DLAtom):
	def __init__(self):
		super(DLSymbolic, self).__init__('Symbolic')

	def is_a(self, obj):
		return type(obj) == SymbolicData or isinstance(obj, spw.Basic) and len(obj.free_symbols) > 0


class DLRobot(DLAtom):
	def __init__(self):
		super(DLRobot, self).__init__('Robot')

	def is_a(self, obj):
		return isinstance(obj, Robot)

class DLGripper(DLAtom):
	def __init__(self):
		super(DLGripper, self).__init__('Gripper')

	def is_a(self, obj):
		return type(obj) is Gripper


DLSphere   = DLExistsRA('radius', DLScalar())
DLPartialSphere = DLConjunction(DLSphere,  DLExistsRA('angle', DLScalar()))
DLRectangle = DLConjunction(DLExistsRA('width', DLScalar()), DLExistsRA('length', DLScalar()))
DLCube      = DLConjunction(DLExistsRA('height', DLScalar()), DLRectangle)
DLCylinder  = DLConjunction(DLExistsRA('radius', DLScalar()), DLExistsRA('height', DLScalar()))
DLShape     = DLDisjunction(DLSphere, DLCube, DLCylinder, DLRectangle)
DLIded      = DLExistsRA('id', DLTop())
DLManipulator = DLDisjunction(DLGripper())

DLMultiManipulatorRobot   = DLConjunction(DLRobot(), DLExistsRA('grippers', DLManipulator))
DLSingleManipulatorRobot  = DLConjunction(DLRobot(), DLExistsRA('gripper', DLManipulator))
DLManipulationCapable     = DLDisjunction(DLMultiManipulatorRobot, DLSingleManipulatorRobot)

#DLMesh     = DLConjunction(DLShape, DLExistsRA('radius', DLScalar))

DLPhysicalThing  = DLExistsRA('pose', DLTransform())
DLRigidObject    = DLConjunction(DLPhysicalThing, DLDisjunction(DLShape, DLExistsRA('subObject', DLTop())))
DLCompoundObject = DLConjunction(DLRigidObject, DLExistsRA('subObject', DLRigidObject))

BASIC_TBOX_LIST = [DLTop(), DLBottom(), DLString(), DLScalar(),
				   DLVector(), DLPoint(), DLRotation(), DLTranslation(),
				   DLTransform(), DLStamp(), DLSymbolic(), DLRobot(), DLGripper(),
				   DLManipulator, DLMultiManipulatorRobot, DLSingleManipulatorRobot, DLManipulationCapable,
				   DLSphere, DLPartialSphere, DLRectangle, DLCube, DLCylinder, DLShape, DLIded]

BASIC_TBOX = dict([(str(c), c) for c in BASIC_TBOX_LIST])
# Add convenient aliases
BASIC_TBOX.update({'Sphere': DLSphere, 'PartialSphere': DLPartialSphere,'Rectangle': DLRectangle,
				   'Cube': DLCube, 'Cylinder': DLCylinder, 'Shape': DLShape,
				   'PhysicalThing': DLPhysicalThing, 'RigidObject': DLRigidObject,
				   'CompoundObject': DLCompoundObject, 'Ided': DLIded, 'Manipulator': DLManipulator,
				   'ManipulationCapable': DLManipulationCapable})

from giskard_affordances.expression_parser import UnaryOp, BinaryOp, Function

def bool_expr_tree_to_dl(node, tbox):
	tn = type(node)
	if tn == str:
		if node in tbox:
			return tbox[node]
		else:
			raise Exception('Atomic concept {} is not defined in TBox'.format(node))
	elif tn == UnaryOp:
		if node.op == 'not':
			return DLNegation(bool_expr_tree_to_dl(node.a, tbox))
	elif tn == BinaryOp:
		if node.op == 'and':
			return DLConjunction(bool_expr_tree_to_dl(node.a, tbox), bool_expr_tree_to_dl(node.b, tbox))
		elif node.op == 'or':
			return DLDisjunction(bool_expr_tree_to_dl(node.a, tbox), bool_expr_tree_to_dl(node.b, tbox))

	raise Exception('No way of interpreting node {} of type {} as a dl concept.'.format(node, tn))



