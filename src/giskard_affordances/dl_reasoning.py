from collections import namedtuple
import symengine as spw
from giskardpy.robot import Robot, Gripper, Camera
from giskardpy.symengine_wrappers import *
from giskard_affordances.utils import StampedData, JointState


SymbolicData = namedtuple('SymbolicData', ['data', 'f_convert', 'args'])

DLInstance = namedtuple('DLInstance', ['thing', 'label', 'concepts'])

class DLConcept(object):
	def is_a(self, obj):
		raise (NotImplementedError)

	def __hash__(self):
		return hash(self.__str__())

	def __str__(self):
		raise (NotImplementedError)

	def __lt__(self, other):
		return str(self) < str(other)

	def to_NNF(self):
		return self

	def cover_intersection(self, other):
		return self.covered_by.intersection(other.covered_by)

class DLTop(DLConcept):
	def __init__(self):
		self.covered_by = {self}
		self.covers = {self}

	def is_a(self, obj):
		return True

	def __str__(self):
		return 'T'

	def __eq__(self, other):
		return type(other) == DLTop

	def cover_intersection(self, other):
		return other.covered_by

class DLBottom(DLConcept):
	def __init__(self):
		self.covered_by = {}
		self.covers = {}

	def is_a(self, obj):
		return False

	def __str__(self):
		return '_'

	def __eq__(self, other):
		return type(other) == DLBottom

	def cover_intersection(self, other):
		return {}

class DLAtom(DLConcept):
	def __init__(self, name):
		if name == 'T' or name == '_':
			raise Exception('A DL-atom can not be named {}. This name is reserved by the top and bottom concepts.'.format(name))
		self.name = name
		self.covered_by = {self}
		self.covers = {self}

	def is_a(self, obj):
		return self.name in obj.concepts

	def __str__(self):
		return self.name

	def __eq__(self, other):
		return isinstance(other, DLAtom) and other.name == self.name


class DLNegation(DLConcept):
	def __init__(self, concept):
		self.concept = concept
		self.covered_by = {self}
		self.covers = {self}

	def is_a(self, obj):
		return not self.concept.is_a(obj)

	def __str__(self):
		return '-{}'.format(str(self.concept))

	def __eq__(self, other):
		return type(other) == DLNegation and other.concept == self.concept

	def to_NNF(self):
		if not isinstance(self.concept, DLAtom):
			tc = type(self.concept)
			if tc == DLTop:
				return DLBottom()
			elif tc == DLBottom:
				return DLTop()
			elif tc == DLConjunction:
				return DLDisjunction(*[DLNegation(c).to_NNF() for c in self.concept.concepts])
			elif tc == DLDisjunction:
				return DLConjunction(*[DLNegation(c).to_NNF() for c in self.concept.concepts])
			elif tc == DLExistsRA:
				return DLAllRA(self.concept.relation, DLNegation(self.concept.concept).to_NNF())
			elif tc == DLAllRA:
				return DLExistsRA(self.concept.relation, DLNegation(self.concept.concept).to_NNF())
			elif tc == DLInclusion:
				raise Exception('Inclusive concepts can not be normalized. Use inclusions only for TBox-statements. Concept:\n   {}'.format(self))
			elif tc == DLEquivalence:
				raise Exception('Equivalence concepts can not be normalized. Use equivalencies only for TBox-statements. Concept:\n   {}'.format(self))
		return self


class DLConjunction(DLConcept):
	def __init__(self, *args):
		self.concepts = sorted(set(args))
		self.covers = {self}
		for a in args:
			if type(a) == DLBottom:
				print('A non-satisfiable conjunction was created:\n   {}\nSubterm "{}" is equal to _'.format(str(self), str(a)))
				self.concepts = [DLBottom()]
				break
			self.covers = self.covers.union(a.covers)
		self.covered_by = {self}

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
		self.concepts = sorted(set(args))
		for a in args:
			if type(a) == DLTop:
				print('A tautological disjunction was created:\n   {}\nSubterm "{}" is equal to _'.format(str(self), str(a)))
				self.concepts = [DLTop()]
				break
		self.covers = args[0].covers
		self.covered_by = {self}
		for a in self.concepts:
			self.covered_by = self.covered_by.union(a.covered_by)
			self.covers = self.covers.intersection(a.covers)
		self.covers.add(self)

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
		self.covered_by = {DLExistsRA(self.relation, x) for x in self.concept.covered_by if x != concept}
		self.covered_by.add(self)
		self.covers = {self}

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
			pass
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
		self.covered_by = {DLALLRA(self.relation, x) for x in self.concept.covered_by if x != concept}
		self.covered_by.add(self)
		self.covers = {self}

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
			pass
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

	def to_NNF(self):
		return DLInclusion(self.concept_a.to_NNF(), self.concept_b.to_NNF())

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

	def to_NNF(self):
		return DLEquivalence(self.concept_a.to_NNF(), self.concept_b.to_NNF())

class Reasoner(object):
	def __init__(self, tbox, abox):
		self.__top    = DLTop()
		self.__bottom = DLBottom()
		self.abox = abox
		self.included_by = {}
		self.included_by[self.__top] = set()
		self.includes    = {}
		self.includes[self.__top]   = set()

		for c in tbox:
			if type(c) == DLInclusion or type(c) == DLEquivalence:
				c = c.to_NNF()
				if c.concept_a not in self.included_by:
					self.included_by[c.concept_a] = {self.__top}
				if c.concept_a not in self.includes:
					self.includes[c.concept_a] = set()

				if c.concept_b not in self.included_by:
					self.included_by[c.concept_b] = {self.__top}
				if c.concept_b not in self.includes:
					self.includes[c.concept_b] = set()

				self.includes[self.__top].add(c.concept_a)
				self.includes[self.__top].add(c.concept_b)

				self.__resolve_subsumption(c.concept_a, c.concept_b)
				if type(c) == DLEquivalence:
					self.__resolve_subsumption(c.concept_b, c.concept_a)
			elif type(c) == tuple:
				k = c[0]
				c = c[1].to_NNF()
				new_atom = DLAtom(k)
				print('Adding {} >= {}'.format(k, str(c)))
				if new_atom not in self.included_by:
					self.included_by[new_atom] = {self.__top}
				if new_atom not in self.includes:
					self.includes[new_atom] = set()

				if c not in self.included_by:
					self.included_by[c] = {self.__top}
				if c not in self.includes:
					self.includes[c] = set()

				self.includes[self.__top].add(new_atom)
				self.includes[self.__top].add(c)
				self.__resolve_subsumption(c, new_atom)
			else:
				if c not in self.included_by:
					self.included_by[c] = {self.__top}
				if c not in self.includes:
					self.includes[c] = set()

				self.includes[self.__top].add(c)

		self.tbox = {}
		for k, c in self.includes.items():
			if isinstance(k, DLAtom):
				if len(c) == 0:
					self.tbox[k] = k
				elif len(c) == 1:
					self.tbox[k] = list(c)[0]
				else:
					self.tbox[k] = DLDisjunction(*list(c))


	# Resolve A <= B relation
	def __resolve_subsumption(self, a, b):
		self.includes[b].add(a)
		self.included_by[a].add(b)

		# Add A >= C -> B >= C
		for i in self.includes[a]:
			self.includes[b].add(i)
			self.included_by[i].add(b)


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

	def __getitem__(self, atom):
		return self.get_expanded_concept(atom)

	def __contains__(self, atom):
		return self.get_expanded_concept(atom) != self.__bottom

	def get_expanded_concept(self, atom):
		if type(atom) == str:
			if atom == 'T':
				return self.__top
			else:
				atom = DLAtom(atom)

		if atom in self.tbox:
			return self.tbox[atom]
		return self.__bottom

	def classify(self, obj):
		return [atom for atom, c in self.tbox.items() if c.is_a(obj)]

	# Checks a <= b given the current tbox
	def is_subsumed(self, a, b):
		if a in self.included_by:
			return b in self.included_by[a]
		return False

	def inclusions_str(self):
		return '\n'.join(['{} >= ...\n   {}'.format(str(x), '\n   '.join([str(c) for c in s])) for x, s in self.includes.items()])

	def included_str(self):
		return '\n'.join(['{} <= ...\n   {}'.format(str(x), '\n   '.join([str(c) for c in s])) for x, s in self.included_by.items()])

	def tbox_str(self):
		return '\n'.join(['{:>15}: {}'.format(str(x), str(y)) for x, y in self.tbox.items()])


class DLScalar(DLAtom):
	def __init__(self):
		super(DLScalar, self).__init__('Scalar')

	def is_a(self, obj):
		return type(obj) == float or type(obj) == spw.Symbol

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
		return type(obj) == SymbolicData or type(obj) == spw.Symbol or isinstance(obj, spw.Basic) and len(obj.free_symbols) > 0


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


class DLCamera(DLAtom):
	def __init__(self):
		super(DLCamera, self).__init__('Camera')

	def is_a(self, obj):
		return type(obj) is Camera


class DLJointState(DLAtom):
	def __init__(self):
		super(DLJointState, self).__init__('JointState')

	def is_a(self, obj):
		return type(obj) is JointState


class DLBodyPosture(DLAtom):
	def __init__(self):
		super(DLBodyPosture, self).__init__('BodyPosture')

	def is_a(self, obj):
		return type(obj) is dict and len(obj) > 0 and DLJointState().is_a(obj.values()[0])


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

DLObserver = DLExistsRA('frame_of_reference', DLTransform())

#DLMesh     = DLConjunction(DLShape, DLExistsRA('radius', DLScalar))

DLPhysicalThing  = DLExistsRA('pose', DLTransform())
DLRigidObject    = DLConjunction(DLPhysicalThing, DLShape, DLExistsRA('mass', DLScalar()))
DLCompoundObject = DLConjunction(DLRigidObject, DLExistsRA('subObject', DLRigidObject))

BASIC_TBOX_LIST = [DLString(), DLScalar(),
				   DLVector(), DLPoint(), DLRotation(), DLTranslation(),
				   DLTransform(), DLStamp(), DLSymbolic(), DLRobot(), DLGripper(), DLJointState(), DLBodyPosture(),
				   DLManipulator, DLMultiManipulatorRobot, DLSingleManipulatorRobot, DLManipulationCapable,
				   DLSphere, DLPartialSphere, DLRectangle, DLCube, DLCylinder, DLShape, DLIded, DLObserver]

# Add convenient aliases
BASIC_TBOX = BASIC_TBOX_LIST + [('Sphere', DLSphere),
							    ('PartialSphere', DLPartialSphere),
							    ('Rectangle', DLRectangle),
							    ('Cube', DLCube),
							    ('Cylinder', DLCylinder),
							    ('Shape', DLShape),
				   				('PhysicalThing', DLPhysicalThing),
				   				('RigidObject', DLRigidObject),
				   				('CompoundObject', DLCompoundObject),
				   				('Ided', DLIded),
				   				('Manipulator', DLManipulator),
				   				('ManipulationCapable', DLManipulationCapable),
				   				('Observer', DLObserver),
				   				DLInclusion(DLRigidObject, DLCompoundObject),
				   				DLInclusion(DLCamera(), DLPhysicalThing),
				   				DLInclusion(DLManipulator, DLPhysicalThing)]

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



