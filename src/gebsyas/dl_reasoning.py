from collections import namedtuple
import symengine as spw
from giskardpy.robot import Robot, Gripper, Camera
from giskardpy.symengine_wrappers import *
from gebsyas.utils import StampedData, JointState

# Class structure for symbolic data. Consists of the data itself, a conversion function and the arguments needed for the conversion.
SymbolicData = namedtuple('SymbolicData', ['data', 'f_convert', 'args'])

DLInstance = namedtuple('DLInstance', ['thing', 'label', 'concepts'])

class DLConcept(object):
	"""
	@brief      Superclass for description logical concepts.
	"""
	def is_a(self, obj):
		"""Tests whether the object is an instance of this concept."""
		raise (NotImplementedError)

	def __hash__(self):
		return hash(self.__str__())

	def __str__(self):
		raise (NotImplementedError)

	def __lt__(self, other):
		return str(self) < str(other)

	def to_NNF(self):
		"""Converts this concept to NNF. This means that only atomic concepts will be negated."""
		return self

	def implication_intersection(self, other):
		"""Returns all concepts which are jointly implied by this concept and the other."""
		return self.implies.intersection(other.implies)

class DLTop(DLConcept):
	"""
	@brief      Constant existence concept.
	"""
	def __init__(self):
		self.implied_by = {self}
		self.implies = {self}

	def is_a(self, obj):
		"""All objects are instances of T."""
		return True

	def __str__(self):
		return 'T'

	def __eq__(self, other):
		return type(other) == DLTop

	def implication_intersection(self, other):
		return other.implies

class DLBottom(DLConcept):
	"""
	@brief      Constant non-existence concept.
	"""
	def __init__(self):
		self.implied_by = {}
		self.implies = {}

	def is_a(self, obj):
		"""No object can be an instance of _."""
		return False

	def __str__(self):
		return '_'

	def __eq__(self, other):
		return type(other) == DLBottom

	def implication_intersection(self, other):
		return set()

class DLAtom(DLConcept):
	"""
	@brief      Superclass for all atomic concepts.
	"""
	def __init__(self, name):
		if name == 'T' or name == '_':
			raise Exception('A DL-atom can not be named {}. This name is reserved by the top and bottom concepts.'.format(name))
		self.name = name
		self.implied_by = {self}
		self.implies = {self}

	def is_a(self, obj):
		"""Expects objects to have a member set called "concepts". Will return true if its name is in that set."""
		return self.name in obj.concepts

	def __str__(self):
		return self.name

	def __eq__(self, other):
		return isinstance(other, DLAtom) and other.name == self.name


class DLNegation(DLConcept):
	"""
	@brief      Negating concept.
	"""
	def __init__(self, concept):
		"""
		Constructor. Requires concept to negate.
		"""
		self.concept = concept
		self.implied_by = {self}
		self.implies = {self}

	def is_a(self, obj):
		"""Returns True if object doesn't match the negated concept."""
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
	"""
	@brief      Implementation of the description logical "and".
	"""
	def __init__(self, *args):
		self.concepts = sorted(set(args))
		self.implies = set()
		for a in args:
			if type(a) == DLBottom:
				print('A non-satisfiable conjunction was created:\n   {}\nSubterm "{}" is equal to _'.format(str(self), str(a)))
				self.concepts = [DLBottom()]
				break
			self.implies = self.implies.union(a.implies)
		self.implied_by = {self}

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
	"""
	@brief      Implementation of the description logical "or".
	"""
	def __init__(self, *args):
		self.concepts = sorted(set(args))
		for a in args:
			if type(a) == DLTop:
				print('A tautological disjunction was created:\n   {}\nSubterm "{}" is equal to _'.format(str(self), str(a)))
				self.concepts = [DLTop()]
				break
		self.implies = args[0].implies
		self.implied_by = set()
		for a in self.concepts:
			self.implied_by = self.implied_by.union(a.implied_by)
			self.implies = self.implies.intersection(a.implies)
		self.implies.add(self)

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
	"""
	@brief      Implementation of the description logical "exists R successor" concept.
	"""
	def __init__(self, relation, concept=DLTop()):
		self.concept = concept
		self.relation = relation
		self.implied_by = {DLExistsRA(self.relation, x) for x in self.concept.implied_by if x != concept}
		self.implied_by.add(self)
		self.implies = {self}

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
	"""
	@brief      Implementation of the description logical "for all R successors" concept.
	"""
	def __init__(self, relation, concept=DLTop()):
		self.concept = concept
		self.relation = relation
		self.implied_by = {DLALLRA(self.relation, x) for x in self.concept.implied_by if x != concept}
		self.implied_by.add(self)
		self.implies = {self}

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
	"""
	@brief      Implementation of the description logical subsumption.
	"""
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
	"""
	@brief      Implementation of the description logical equivalence.
	"""
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
	"""
	@brief      Reasoner class. Does subsumption resolution for all concepts which are fed to it. Stores the expanded concepts.
	"""
	def __init__(self, tbox, abox):
		self.__top    = DLTop()
		self.__bottom = DLBottom()
		self.abox = abox
		self.implied_by = {}
		self.implied_by[self.__top] = set()
		self.implies    = {}
		self.implies[self.__top] = set()

		for c in tbox:
			if type(c) == DLInclusion or type(c) == DLEquivalence:
				c = c.to_NNF()

				self.__add_concept(c.concept_a)
				self.__add_concept(c.concept_b)

				self.__resolve_subsumption(c.concept_a, c.concept_b)
				if type(c) == DLEquivalence:
					self.__resolve_subsumption(c.concept_b, c.concept_a)
			elif type(c) == tuple:
				k = c[0]
				c = c[1].to_NNF()
				new_atom = DLAtom(k)
				print('Adding {} >= {}'.format(k, str(c)))
				self.__add_concept(new_atom)
				self.__add_concept(c)

				self.__resolve_subsumption(new_atom, c)
			else:
				self.__add_concept(c)

		self.tbox = {}
		for k, c in self.implies.items():
			if isinstance(k, DLAtom):
				if len(c) == 0:
					self.tbox[k] = k
				elif len(c) == 1:
					self.tbox[k] = list(c)[0]
				else:
					# Differentiation between normal atoms and those analyzing data structures
					self.tbox[k] = DLConjunction(*[x for x in c if type(x) != DLAtom and x != self.__top])


	def __add_concept(self, concept):
		if concept not in self.implies:
			self.implies[concept] = concept.implies.union({self.__top})
			for c in concept.implies:
				self.__add_concept(c)
				self.implies[concept] = self.implies[concept].union(self.implies[c])

			self.implied_by[concept] = concept.implied_by
			for c in concept.implied_by:
				self.__add_concept(c)
				self.implied_by[concept] = self.implied_by[concept].union(self.implied_by[c])

			for c in self.implies[concept]:
				self.implied_by[c] = self.implied_by[c].union(self.implied_by[concept])

			for c in self.implied_by[concept]:
				self.implies[c] = self.implies[c].union(self.implies[concept])


	# Resolve A <= B relation
	def __resolve_subsumption(self, a, b):
		self.implies[a] = self.implies[a].union(self.implies[b])
		self.implied_by[b] = self.implied_by[b].union(self.implied_by[a])

		# Add A >= C -> B >= C
		for i in self.implies[b]:
			self.implied_by[i] = self.implied_by[i].union(self.implied_by[b])

		for i in self.implied_by[a]:
			self.implies[i] = self.implies[i].union(self.implies[b])

		# self.implies[b].add(a)
		# self.implied_by[a] = self.implied_by[a].union(self.implied_by[b])

		# # Add A >= C -> B >= C
		# for i in self.implies[a]:
		# 	self.implies[b].add(i)
		# 	self.implied_by[i] = self.implied_by[i].union(self.implied_by[b])


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
		"""Returns the expanded concept for an atomic concept."""
		if type(atom) == str:
			if atom == 'T':
				return self.__top
			else:
				atom = DLAtom(atom)

		if atom in self.tbox:
			return self.tbox[atom]
		return self.__bottom

	def classify(self, obj):
		"""Returns a list of all known concepts which the object matches."""
		return [atom for atom, c in self.tbox.items() if c.is_a(obj)]

	# Checks a <= b given the current tbox
	def is_subsumed(self, a, b):
		if a in self.implied_by:
			return b in self.implied_by[a]
		return False

	def inclusions_str(self):
		return '\n'.join(['{} >= ...\n   {}'.format(str(x), '\n   '.join([str(c) for c in s])) for x, s in self.implies.items()])

	def included_str(self):
		return '\n'.join(['{} <= ...\n   {}'.format(str(x), '\n   '.join([str(c) for c in s])) for x, s in self.implied_by.items()])

	def tbox_str(self):
		return '\n'.join(['{:>15}: {}'.format(str(x), str(y)) for x, y in self.tbox.items()])


class DLScalar(DLAtom):
	"""Description logical representation of a scalar."""
	def __init__(self):
		super(DLScalar, self).__init__('Scalar')

	def is_a(self, obj):
		return type(obj) == float or type(obj) == spw.Symbol

class DLString(DLAtom):
	"""Description logical representation of a string."""
	def __init__(self):
		super(DLString, self).__init__('String')

	def is_a(self, obj):
		return type(obj) == str

class DLVector(DLAtom):
	"""Description logical representation of a 3d vector."""
	def __init__(self):
		super(DLVector, self).__init__('Vector')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 1 and obj[obj.nrows() - 1] == 0

class DLPoint(DLAtom):
	"""Description logical representation of a 3d point."""
	def __init__(self):
		super(DLPoint, self).__init__('Point')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 1 and obj[obj.nrows() - 1] == 1

class DLRotation(DLAtom):
	"""Description logical representation of a 3d rotation."""
	def __init__(self):
		super(DLRotation, self).__init__('Rotation')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 4 and obj.nrows() == 4 and norm(obj[:4, 3:]) == 0

class DLTranslation(DLAtom):
	"""Description logical representation of a 3d translation."""
	def __init__(self):
		super(DLTranslation, self).__init__('Translation')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 4 and obj.nrows() == 4 and obj[:3, :3] == spw.eye(3)

class DLTransform(DLAtom):
	"""Description logical representation of a 3d transformation."""
	def __init__(self):
		super(DLTransform, self).__init__('Transform')

	def is_a(self, obj):
		return type(obj) is spw.Matrix and obj.ncols() == 4 and obj.nrows() == 4

class DLStamp(DLAtom):
	"""Description logical representation of a time stamp."""
	def __init__(self):
		super(DLStamp, self).__init__('Stamp')

	def is_a(self, obj):
		return type(obj) == StampedData


class DLSymbolic(DLAtom):
	"""Description logical representation of symbolic data."""
	def __init__(self):
		super(DLSymbolic, self).__init__('Symbolic')

	def is_a(self, obj):
		return type(obj) == SymbolicData or type(obj) == spw.Symbol or isinstance(obj, spw.Basic) and len(obj.free_symbols) > 0


class DLRobot(DLAtom):
	"""Description logical representation of a robot."""
	def __init__(self):
		super(DLRobot, self).__init__('Robot')

	def is_a(self, obj):
		return isinstance(obj, Robot)


class DLGripper(DLAtom):
	"""Description logical representation of a gripper."""
	def __init__(self):
		super(DLGripper, self).__init__('Gripper')

	def is_a(self, obj):
		return type(obj) is Gripper


class DLCamera(DLAtom):
	"""Description logical representation of a camera."""
	def __init__(self):
		super(DLCamera, self).__init__('Camera')

	def is_a(self, obj):
		return type(obj) is Camera


class DLJointState(DLAtom):
	"""Description logical representation of a joint state."""
	def __init__(self):
		super(DLJointState, self).__init__('JointState')

	def is_a(self, obj):
		return type(obj) is JointState


class DLBodyPosture(DLAtom):
	"""Description logical representation of a robot's complete joint state."""
	def __init__(self):
		super(DLBodyPosture, self).__init__('BodyPosture')

	def is_a(self, obj):
		return type(obj) is dict and len(obj) > 0 and DLJointState().is_a(obj.values()[0])

# Description logical concept modeling a sphere
DLSphere   = DLExistsRA('radius', DLScalar())

# Description logical concept modeling a dome
DLPartialSphere = DLConjunction(DLSphere,  DLExistsRA('angle', DLScalar()))

# Description logical concept modeling a rectangle
DLRectangle = DLConjunction(DLExistsRA('width', DLScalar()), DLExistsRA('length', DLScalar()))

# Description logical concept modeling a cube
DLCube      = DLConjunction(DLExistsRA('height', DLScalar()), DLRectangle)

# Description logical concept modeling a cylinder
DLCylinder  = DLConjunction(DLExistsRA('radius', DLScalar()), DLExistsRA('height', DLScalar()))

# Description logical concept modeling a geometric shape
DLShape     = DLDisjunction(DLSphere, DLCube, DLCylinder, DLRectangle)

# Description logical concept modeling an piece of data which has an Id
DLIded      = DLExistsRA('id', DLTop())

# Description logical concept modeling any manipulator
DLManipulator = DLDisjunction(DLGripper())

# Description logical concept modeling a robot with more than one manipulator
DLMultiManipulatorRobot   = DLConjunction(DLRobot(), DLExistsRA('grippers', DLManipulator))

# Description logical concept modeling a robot with only one manipulator
DLSingleManipulatorRobot  = DLConjunction(DLRobot(), DLExistsRA('gripper', DLManipulator))

# Description logical concept modeling any thing which is capable of manipulating things
DLManipulationCapable     = DLDisjunction(DLMultiManipulatorRobot, DLSingleManipulatorRobot)

# Description logical concept modeling an observer
DLObserver = DLExistsRA('frame_of_reference', DLTransform())

#DLMesh     = DLConjunction(DLShape, DLExistsRA('radius', DLScalar))

# Description logical concept modeling a physical thing
DLPhysicalThing  = DLExistsRA('pose', DLTransform())

# Description logical concept modeling a hard-surface object
DLRigidObject    = DLConjunction(DLPhysicalThing, DLShape, DLExistsRA('mass', DLScalar()))

# Description logical concept modeling an object which is composed of other objects
DLCompoundObject = DLConjunction(DLRigidObject, DLExistsRA('subObject', DLRigidObject))

# List of all concepts in this file
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
				   				DLInclusion(DLCamera(), DLPhysicalThing),
				   				DLInclusion(DLManipulator, DLPhysicalThing)]

from gebsyas.expression_parser import UnaryOp, BinaryOp, Function

def bool_expr_tree_to_dl(node, tbox):
	"""Turns an expression tree into a DL concept"""
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



