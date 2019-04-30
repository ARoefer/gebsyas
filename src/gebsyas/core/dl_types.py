from gebsyas.core.dl_reasoning import DLTop,DLBottom, DLInclusion, DLAtom, DLConjunction, DLDisjunction, DLExistsRA
from giskardpy.symengine_robot import Robot, Gripper, Camera
from gebsyas.data_structures import StampedData, JointState, SymbolicData, GaussianPoseComponent
from giskardpy.symengine_wrappers import *


class DLScalar(DLAtom):
    """Description logical representation of a scalar."""
    def __init__(self):
        super(DLScalar, self).__init__('Scalar')

    def is_a(self, obj):
        return type(obj) == float or type(obj) == sp.RealDouble or type(obj) == sp.Symbol

DLScalar = DLScalar()


class DLString(DLAtom):
    """Description logical representation of a string."""
    def __init__(self):
        super(DLString, self).__init__('String')

    def is_a(self, obj):
        return type(obj) == str

DLString = DLString()


class DLVector(DLAtom):
    """Description logical representation of a 3d vector."""
    def __init__(self):
        super(DLVector, self).__init__('Vector')

    def is_a(self, obj):
        return type(obj) is sp.Matrix and obj.ncols() == 1 and obj[obj.nrows() - 1] == 0

DLVector = DLVector()


class DLPoint(DLAtom):
    """Description logical representation of a 3d point."""
    def __init__(self):
        super(DLPoint, self).__init__('Point')

    def is_a(self, obj):
        return type(obj) is sp.Matrix and obj.ncols() == 1 and obj[obj.nrows() - 1] == 1

DLPoint = DLPoint()


class DLRotation(DLAtom):
    """Description logical representation of a 3d rotation."""
    def __init__(self):
        super(DLRotation, self).__init__('Rotation')

    def is_a(self, obj):
        return type(obj) is sp.Matrix and obj.ncols() == 4 and obj.nrows() == 4 and norm(obj[:3, 3:]) == 0

DLRotation = DLRotation()


class DLTranslation(DLAtom):
    """Description logical representation of a 3d translation."""
    def __init__(self):
        super(DLTranslation, self).__init__('Translation')

    def is_a(self, obj):
        return type(obj) is sp.Matrix and obj.ncols() == 4 and obj.nrows() == 4 and obj[:3, :3] == sp.eye(3)

DLTranslation = DLTranslation()


class DLTransform(DLAtom):
    """Description logical representation of a 3d transformation."""
    def __init__(self):
        super(DLTransform, self).__init__('Transform')

    def is_a(self, obj):
        return type(obj) is sp.Matrix and obj.ncols() == 4 and obj.nrows() == 4

DLTransform = DLTransform()


class DLCovarianceMatrix(DLAtom):
    """Description logical representation of a 3d covariance matrix."""
    def __init__(self):
        super(DLCovarianceMatrix, self).__init__('CovarianceMatrix')

    def is_a(self, obj):
        return type(obj) is sp.Matrix and obj.ncols() == 6 and obj.nrows() == 6

DLCovarianceMatrix = DLCovarianceMatrix()


class DLStamp(DLAtom):
    """Description logical representation of a time stamp."""
    def __init__(self):
        super(DLStamp, self).__init__('Stamp')

    def is_a(self, obj):
        return type(obj) == StampedData

DLStamp = DLStamp()


class DLSymbolic(DLAtom):
    """Description logical representation of symbolic data."""
    def __init__(self):
        super(DLSymbolic, self).__init__('Symbolic')

    def is_a(self, obj):
        return type(obj) == SymbolicData or type(obj) == sp.Symbol or isinstance(obj, sp.Basic) and len(obj.free_symbols) > 0

DLSymbolic = DLSymbolic()


class DLRobot(DLAtom):
    """Description logical representation of a robot."""
    def __init__(self):
        super(DLRobot, self).__init__('Robot')

    def is_a(self, obj):
        return isinstance(obj, Robot)

DLRobot = DLRobot()


class DLGripper(DLAtom):
    """Description logical representation of a gripper."""
    def __init__(self):
        super(DLGripper, self).__init__('Gripper')

    def is_a(self, obj):
        return type(obj) is Gripper

DLGripper = DLGripper()


class DLCamera(DLAtom):
    """Description logical representation of a camera."""
    def __init__(self):
        super(DLCamera, self).__init__('Camera')

    def is_a(self, obj):
        return type(obj) is Camera

DLCamera = DLCamera()


class DLJointState(DLAtom):
    """Description logical representation of a joint state."""
    def __init__(self):
        super(DLJointState, self).__init__('JointState')

    def is_a(self, obj):
        return type(obj) is JointState

DLJointState = DLJointState()


class DLBodyPosture(DLAtom):
    """Description logical representation of a robot's complete joint state."""
    def __init__(self):
        super(DLBodyPosture, self).__init__('BodyPosture')

    def is_a(self, obj):
        return type(obj) is dict and len(obj) > 0 and DLJointState().is_a(obj.values()[0])

DLBodyPosture = DLBodyPosture()


class DLGMMPoseComponent(DLAtom):
    def __init__(self):
        super(DLGMMPoseComponent, self).__init__('GMMPoseComponent')

    def is_a(self, obj):
        return type(obj) == GaussianPoseComponent

DLGMMPoseComponent = DLGMMPoseComponent()


# Description logical concept modelling a sphere
DLSphere   = DLExistsRA('radius', DLScalar)

# Description logical concept modelling a dome
DLPartialSphere = DLConjunction(DLSphere,  DLExistsRA('angle', DLScalar))

# Description logical concept modelling a rectangle
DLRectangle = DLConjunction(DLExistsRA('width', DLScalar), DLExistsRA('length', DLScalar))

# Description logical concept modelling a cube
DLCube      = DLConjunction(DLExistsRA('height', DLScalar), DLRectangle)

# Description logical concept modelling a cylinder
DLCylinder  = DLConjunction(DLExistsRA('radius', DLScalar), DLExistsRA('height', DLScalar))

# Description logical concept modelling a geometric shape
DLShape     = DLDisjunction(DLSphere, DLCube, DLCylinder, DLRectangle)

# Description logical concept modelling an piece of data which has an Id
DLIded      = DLExistsRA('id', DLTop)

# Description logical concept modelling any manipulator
DLManipulator = DLDisjunction(DLGripper)

# Description logical concept modelling a robot with more than one manipulator
DLMultiManipulatorRobot   = DLConjunction(DLRobot, DLExistsRA('grippers', DLManipulator))

# Description logical concept modelling a robot with only one manipulator
DLSingleManipulatorRobot  = DLConjunction(DLRobot, DLExistsRA('gripper', DLManipulator))

# Description logical concept modelling any thing which is capable of manipulating things
DLManipulationCapable     = DLDisjunction(DLMultiManipulatorRobot, DLSingleManipulatorRobot)

# Description logical concept modelling an observer
DLObserver = DLExistsRA('frame_of_reference', DLTransform)

DLMesh     = DLConjunction(DLShape, DLExistsRA('radius', DLScalar))

# Description logical concept modelling a physical thing
DLPhysicalThing  = DLExistsRA('pose', DLTransform)

# Description logical concept modelling a hard-surface object
DLRigidObject    = DLConjunction(DLPhysicalThing, DLShape, DLExistsRA('mass', DLScalar))

# Description logical concept modelling an object with a probabilistic location and rotation
DLProbabilisticThing = DLConjunction(DLPhysicalThing, DLExistsRA('pose_cov', DLCovarianceMatrix))


# Physical thing with its pose represented as gmm
DLPhysicalGMMThing = DLExistsRA('gmm', DLGMMPoseComponent)

DLRigidGMMObject   = DLConjunction(DLPhysicalGMMThing, DLShape, DLExistsRA('mass', DLScalar))


# Description logical concept modelling an object which is composed of other objects
DLCompoundObject = DLConjunction(DLRigidObject, DLExistsRA('subObject', DLRigidObject))

# List of all concepts in this file
BASIC_TBOX_LIST = [DLString, DLScalar, DLVector, DLPoint, DLRotation, DLTranslation, DLTransform, DLStamp, DLSymbolic, DLRobot, DLGripper, DLJointState, DLBodyPosture, DLManipulator, DLMultiManipulatorRobot, DLSingleManipulatorRobot, DLManipulationCapable, DLSphere, DLPartialSphere, DLRectangle, DLCube, DLCylinder, DLShape, DLIded, DLObserver]

# Add convenient aliases
BASIC_TBOX = BASIC_TBOX_LIST + [('Sphere', DLSphere),
                                ('PartialSphere', DLPartialSphere),
                                ('Rectangle', DLRectangle),
                                ('Cube', DLCube),
                                ('Cylinder', DLCylinder),
                                ('Shape', DLShape),
                                ('PhysicalThing', DLPhysicalThing),
                                ('ProbabilisticThing', DLProbabilisticThing),
                                ('RigidObject', DLRigidObject),
                                ('CompoundObject', DLCompoundObject),
                                ('Ided', DLIded),
                                ('Manipulator', DLManipulator),
                                ('ManipulationCapable', DLManipulationCapable),
                                ('Observer', DLObserver),
                                DLInclusion(DLCamera, DLPhysicalThing),
                                DLInclusion(DLManipulator, DLPhysicalThing)
                                ]