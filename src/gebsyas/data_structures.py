from collections import namedtuple


StampedData = namedtuple('StampedData', ['stamp', 'data'])

# Class structure for symbolic data. Consists of the data itself, a conversion function and the arguments needed for the conversion.
SymbolicData = namedtuple('SymbolicData', ['data', 'f_convert', 'args'])

JointState = namedtuple('JointState', ['position', 'velocity', 'effort'])

LocalizationPose = namedtuple('LocalizationPose', ['x', 'y', 'z', 'az'])


class GaussianPoseComponent(object):
    def __init__(self, weight, pose, cov):
        self.weight = weight
        self.pose   = pose
        self.cov    = cov

    def __le__(self, other):
        if type(other) != GaussianPoseComponent:
            raise Exception('<= only defined for GaussianPoseComponent')
        return self.weight <= other.weight

    def __cmp__(self, other):
        if type(other) != GaussianPoseComponent:
            raise Exception('__cmp__ only defined for GaussianPoseComponent')
        return cmp(self.weight, other.weight)