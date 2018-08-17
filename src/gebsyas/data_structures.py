from collections import namedtuple


StampedData = namedtuple('StampedData', ['stamp', 'data'])

# Class structure for symbolic data. Consists of the data itself, a conversion function and the arguments needed for the conversion.
SymbolicData = namedtuple('SymbolicData', ['data', 'f_convert', 'args'])

JointState = namedtuple('JointState', ['position', 'velocity', 'effort'])

LocalizationPose = namedtuple('LocalizationPose', ['x', 'y', 'z', 'az'])