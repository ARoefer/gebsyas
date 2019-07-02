import giskardpy.symengine_wrappers as spw

TYPE_UNKNOWN  = 0
TYPE_POSITION = 1
TYPE_VELOCITY = 2
TYPE_ACCEL    = 3
TYPE_EFFORT   = 4
TYPE_SUFFIXES = {'_p': TYPE_POSITION, 
                 '_v': TYPE_VELOCITY, 
                 '_a': TYPE_ACCEL,
                 '_e': TYPE_EFFORT}
TYPE_SUFFIXES_INV = {v: k for k, v in TYPE_SUFFIXES.items()}

def get_symbol_type(symbol):
    return TYPE_SUFFIXES[str(symbol)[-2:]] if str(symbol)[-2:] in TYPE_SUFFIXES else TYPE_UNKNOWN

def get_diff_symbol(symbol):
    s_type = get_symbol_type(symbol)
    if s_type == TYPE_UNKNOWN or s_type == TYPE_EFFORT:
        raise Exception('Cannot generate derivative symbol for {}! The type is {}'.format(symbol, s_type))
    return spw.Symbol('{}{}'.format(str(symbol)[:-2], TYPE_SUFFIXES_INV[s_type + 1]))

def get_int_symbol(symbol):
    s_type = get_symbol_type(symbol)
    if s_type == TYPE_UNKNOWN or s_type == TYPE_POSITION:
        raise Exception('Cannot generate integrated symbol for {}! The type is {}'.format(symbol, s_type))
    return spw.Symbol('{}{}'.format(str(symbol)[:-2], TYPE_SUFFIXES_INV[s_type - 1]))

def is_scalar(expr):
    return type(expr) == int or type(expr) == float or expr.is_Add or expr.is_AlgebraicNumber or expr.is_Atom or expr.is_Derivative or expr.is_Float or expr.is_Function or expr.is_Integer or expr.is_Mul or expr.is_Number or expr.is_Pow or expr.is_Rational or expr.is_Symbol or expr.is_finite or expr.is_integer or expr.is_number or expr.is_symbol


class GradientContainer(object):
    def __init__(self, expr, gradient_exprs=None):
        self.expr         = expr
        self.gradients    = gradient_exprs if gradient_exprs is not None else {}
        self.free_symbols = expr.free_symbols if hasattr(expr, 'free_symbols') else set()
        self.free_diff_symbols = {get_diff_symbol(s) for s in self.free_symbols if get_diff_symbol(s) not in self.gradients}

    def copy(self):
        return GradientContainer(self.expr, self.gradients.copy())

    def __contains__(self, symbol):
        return symbol in self.gradients or symbol in self.free_diff_symbols

    def __getitem__(self, symbol):
        if symbol in self.gradients:
            return self.gradients[symbol]
        elif symbol in self.free_diff_symbols:
            new_term = self.expr.diff(get_int_symbol(symbol))
            self[symbol] = new_term
            return new_term
        else:
            raise Exception('Cannot reproduce or generate gradient terms for variable "{}".\n  Free symbols: {}\n  Free diff symbols: {}'.format(symbol, self.free_symbols, self.free_diff_symbols))

    def __setitem__(self, symbol, expr):
        if symbol in self.free_diff_symbols:
            self.free_diff_symbols.remove(symbol)
        self.gradients[symbol] = expr


    def __neg__(self):
        return GradientContainer(-self.expr, {s: -d for s, d in self.gradients.items()})

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if type(other) == GradientContainer:
            gradients = self.gradients.copy()
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] += d
                else:
                    gradients[s] = d
            return GradientContainer(self.expr + other.expr, gradients)        
        return GradientContainer(self.expr + other, self.gradients.copy())

    def __rsub__(self, other):
        return GradientContainer(other) - self

    def __sub__(self, other):
        if type(other) == GradientContainer:
            gradients = self.gradients.copy()
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] -= d
                else:
                    gradients[s] = -d
            return GradientContainer(self.expr - other.expr, gradients)
        return GradientContainer(self.expr - other, {s: -d for s, d in self.gradients.items()})

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if type(other) == GradientContainer:
            gradients = {s: d * other.expr for s, d in self.gradients.items()}
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] += d * self.expr
                else:
                    gradients[s] = d * self.expr
            return GradientContainer(self.expr * other.expr, gradients)
        return GradientContainer(self.expr * other, {s: d * other for s, d in self.gradients.items()})

    def __iadd__(self, other):
        if type(other) == GradientContainer:
            self.expr += other.expr
            for k, v in other.gradients.items():
                if k in self.gradients:
                    self.gradients[k] += v
                else:
                    self.gradients[k]  = v
        else:
            self.expr += other
            if hasattr(other, 'free_symbols'):
                for f in other.free_symbols:
                    if get_diff_symbol(f) in self.gradients:
                        self.gradients[get_diff_symbol(f)] += self.expr.diff(f)
        return self

    def __isub__(self, other):
        if type(other) == GradientContainer:
            self.expr += other.expr
            for k, v in other.gradients.items():
                if k in self.gradients:
                    self.gradients[k] -= v
                else:
                    self.gradients[k]  = v
        else:
            self.expr -= other
            if hasattr(other, 'free_symbols'):
                for f in other.free_symbols:
                    if get_diff_symbol(f) in self.gradients:
                        self.gradients[get_diff_symbol(f)] -= self.expr.diff(f)
        return self

    def __imul__(self, other):
        if type(other) == GradientContainer:
            self.expr *= other.expr
            for k, v in other.gradients.items():
                if k in self.gradients:
                    self.gradients[k] += v * self.expr
                else:
                    self.gradients[k]  = v * self.expr
        else:
            temp           = self * other
            self.expr      = temp.expr
            self.gradients = temp.gradients
        return self

    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        if type(other) == GradientContainer:
            gradients = {s: d * other.expr for s, d in self.gradients.items()}
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] -= d * self.expr
                else:
                    gradients[s] = -d * self.expr
            return GradientContainer(self.expr / other.expr, {s: d / (other.expr**2) for s, d in gradients.items()})
        return GradientContainer(self.expr / other, {s: d / (other ** 2) for s, d in self.gradients.items()})


    def __pow__(self, other):
        if type(other) == GradientContainer:
            gradients = {s: d * other.expr * (self.expr ** (other.expr - 1)) for s, d in self.gradients.items()}
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] += d * self.expr * spw.log(self.expr) * (self.expr ** (other.expr - 1))
                else:
                    gradients[s] = spw.log(self.expr) * d * (self.expr ** other.expr)
            return GradientContainer(self.expr**other.expr, gradients)
        return GradientContainer(self.expr**other, {s: d * other * (self.expr** (other - 1)) for s, d in self.gradients.items()})

    def __str__(self):
        return '{} ({})'.format(str(self.expr), ', '.join([str(k) for k in self.gradients.keys()]))

GC = GradientContainer

def gradient_from_list(l):
    if len(l) == 0:
        return [], []
    elif type(l[0]) == list:
        return [[x.expr if type(x) == GradientContainer else x for x in r] for r in l], [[x if type(x) == GradientContainer else GradientContainer(x) for x in r] for r in l]
    else:
        return [x.expr if type(x) == GradientContainer else x for x in l], [x if type(x) == GradientContainer else GradientContainer(x) for x in l]

class GradientMatrix(object):
    def __init__(self, expr, gradient_exprs=None):
        if type(expr) == list:
            m_list, self.gradients = gradient_from_list(expr)
            self.expr              = spw.sp.Matrix(m_list)
        else:
            self.expr         = expr
            self.gradients    = [[gradient_exprs[y][x] if type(gradient_exprs[y][x]) == GC else GC(self.expr[y, x]) for x in range(self.expr.ncols())] for y in range(self.expr.nrows())] if gradient_exprs is not None else [[GC(e) for e in expr[x,:]] for x in range(expr.nrows())]

        if len(self.gradients) != self.expr.nrows() or len(self.gradients[0]) != self.expr.ncols():
            raise Exception('Gradient dimensions do not match matrix dimensions!\n Matrix: {}, {}\n Gradient: {}, {}'.format(self.expr.nrows(), self.expr.ncols(), len(self.gradients), len(self.gradients[0])))

        self.free_symbols = self.expr.free_symbols
        self.free_diff_symbols = {get_diff_symbol(s) for s in self.free_symbols if get_diff_symbol(s) not in self.gradients}

    def __contains__(self, symbol):
        return symbol in self.gradients or symbol in self.free_diff_symbols

    def __getitem__(self, idx):
        if type(idx) == int:
            return self.gradients[idx / self.expr.ncols()][idx % self.expr.ncols()]
        elif type(idx) == slice:
            return sum(self.gradients, [])[idx]
        elif type(idx) == tuple:
            return GradientMatrix(self.expr[idx], self.gradients[idx[0]][idx[1]])

    def __setitem__(self, idx, expr):
        if type(idx) == int:
            if type(expr) == GC:
                self.expr[idx] = expr.expr
                self.gradients[idx / self.expr.ncols()][idx % self.expr.ncols()] = expr
            else:
                self.expr[idx] = expr
                self.gradients[idx / self.expr.ncols()][idx % self.expr.ncols()] = GC(expr)
        elif type(idx) == tuple:
            if type(expr) == GC:
                self.expr[idx] = expr.expr
                self.gradients[idx[0]][idx[1]] = expr
            else:
                self.expr[idx] = expr
                self.gradients[idx[0]][idx[1]] = GC(expr)

    def __len__(self):
        return self.expr.ncols() * self.expr.nrows()

    def __iter__(self):
        return iter(sum(self.gradients, []))

    def __str__(self):
        return '\n'.join(['[{}]'.format(', '.join([str(x) for x in r])) for r in self.gradients])

    @property
    def T(self):
        return GradientMatrix(self.expr.T, [[self.gradients[y][x].copy() 
                                             for y in range(self.expr.nrows())] 
                                             for x in range(self.expr.ncols())])

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if type(other) == GradientMatrix:
            return GradientMatrix(self.expr + other.expr, 
                                 [[a + b for a, b in zip(self.gradients[x], other.gradients[x])] for x in range(len(self.gradients))])
        else:
            return self + GradientMatrix(other)

    def __rsub__(self, other):
        return GradientMatrix(other) - self

    def __sub__(self, other):
        if type(other) == GradientMatrix:
            return GradientMatrix(self.expr - other.expr, 
                                 [[a - b for a, b in zip(self.gradients[x], other.gradients[x])] for x in range(len(self.gradients))])
        else:
            return self - GradientMatrix(other)

    def __mul__(self, other):
        if type(other) == GradientMatrix:
            gradients = [[GC(0) for y in range(other.expr.ncols())] for x in range(self.expr.nrows())]
            for x in range(other.expr.ncols()):
                for z in range(self.expr.nrows()):
                    for y in range(other.expr.nrows()):
                        gradients[z][x] += self.gradients[y][x]
            return GradientMatrix(self.expr * other.expr, gradients)
        elif other.is_Matrix:
            return self * GradientMatrix(other)
        elif is_scalar(other):
            return GradientMatrix(self.expr * other, [[g * other for g in r] for r in self.gradients])
        elif type(other) == GC:
            return GradientMatrix(self.expr * other.expr, [[g * other for g in r] for r in self.gradients])
        else:
            raise Exception('Operation {} * {} is undefined.'.format(type(self), type(other)))

    def __div__(self, other):
        if type(other) == GC:
            return GradientMatrix(self.expr / other.expr, [[g / other for g in r] for r in self.gradients])
        elif is_scalar(other):
            return GradientMatrix(self.expr / other, [[g / other for g in r] for r in self.gradients])
        else:
            raise Exception('Operation {} / {} is undefined.'.format(type(self), type(other)))

GM = GradientMatrix

def sin(expr):
    if type(expr) == GC:
        return GC(spw.sin(expr.expr), {s: spw.cos(expr.expr) * d for s, d in expr.gradients.items()})
    return spw.sin(expr)

def cos(expr):
    if type(expr) == GC:
        return GC(spw.cos(expr.expr), {s: -spw.sin(expr.expr) * d for s, d in expr.gradients.items()})
    return spw.cos(expr)

def tan(expr):
    if type(expr) == GC:
        return GC(spw.tan(expr.expr), {s: d * (1 + spw.tan(expr.expr)**2) for s, d in expr.gradients.items()})
    return spw.tan(expr)

def asin(expr):
    if type(expr) == GC:
        return GC(spw.asin(expr.expr), {s: d / spw.sqrt(1 - expr.expr**2) for s, d in expr.gradients.items()})
    return spw.asin(expr)

def acos(expr):
    if type(expr) == GC:
        return GC(spw.acos(expr.expr), {s: -d / spw.sqrt(1 - expr.expr**2) for s, d in expr.gradients.items()})
    return spw.acos(expr)

def atan(expr):
    if type(expr) == GC:
        return GC(spw.atan(expr.expr), {s: d / (1 + expr.expr**2) for s, d in expr.gradients.items()})
    return spw.atan(expr)

def sinh(expr):
    if type(expr) == GC:
        return GC(spw.sp.sinh(expr.expr), {s: d * spw.sp.cosh(expr.expr) for s, d in expr.gradients.items()})
    return spw.sp.sinh(expr)

def cosh(expr):
    if type(expr) == GC:
        return GC(spw.sp.cosh(expr.expr), {s: d * spw.sp.sinh(expr.expr) for s, d in expr.gradients.items()})
    return spw.sp.cosh(expr)

def tanh(expr):
    if type(expr) == GC:
        return GC(spw.sp.tanh(expr.expr), {s: d * (1 - spw.sp.tanh(expr.expr)**2) for s, d in expr.gradients.items()})
    return spw.sp.tanh(expr)

def asinh(expr):
    if type(expr) == GC:
        return GC(spw.sp.asinh(expr.expr), {s: d / spw.sqrt(expr.expr**2 + 1) for s, d in expr.gradients.items()})
    return spw.sp.asinh(expr)

def acosh(expr):
    if type(expr) == GC:
        return GC(spw.sp.acosh(expr.expr), {s: d / spw.sqrt(expr.expr**2 - 1) for s, d in expr.gradients.items()})
    return spw.sp.acosh(expr)

def atanh(expr):
    if type(expr) == GC:
        return GC(spw.sp.atanh(expr.expr), {s: d / (1 - expr.expr**2) for s, d in expr.gradients.items()})
    return spw.sp.atanh(expr)


def exp(expr):
    if type(expr) == GC:
        return GC(spw.sp.exp(expr.expr), {s: d * spw.sp.exp(expr.expr) for s, d in expr.gradients.items()})
    return spw.exp(expr)
    
def log(expr):
    if type(expr) == GC:
        return GC(spw.log(expr.expr), {s: d / expr.expr for s, d in expr.gradients.items()})
    return spw.log(expr)

def sqrt(expr):
    if type(expr) == GC:
        return GC(spw.sqrt(expr.expr), {s: d / (2 * spw.sqrt(expr.expr)) for s, d in expr.gradients.items()})
    return spw.sqrt(expr)

def abs(expr):
    if type(expr) == GC:
        return GC(spw.fake_Abs(expr.expr), {s: d * expr.expr / spw.sqrt(expr.expr ** 2) for s, d in expr.gradients.items()})
    return spw.fake_Abs(expr)

def norm(v):
    r = 0
    for x in v:
        r += x ** 2
    return sqrt(r)

def cross(u, v):
    if type(u) == GradientMatrix or type(v) == GradientMatrix:
        return GradientMatrix([u[1] * v[2] - u[2] * v[1],
                               u[2] * v[0] - u[0] * v[2],
                               u[0] * v[1] - u[1] * v[0], 0])
    return spw.sp.Matrix([u[1] * v[2] - u[2] * v[1],
                      u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0], 0])

def translation3(x, y, z, w=1):
    a = [x, y, z, w]
    if max([type(v) == GC for v in a]):
        return GradientMatrix(spw.translation3(*[v.expr if type(v) == GC else v for v in a]), 
                              [([None] * 3) + [x] if type(x) == GC else [None]* 4 for x in a])
    return spw.translation3(x, y, z, w)


def rotation3_rpy(roll, pitch, yaw):
    """ Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    """
    # TODO don't split this into 3 matrices

    a = [roll, pitch, yaw]
    if max([type(v) == GC for v in a]):
        rx = GM([[1, 0, 0, 0],
                 [0, cos(roll), -sin(roll), 0],
                 [0, sin(roll), cos(roll), 0],
                 [0, 0, 0, 1]])
        ry = GM([[cos(pitch), 0, sin(pitch), 0],
                 [0, 1, 0, 0],
                 [-sin(pitch), 0, cos(pitch), 0],
                 [0, 0, 0, 1]])
        rz = GM([[cos(yaw), -sin(yaw), 0, 0],
                 [sin(yaw), cos(yaw), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
        return (rz * ry * rx)
    return spw.rotation3_rpy(roll, pitch, yaw)


def rotation3_axis_angle(axis, angle):
    """ Conversion of unit axis and angle to 4x4 rotation matrix according to:
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    """
    if type(axis) == GM or type(angle) == GC:
        ct = cos(angle)
        st = sin(angle)
        vt = 1 - ct
        m_vt_0 = vt * axis[0]
        m_vt_1 = vt * axis[1]
        m_vt_2 = vt * axis[2]
        m_st_0 = axis[0] * st
        m_st_1 = axis[1] * st
        m_st_2 = axis[2] * st
        m_vt_0_1 = m_vt_0 * axis[1]
        m_vt_0_2 = m_vt_0 * axis[2]
        m_vt_1_2 = m_vt_1 * axis[2]
        return GM([[ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2, 0],
                      [m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2, 0],
                      [-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2], 0],
                      [0, 0, 0, 1]])
    return spw.rotation3_axis_angle(axis, angle)


def rotation3_quaternion(x, y, z, w):
    """ Unit quaternion to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    """
    a = [x, y, z, w]
    if max([type(v) == GC for v in a]):
        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w
        return GM([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                   [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                   [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                   [0, 0, 0, 1]])
    return spw.rotation3_quaternion(x, y, z, w)


def frame3_axis_angle(axis, angle, loc):
    return translation3(*loc) * rotation3_axis_angle(axis, angle)


def frame3_rpy(r, p, y, loc):
    return translation3(*loc) * rotation3_rpy(r, p, y)


def frame3_quaternion(x, y, z, qx, qy, qz, qw):
    return translation3(x, y, z) * rotation3_quaternion(qx, qy, qz, qw)


def inverse_frame(frame):
    if type(frame) == GM:
        inv = GM(sp.eye(4))
        inv[:3, :3] = frame[:3, :3].T
        inv[:3, 3] = -inv[:3, :3] * frame[:3, 3]
        return inv
    return spw.inverse_frame(frame)