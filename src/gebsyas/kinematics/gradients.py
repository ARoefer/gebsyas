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