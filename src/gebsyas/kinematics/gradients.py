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


class GradientContainer(object):
    def __init__(self, expr, gradient_exprs=None):
        self.expr         = expr
        self.gradients    = gradient_exprs if gradient_exprs is not None else {}
        self.free_symbols = expr.free_symbols
        self.free_diff_symbols = {get_diff_symbol(s) for s in self.free_symbols if get_diff_symbol(s) not in self.gradients}

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


    def __sub__(self, other):
        if type(other) == GradientContainer:
            gradients = self.gradients.copy()
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] -= d
                else:
                    gradients[s] = -d
            return GradientContainer(self.expr - other.expr, gradients)
        return GradientContainer(self.expr - other, {s: -d for s, d in self.gradients})

    def __mul__(self, other):
        if type(other) == GradientContainer:
            gradients = {s: d * other.expr for s, d in self.gradients.items()}
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] += d * self.expr
                else:
                    gradients[s] = d * self.expr
            return GradientContainer(self.expr * other.expr, gradients)
        return GradientContainer(self.expr * other, {s: d * other for s, d in self.gradients})

    def __truediv__(self, other):
        if type(other) == GradientContainer:
            gradients = {s: d * other.expr for s, d in self.gradients.items()}
            for s, d in other.gradients.items():
                if s in gradients:
                    gradients[s] -= d * self.expr
                else:
                    gradients[s] = -d * self.expr
            return GradientContainer(self.expr / other.expr, {s: d / (other.expr**2) for s, d in gradients.items()})
        return GradientContainer(self.expr / other, {s: d / (other ** 2) for s, d in self.gradients})


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


def sin(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.sin(expr.expr), {s: spw.cos(expr.expr) * d for s, d in expr.gradients.items()})
    return GradientContainer(spw.sin(expr))

def cos(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.cos(expr.expr), {s: -spw.sin(expr.expr) * d for s, d in expr.gradients.items()})
    return GradientContainer(spw.cos(expr))

def tan(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.tan(expr.expr), {s: d * (1 + spw.tan(expr.expr)**2) for s, d in expr.gradients.items()})
    return GradientContainer(spw.tan(expr))

def asin(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.asin(expr.expr), {s: d / spw.sqrt(1 - expr.expr**2) for s, d in expr.gradients.items()})
    return GradientContainer(spw.asin(expr))

def acos(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.acos(expr.expr), {s: -d / spw.sqrt(1 - expr.expr**2) for s, d in expr.gradients.items()})
    return GradientContainer(spw.acos(expr))

def atan(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.atan(expr.expr), {s: d / (1 + expr.expr**2) for s, d in expr.gradients.items()})
    return GradientContainer(spw.atan(expr))

def sinh(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.sinh(expr.expr), {s: d * spw.cosh(expr.expr) for s, d in expr.gradients.items()})
    return GradientContainer(spw.sinh(expr))

def cosh(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.cosh(expr.expr), {s: d * spw.sinh(expr.expr) for s, d in expr.gradients.items()})
    return GradientContainer(spw.cosh(expr))

def tanh(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.tanh(expr.expr), {s: d * (1 - spw.tanh(expr.expr)**2) for s, d in expr.gradients.items()})
    return GradientContainer(spw.tanh(expr))

def asinh(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.asinh(expr.expr), {s: d / spw.sqrt(expr.expr**2 + 1) for s, d in expr.gradients.items()})
    return GradientContainer(spw.asinh(expr))

def acosh(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.acosh(expr.expr), {s: d / (spw.sqrt(expr.expr - 1) * spw.sqrt(expr.expr + 1)) for s, d in expr.gradients.items()})
    return GradientContainer(spw.acosh(expr))

def atanh(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.atanh(expr.expr), {s: d / (1 - expr.expr**2) for s, d in expr.gradients.items()})
    return GradientContainer(spw.atanh(expr))


def exp(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.exp(expr.expr), {s: d * spw.exp(expr.expr) for s, d in expr.gradients.items()})
    return GradientContainer(spw.exp(expr))
    
def log(expr):
    if type(expr) == GradientContainer:
        return GradientContainer(spw.log(expr.expr), {s: d / expr.expr for s, d in expr.gradients.items()})
    return GradientContainer(spw.log(expr))