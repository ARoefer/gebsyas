import re
from collections import namedtuple

# query_things_of_X = 'What things do you know which are'
# query_what_is_X   = 'What is X?'

# if inp[:len(query_things_of_X)] == query_things_of_X and inp[-1:]:
# 	inp = inp[len(query_things_of_X):-1]

# Unary operator
UnaryOp = namedtuple('UnaryOp', ['op', 'a'])

# Binary operator
BinaryOp = namedtuple('BinaryOp', ['op', 'a', 'b'])

# Function call
Function = namedtuple('Function', ['name', 'args'])


def parse_bool_expr(string):
	"""Top level parsing rule. Returns expression and unparsed remainder of string."""
	a, remainder = parse_bool_prefix(string)
	string = remainder.lstrip()

	if string[:4] == 'and ':
		b, remainder = parse_bool_expr(string[4:])
		return BinaryOp('and', a, b), remainder
	elif string[:3] == 'or ':
		b, remainder = parse_bool_expr(string[4:])
		return BinaryOp('or', a, b), remainder

	return a, string

def parse_bool_prefix(string):
	"""Prefix parsing rule. Returns expression and unparsed remainder of string."""
	string = string.lstrip()
	if string[:4] == 'not ':
		a, remainder = parse_bool_atom(string[4:])
		return UnaryOp('not', a), remainder
	else:
		return parse_bool_atom(string)

def parse_bool_atom(string):
	"""Atomic parsing rule. Returns expression and unparsed remainder of string."""
	string = string.lstrip()

	if string[:5] == 'True ' or string == 'True':
		return True, string[5:]
	elif string[:6] == 'False ' or string == 'False':
		return False, string[6:]
	else:
		name, remainder = parse_name(string)
		string = remainder
		if len(string) > 0 and string[0] == '(':
			args, remainder = parse_homogenous_list(string[1:], parse_path)
			string = remainder.lstrip()
			if string[0] != ')':
				raise Exception('Expected \')\'')
			return Function(name, args), string[1:]
		return name, string


def parse_homogenous_list(string, sub_parser):
	"""List parsing rule. Returns tuple of expressions and unparsed remainder of string."""
	out = []
	while True:
		value, remainder = sub_parser(string)
		string = remainder.lstrip()
		out.append(value)
		if len(string) == 0 or string[0] != ',':
			break
		string = string[1:]
	return tuple(out), string

def parse_path(string):
	"""Path parsing rule. Returns string and unparsed remainder of string."""
	string = string.lstrip()
	name, remainder = parse_name(string)
	string = remainder
	if len(string) > 0 and string[0] == '/':
		subpath, remainder = parse_path(string[1:])
		return '{}/{}'.format(name, subpath), remainder
	return name, string

def parse_name(string):
	"""Name parsing rule. Returns string and unparsed remainder of string."""
	string = string.lstrip()
	m = re.search('^[a-zA-Z][a-zA-Z0-9_]*', string)
	if m is None:
		raise Exception('Pattern match for name "{}" returned None!'.format(string))
	return m.group(0), string[len(m.group(0)):]


def normalize(expr):
	"""Normalization function, which propagates all negations inwards."""
	t = type(expr)
	if t == UnaryOp and expr.op == 'not':
		ti = type(expr.a)
		if ti == UnaryOp and expr.a.op == 'not':
			return normalize(expr.a.a)
		elif ti == BinaryOp:
			if expr.a.op == 'and':
				return normalize(BinaryOp('or', UnaryOp('not', expr.a.a), UnaryOp('not', expr.a.b)))
			elif expr.a.op == 'or':
				return normalize(BinaryOp('and', UnaryOp('not', expr.a.a), UnaryOp('not', expr.a.b)))
	elif t == BinaryOp:
		return BinaryOp(expr.op, normalize(expr.a), normalize(expr.b))

	return expr