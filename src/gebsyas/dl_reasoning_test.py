#!/usr/bin/env python
from gebsyas.dl_reasoning import *
from pprint import pprint

from gebsyas.agent import TBOX as ATBOX
from gebsyas.sensors import TBOX as STBOX
from gebsyas.trackers import TBOX as TTBOX
from gebsyas.predicates import PREDICATE_TBOX

if __name__ == '__main__':

	elephant = DLConjunction(DLAtom('Mammal'), DLAtom('Big'), DLAtom('Grey'))

	tbox = BASIC_TBOX + ATBOX + STBOX + TTBOX + PREDICATE_TBOX
	TBOX_LIST = [DLAtom('Animal'),
				 DLAtom('Mammal'),
				 DLAtom('Reptile'),
				 DLInclusion(DLAtom('Mammal'), DLExistsRA('is', DLAtom('WarmBlooded'))),
				 DLInclusion(DLAtom('Animal'), DLExistsRA('is', DLAtom('Alive'))),
				 ('Reptile', DLAtom('Animal')),
				 ('Mammal', DLAtom('Animal')),
				 ('Elephant', elephant)]

	print('Shape implies:\n  {}'.format('\n  '.join([str(x) for x in DLShape.implies])))


	reasoner = Reasoner(tbox, {})
	for atom, implies in sorted(reasoner.tbox.items()):
		if type(implies) == DLConjunction:
			print('{}:\n  {}'.format(str(atom), '\n  '.join([str(c) for c in implies.concepts])))
			#print('Implied by:\n  {}'.format('\n  '.join(['{}:\n    {}'.format(str(c), '\n    '.join([str(x) for x in c.implied_by])) for c in implies.concepts])))
		else:
			print('{}:\n  {}'.format(str(atom), str(implies)))
