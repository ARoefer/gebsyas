#!/usr/bin/env python

from gebsyas.subs_ds import ListStructure, Structure
from gebsyas.utils import object_to_ks, bb

from pprint import pprint

if __name__ == '__main__':
    o = bb(lol=4.0, kek='Ingnore me', wazzup=bb(trololo=1.2))
    o2 = bb(lol=6.0, kek='Ingnore me', wazzup=bb(trololo=-1.4))

    state = {}
    sym_o = object_to_ks(o, ('o',), state)[0]


    print(sym_o)
    pprint(state)

    sym_o.rf(o2, state)

    pprint(state)