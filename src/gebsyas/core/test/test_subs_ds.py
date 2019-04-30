import unittest
from gebsyas.core.subs_ds import ListStructure, Structure, ks_from_obj, to_sym
from gebsyas.utils import bb
from symengine import symbols, Symbol

class TestSubsDS(unittest.TestCase):

    def test_to_sym(self):
        t = ('lol', 'kek', 'foo')
        with self.assertRaises(Exception):
            to_sym(tuple())

        self.assertEqual(str(to_sym(t)), '__'.join(t))
        with self.assertRaises(Exception):
            to_sym((1,None, 'lol'))


    def test_list_structure(self):
        x, y, z = symbols('x y z')

        def update(o, state):
            state[x] = o[0]
            state[y] = o[1]
            state[z] = o[2]

        l = ListStructure('arny', update, x, y, z)
        self.assertIs(x, l[0])
        self.assertIs(y, l[1])
        self.assertIs(z, l[2])
        self.assertIn(x, l.free_symbols)
        self.assertIn(y, l.free_symbols)
        self.assertIn(z, l.free_symbols)

        u_o     = [4, -1, 3] 
        state = {}
        l.rf(u_o, state)

        self.assertEqual(state[x],  4)
        self.assertEqual(state[y], -1)
        self.assertEqual(state[z],  3)

        l2 = l.subs({y: 6})

        self.assertIsNot(l, l2)
        self.assertEqual(6, l2[1])
        self.assertNotIn(y, l2.free_symbols)


    def test_structure(self):
        x, y, z = symbols('x y z')

        def update(o, state):
            state[x] = o.x
            state[y] = o.y
            state[z] = o.z

        s = Structure('arny', update, x=x, y=y, z=z)
        self.assertEqual(len(s), 3)
        self.assertIs(x, s.x)
        self.assertIs(y, s.y)
        self.assertIs(z, s.z)
        self.assertIn(x, s.free_symbols)
        self.assertIn(y, s.free_symbols)
        self.assertIn(z, s.free_symbols)
        self.assertIn('x', s.edges)
        self.assertIn('y', s.edges)
        self.assertIn('z', s.edges)

        u_o     = bb(x=4, y=-1, z=3) 
        state = {}
        s.rf(u_o, state)

        self.assertEqual(state[x],  4)
        self.assertEqual(state[y], -1)
        self.assertEqual(state[z],  3)

        s2 = s.subs({y: 6})

        self.assertIsNot(s, s2)
        self.assertEqual(6, s2.y)
        self.assertNotIn(y, s2.free_symbols)        


    def test_ks_from_obj(self):
        obj1 = bb(x=4, y=-1, z=3, w=bb(u=True, r=22, l='lol'), l=[1,2,3])

        state = {}
        s_obj, _ = ks_from_obj(obj1, ('obj', ), state)

        sx, sy, sz, su, sr, sl, sl0 = symbols('obj__x obj__y obj__z obj__w__u obj__w__r obj__l obj__l__0')

        self.assertIn(sx, state)
        self.assertIn(sy, state)
        self.assertIn(sz, state)
        self.assertIn(sr, state)
        self.assertIn(sl0, state)
        self.assertEqual(state[sx],  4)
        self.assertEqual(state[sy], -1)
        self.assertEqual(state[sz],  3)
        self.assertEqual(state[sr], 22)
        self.assertEqual(state[sl0], 1)

        self.assertNotIn(su, state)
        self.assertNotIn(sl, state)


        obj2 = bb(x=2, y=6, z=3, w=bb(u=False, r=12, l='lol'), l=[7,2,3])

        s_obj.rf(obj2, state)

        self.assertEqual(state[sx],  2)
        self.assertEqual(state[sy],  6)
        self.assertEqual(state[sz],  3)
        self.assertEqual(state[sr], 12)
        self.assertEqual(state[sl0], 7)

        # Checks for transform correctness are missing