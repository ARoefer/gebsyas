

class LPNode(object):
    def __init__(self, lp):
        self.lp = lp
        self.children = {}

    def __contains__(self, key):
        return key in self.children

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, item):
        self.children[key] = item


class KinematicState(object):
    def __init__(self, data_state):
        self.data_state = data_state
        self.lp_tree    = {}
        self.hc_set     = {}
        self.hp_set     = set()

