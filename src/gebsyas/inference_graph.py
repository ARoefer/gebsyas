from gebsyas.predicates import *
from graph_tool.all import *


class GraphNode(object):
    def __init__(self, concept):
        self.concept = concept
        self.depth = 0
        self.successor_relation = {} # Arranged as {Predicate: {Concept: (node, weight)}}
        self.successors = {} # Arranged as {Concept: (Node, Predicate, weight)}

    def add_successor(self, node, predicate, weight=1.0):
        if predicate not in self.successor_relation:
            self.successor_relation[predicate] = {}

        self.successor_relation[predicate][node.concept] = (node, weight)
        self.successors[node.concept] = (node, predicate, weight)

        if DLDirectedSpatialPredicate.is_a(predicate):
            node.submerge(self.depth + 1)

    def submerge(self, new_depth):
        if new_depth > self.depth:
            self.depth = new_depth
            for p, p_dict in self.successor_relation.items():
                # Directed -> change depth
                if DLDirectedSpatialPredicate.is_a(p):
                    for c, (n, w) in p_dict.items():
                        n.submerge(self.depth + 1)
                # Non-directed -> depth stays the same (or is equalized?)

    def get_indicators(self, observed_types, indicators):
        """Attempts to find the nodes which are direct predecessors of the observed types.
        These nodes will be added to indicators.

        :type observed_types: set
        :type indicators: set
        """
        if self in indicators:
            return

        for o in observed_types:
            if o in self.successors:
                indicators.add(self)

        for c, (n, p, w) in self.successors.items():
            if n.depth >= self.depth: # Only search deeper into the graph
                n.get_indicators(observed_types, indicators)


    def __hash__(self):
        return hash(self.concept)

    def __eq__(self, other):
        return type(other) == GraphNode and other.concept == self.concept


class InferenceGraph(object):
    def __init__(self, reasoner):
        self.graph = Graph()
        self.edge_predicates = self.graph.new_edge_property('object')
        self.edge_weights    = self.graph.new_edge_property('float')
        self.node_concepts   = self.graph.new_vertex_property('object')
        self.node_depth      = self.graph.new_vertex_property('int32_t')
        self.graph.edge_properties['predicates'] = self.edge_predicates
        self.graph.edge_properties['weights']    = self.edge_weights
        self.graph.vertex_properties['concepts'] = self.node_concepts
        self.graph.vertex_properties['depth']    = self.node_depth

        self.nodes = {}
        self.conceptIdx = {}
        self.reasoner = reasoner

    def add_concept(self, concept):
        if concept not in self.nodes:
            self.nodes[concept] = GraphNode(concept)
            self.conceptIdx[concept] = len(self.conceptIdx)
            v = self.graph.add_vertex()
            self.node_concepts[v] = concept
            self.node_depth[v] = 0

    def connect_nodes(self, conceptA, conceptB, predicate):
        if not DLSpatialPredicate.is_a(predicate):
            raise Exception('This graph is meant for spatial relations inference only. {} is not a spatial predicate'.format(predicate.P))

        if len(predicate.dl_args) != 2:
            raise Exception('Only binary predicates are supported. {} has {:d} parameters.'.format(predicate.P, len(predicate.dl_args)))

        if not self.reasoner.is_subsumed(predicate.dl_args[0], conceptA):
            raise Exception('Type {} is not subsumed by the first parameter type of {}, which is {}'.format(conceptA, predicate.P, predicate.dl_args[0]))

        if not self.reasoner.is_subsumed(predicate.dl_args[0], conceptB):
            raise Exception('Type {} is not subsumed by the second parameter type of {}, which is {}'.format(conceptB, predicate.P, predicate.dl_args[0]))

        if conceptA not in self.nodes:
            self.add_concept(conceptA)

        if conceptB not in self.nodes:
            self.add_concept(conceptB)

        nodeA = self.nodes[conceptA]
        nodeB = self.nodes[conceptB]

        if nodeA is nodeB:
            raise Exception('Connections can not be self-referential. Concepts are:\n  {}\n  {}'.format(str(conceptA), str(conceptB)))

        vA = self.graph.vertex(self.conceptIdx[conceptA])
        vB = self.graph.vertex(self.conceptIdx[conceptB])
        e_ab =self.graph.add_edge(vA, vB)
        self.edge_predicates[e_ab] = predicate
        self.edge_weights[e_ab] = 1.0
        nodeA.add_successor(nodeB, predicate, 1.0)
        
        # If the relation is bi-directional, add a as successor for b too
        if DLNonDirectedSpatialPredicate.is_a(predicate):
            nodeB.add_successor(nodeA, predicate, 1.0)
            e_ba = self.graph.add_edge(vB, vA)
            self.edge_predicates[e_ba] = predicate
            self.edge_weights[e_ba] = 1.0

        self.node_depth[vA] = nodeA.depth
        self.node_depth[vB] = nodeB.depth

    def get_indicators(self, observed_types, desired_types):
        """Returns a set of all nodes, which are directly succeeded by a known type and are themselves successors of at least one known type.

        :type observed_types: set
        :type  desired_types: set
        :rtype: set
        """

        out = set()
        for dt in desired_types:
            if dt in self.nodes:
                self.nodes[dt].get_indicators(observed_types, out)

        return out

    def draw_graph(self, out_png=None):
        graph_draw(self.graph, vertex_text=self.graph.vertex_index, edge_text=self.graph.edge_index, output_size=(1000, 1000), output=out_png)