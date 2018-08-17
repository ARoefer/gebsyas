#!/usr/bin/env python
from gebsyas.inference_graph import InferenceGraph
from gebsyas.predicates import *
from gebsyas.dl_reasoning import *

DLCannedBeverage = DLAtom('CannedBeverage')
DLCup = DLAtom('Cup')
DLLaptop = DLAtom('Laptop')
DLMonitor = DLAtom('Monitor')
DLPen = DLAtom('Pen')
DLScrewdriver = DLAtom('Screwdriver')
DLCoffeeMachine = DLAtom('CoffeeMachine')
DLCoffeePowderPackage = DLAtom('CoffeePowderPackage')
DLCoffeeFilter = DLAtom('CoffeeFilter')
DLTeabag = DLAtom('Teabag')
DLSugarCube = DLAtom('SugarCube')
DLBowl = DLAtom('Bowl')
DLJar = DLAtom('Jar')
DLTable = DLAtom('Table')
DLCounter = DLAtom('Counter')
DLShelf = DLAtom('Shelf')
DLRoom = DLAtom('Room')
DLKitchen = DLAtom('Kitchen')
DLOffice = DLAtom('Office')
DLWorkshop = DLAtom('Workshop')
DLHallway = DLAtom('Hallway')


if __name__ == '__main__':
    concepts = [DLCannedBeverage,
                DLCup,
                DLLaptop,
                DLMonitor,
                DLPen,
                DLScrewdriver,
                DLCoffeeMachine,
                DLCoffeePowderPackage,
                DLCoffeeFilter,
                DLTeabag,
                DLSugarCube,
                DLBowl,
                DLJar,
                DLTable,
                DLCounter,
                DLShelf,
                DLRoom,
                DLKitchen,
                DLOffice,
                DLWorkshop,
                DLHallway]
    tbox = PREDICATE_TBOX + concepts + [DLInclusion(c, DLRigidObject) for c in concepts]
    reasoner = Reasoner(tbox, {})
    
    # DLCannedBeverage = reasoner.get_expanded_concept(DLAtom('CannedBeverage'))
    # DLCup = reasoner.get_expanded_concept(DLAtom('Cup'))
    # DLLaptop = reasoner.get_expanded_concept(DLAtom('Laptop'))
    # DLMonitor = reasoner.get_expanded_concept(DLAtom('Monitor'))
    # DLPen = reasoner.get_expanded_concept(DLAtom('Pen'))
    # DLScrewdriver = reasoner.get_expanded_concept(DLAtom('Screwdriver'))
    # DLCoffeeMachine = reasoner.get_expanded_concept(DLAtom('CoffeeMachine'))
    # DLCoffeePowderPackage = reasoner.get_expanded_concept(DLAtom('CoffeePowderPackage'))
    # DLCoffeeFilter = reasoner.get_expanded_concept(DLAtom('CoffeeFilter'))
    # DLTeabag = reasoner.get_expanded_concept(DLAtom('Teabag'))
    # DLSugarCube = reasoner.get_expanded_concept(DLAtom('SugarCube'))
    # DLBowl = reasoner.get_expanded_concept(DLAtom('Bowl'))
    # DLJar = reasoner.get_expanded_concept(DLAtom('Jar'))
    # DLTable = reasoner.get_expanded_concept(DLAtom('Table'))
    # DLCounter = reasoner.get_expanded_concept(DLAtom('Counter'))
    # DLShelf = reasoner.get_expanded_concept(DLAtom('Shelf'))
    # DLRoom = reasoner.get_expanded_concept(DLAtom('Room'))
    # DLKitchen = reasoner.get_expanded_concept(DLAtom('Kitchen'))
    # DLOffice = reasoner.get_expanded_concept(DLAtom('Office'))
    # DLWorkshop = reasoner.get_expanded_concept(DLAtom('Workshop'))
    # DLHallway = reasoner.get_expanded_concept(DLAtom('Hallway'))

    ig = InferenceGraph(reasoner)

    ig.connect_nodes(DLCup, DLCounter, OnTop)
    ig.connect_nodes(DLCup, DLTable, OnTop)
    ig.connect_nodes(DLCup, DLKitchen, Inside)
    ig.connect_nodes(DLCup, DLOffice, Inside)
    ig.connect_nodes(DLCup, DLWorkshop, Inside)

    ig.connect_nodes(DLTable, DLOffice, Inside)
    ig.connect_nodes(DLPen, DLLaptop, NextTo)

    ig.connect_nodes(DLCannedBeverage, DLCounter, OnTop)
    ig.connect_nodes(DLCannedBeverage, DLKitchen, Inside)
    ig.connect_nodes(DLCannedBeverage, DLWorkshop, Inside)
    ig.connect_nodes(DLCannedBeverage, DLOffice, Inside)
    ig.connect_nodes(DLLaptop, DLTable, OnTop)
    ig.connect_nodes(DLLaptop, DLKitchen, Inside)
    ig.connect_nodes(DLLaptop, DLOffice, Inside)
    ig.connect_nodes(DLLaptop, DLWorkshop, Inside)
    ig.connect_nodes(DLPen, DLTable, OnTop)
    ig.connect_nodes(DLCoffeeMachine, DLCounter, OnTop)
    ig.connect_nodes(DLCoffeePowderPackage, DLCoffeeMachine, NextTo)
    ig.connect_nodes(DLCoffeePowderPackage, DLCoffeeMachine, OnTop)
    ig.connect_nodes(DLCoffeeMachine, DLKitchen, Inside)

    ig.draw_graph()

    known_objects = {DLKitchen, DLOffice}
    desired_objects = {DLCup, DLLaptop}

    indicators = ig.get_indicators(known_objects, desired_objects)

    print('Indicators:\n  {}'.format('\n  '.join([str(n.concept) for n in indicators])))