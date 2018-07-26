================
Intro to GeBSyAS
================

Hello and welcome to the GeBSyAS system. This page will explain what you've just been welcomed to, what the concepts behind GeBSyAS are and how you can use it.

What is GeBSyAS?
----------------

*GeBSyAS* is short for *Geometry Based Symbolic Agent System* - what a mouthful. The long name is motivated by the overall goal of the system, which is to closely interweave symbolic logic, geometric scene state and motion generation into one entity in the hopes of getting rid of the disconnect between symbolism, geometry and action.

To achieve this, the system uses a whole host of concepts:

- Description Logic
- Agents
- Sensors
- Trackers
- Context
- Constraints
- Constraint Based Predicates
- Symbolic and Non-Symbolic Data
- Actions

Over the course of the next sections all of these concepts will be introduced.


Description Logic
-----------------

Description logic is a logic which is used to represent knowledge. In the Gebsyas environment it is used as an implementation-independent system for classifying data types. This section will briefly go over the basics of description logic, but is by no means a comprehensive introduction to the topic.

Description logic works with *concepts* and *roles*. Concepts can either be atomic, or complex and describe the character of an instance, whereas roles describe the relationships between instances. Let's go over some basic concepts:

- :math:`\top` read as *top*, matches any instance. Equal to **True** in boolean logic.
- :math:`\bot` read as *bottom*, matches no instance. Equal to **False** in boolean logic.
- :math:`C \sqcap D` read as *C and D*, matches all instances which match C and D simultaneously.
- :math:`C \sqcup D` read as *C or D*, matches all instances which match at least C or D.
- :math:`\neg C` read as *not C*, matches all instances which don't match C.
- :math:`\forall R.C` read as *all R-successors are in C*, matches all instances if all their relations of type *R* are to instances matching *C*.
- :math:`\exists R.C` read as *It exists an R-successor in C*, matches all instances which have at least one relation of type *R* to an instance matching *C*.
- :math:`C \sqsubseteq D` read as *all C are D*, behaves as :math:`x \rightarrow y` in boolean logic does.
- :math:`C \equiv D` read as *C and D are equivalent*, behaves as :math:`x \equiv y` in boolean logic does.

The latter two concepts are usually used in the context of a so called *TBox*, usually represented by :math:`\mathcal{T}`, which makes global statements about concepts and thus forms an ontology. Let's look at an example ontology:

.. math::

	\mathcal{T} = \{\\
	Animal &\sqsubseteq \exists is.Alive, \\
	Mammal &\sqsubseteq Animal \sqcap WarmBlooded, \\
	Reptile &\sqsubseteq Animal \sqcap \neg WarmBlooded, \\
	Elephant &\sqsubseteq Mammal \sqcap Grey \sqcap \exists has.Trunk \\
	\}


When we resolve all the statements in this ontology, we come to the conclusion that an :math:`Elephant` implies:

.. math::

	Elephant \sqsubseteq &Animal \sqcap Mammal \sqcap WarmBlooded \\
	&\sqcap \exists is.Alive \sqcap Grey \sqcap \exists has.Trunk


In important question in the field of description logic is the question whether concepts in a TBox are contradictory. If we were to add :math:`Mammal \sqsubseteq Reptile` to our TBox, :math:`Mammal` becomes self-contradictory as it would include :math:`WarmBlooded \sqcap \neg WarmBlooded`.

.. WARNING::
	These contradictions can be difficult to detect, which is why Gebsyas doesn't check for them and puts the onus on the user to not create self-contradictory concepts.

Gebsyas comes with its own implementation of description logical concepts and a matching reasoner that expands concepts based on a TBox. The atomic concepts map directly to Python types and roles are interpreted member-names. With this interpretation, the system is able to directly classify Python objects without any kind of meta-data.
Eventually every piece of data should be classifiable by this logic representation.

.. todo::

	Add link to API-documentation for dl_reasoning.py here


Agents
------

In the Gebsyas world, agents are the highest level structures. If we were to compare them to the human conscious experience, the agent class is the hardware housing the *I*. It abstracts the in- and output of the system to a unified internal data-format and holds the ontology, action-repertoire, current data state and symbolic state of the world. It also provides interfaces for logging and visualization.

An agent can be awake or asleep. By default it activates all its sensors when it is woken up and deactivates them when it's sent to sleep.

.. todo::

	Add link to API-documentation for agent.py here

Sensors
-------

Sensors are wrappers around input operations. Their job is to convert external data into the system internal representation and pass this data along to appropriate callback functions.
By default an implementation wrapping ROS-subscribers is provided.

The following figure shows the main interactions of the components within the overall system.

.. figure:: Gebsyas_Dataflow.png
    :alt: Interaction diagram
    :align: center

.. todo::

	Add link to API-documentation for sensors.py here


Trackers
--------

Trackers track individual pieces of data that can be extracted from a sensor's datastream and perform the necessary update to the system. The point of this tracking system is to handle the update logic independent of the currently instantiated agent or currently executed action.
By default Gebsyas comes equipped with a tracker for joint states, one to track perceived objects and one to track objects whose pose is relative to some other object.

.. todo::

	Add link to API-documentation for trackers.py here


Context
-------

All components which in are in some way reliant on the current state of the system, use a context for their operations. A context holds a reference to an agent, a logger and a visualizer. Actions, planners and predicates require for their operations. The context allows these systems complete access to the current state of the system. It is also a comfortable way to plan in virtual states, without any lower-level system needing to be aware of this fact.

Constraints
-----------

Gebsyas uses the constraint based motion control framework *Giskard* which is currently under development. Giskard uses a special constraint formulation which constrains the instantaneous change which can be made to an expression :math:`e`.
These constraints exist as triples :math:`c = (lb, ub, e)`. To satisfy the constraint, a change :math:`\Delta e` must be achieved so that :math:`lb \leq \Delta e \leq ub`.
We define that the value of :math:`e` satisfies the constraint, when :math:`lb \leq 0 \leq ub`, as this allows for :math:`\Delta e = 0` meaning no change to :math:`e` is necessary.

Constraint Based Predicates
---------------------------

Predicates :math:`P` in Gebsyas are defined as:

.. math::

	P &= (f_P, s_P) \\
	f_P &: \mathcal{C} \times X^n \rightarrow C^m \\
	s_P &= (t \mid t \in \mathcal{T})^n

where :math:`f_P` is a function which generates, given a context and *n* objects, the constraints that model the truth-value for this predicate. The types of the objects are constrained by the type signature :math:`s_P` which consists of description logical types.

To evaluate whether a predicate is true or false, Gebsyas uses function :math:`\phi`

.. math::

	\phi(\mathcal{C}, P, (x_1, \ldots, x_n)) = \bigwedge_{c \in f_P(\mathcal{C}, (x_1, \ldots, x_n))} lb_c \leq 0 \leq ub_c


Symbolic and Non-symbolic data
------------------------------

Gebsyas differentiates between symbolic and non-symbolic data. Non-symbolic data is regular data which can just be read and used in calculations as-is.
Symbolic data might be partially or completely dependent on other data. An example for this kind of data is the position of the robot's endeffector, which is dependent on the robot's joint state.
There are two main motivations for using symbolic data:

- It avoids complicated update cycles for continuously updating the data.
- It implicitly encodes the mechanics of the world into its state.

Especially the last reason is a big argument for symbolic objects. This encoding of the world's mechanics allows the system compute actions to change the systems state.

Symbolic data :math:`Y` is defined in Gebsyas as:

.. math::

	Y &= (y, f_Y, L^n) \\
	f_Y &: X^n \rightarrow X

where :math:`y` is the symbolic data structure, :math:`f_Y` is a conversion function which uses :math:`n` non-symbolic objects to create the current non-symbolic instance of :math:`y`. :math:`L^n` is a tuple of labels, which refer to the :math:`n` objects that should be used by :math:`f_Y`.

.. todo::

	Add link to API-documentation for SymbolicData and numeric_scene_state.py here


Actions
-------

Lastly let's address actions. Action are the active units of the system. All behavior should be implemented in actions. All actions require a context in which they are supposed to operate. Their execution returns :math:`r \in [0,1]` to indicate how successful the execution was. This type of feedback is under reconsideration as is utility is not apparent anymore.

Action Interfaces
`````````````````
The action implementation can not be used directly by the planning system, as actions are runnable components which require actual data for their initialization. To make actions accessible for the planning system, Gebsyas provides *Action Interfaces*. These interfaces define pre- and postconditions in the form of predicates and provide a function to instantiate an action in a context.

.. todo::

	Add link to API-documentations of actions.py here.