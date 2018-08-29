import rospy
from gebsyas.actions import Action
from gebsyas.predicates import IsGrasped, Graspable, PointingAt, ClearlyPerceived
from gebsyas.data_structures import StampedData
from gebsyas.dl_reasoning import DLTop, DLExistsRA, DLDisjunction, DLRigidObject, DLRigidGMMObject
from gebsyas.observation_controller import ObservationController, run_observation_controller
from gebsyas.predicate_state_action import PSAction
from gebsyas.sensors import TopicSensor
from gebsyas.trackers import SearchObjectTracker
from gebsyas.utils import Blank
from giskardpy.symengine_wrappers import translation3
from gop_gebsyas_msgs.msg import SearchObjectList as SearchObjectListMsg

class SingleObjectSearchAction(Action):
    def __init__(self, dl_searched_class):
        super(SingleObjectSearchAction, self).__init__('SingleObjectSearch')
        self.dl_searched_class = dl_searched_class

    def execute(self, context):
        done = False
        context.display.begin_draw_cycle()
        while not rospy.is_shutdown() and not done:
            for Id, data in context.agent.data_state.dl_data_iterator(self.dl_searched_class):
                #feedback = self.execute_subaction(context, PSAction({Graspable: {('gripper', Id): True}}))
                feedback = self.execute_subaction(context, PSAction({IsGrasped: {('gripper', Id): True}}))
                context.log('Execution for grapsing {} finished with {}'.format(Id, feedback))
                done = True
                break
            context.log('Known objects: {}'.format(', '.join([Id for Id in context.agent.data_state.dl_iterator(DLTop())])))
            rospy.sleep(0.3)
        context.display.render()


class MultiObjectSearchAndDeliveryAction(Action):
    def __init__(self, searched_ids, delivery_location, sim_mode=False):
        super(MultiObjectSearchAndDeliveryAction, self).__init__('MultiObjectSearch')
        self.searched_ids = set(searched_ids)
        self.delivery_location   = delivery_location
        self.sim_mode = sim_mode

    def execute(self, context):
        done = False
        filter = self.searched_ids
        context.display.begin_draw_cycle()
        self.initialize_search(context)
        data_state = context.agent.get_data_state()
        robot = context.agent.robot
        observation_controller = ObservationController(context,
                                                data_state.dl_data_iterator(DLDisjunction(DLRigidObject, DLRigidGMMObject)),
                                                set(),
                                                3,
                                                context.log)
        observation_controller.init(context, 
                                    robot.get_fk_expression('map', 'base_link') * translation3(0.1, 0, 0),
                                    robot.camera)
        while not rospy.is_shutdown() and not len(self.searched_ids) == 0:
            self.set_search_request(self.searched_ids)
            b_found_object, m_lf, t_log = run_observation_controller(robot, observation_controller, context.agent, 0.02, 0.9)
            if b_found_object:
                self.searched_ids.remove(observation_controller.get_current_object().id)
            else:
                context.log('observation controller should never return unsuccessfully unless it was terminated from the outside.')
            #rospy.sleep(0.3)
        context.display.render()

    def set_search_request(self, objects):
        pass

    def initialize_search(self, context):
        self.agent = context.agent
        self.so_tracker = SearchObjectTracker('searched_objects', self.agent.get_data_state())
        self.agent.add_tracker(self.so_tracker)
        if not self.sim_mode:
            self.search_sensor = TopicSensor(self.so_tracker.process_data, '/searched_objects', SearchObjectListMsg, 1)
            self.agent.add_sensor('searched objects sensor', self.search_sensor)
        else:
            self.search_sensor = TopicSensor(self.on_objects_sensed, '/perceived_prob_objects', SearchObjectListMsg, 12)
            self.agent.add_sensor('object sensor', self.search_sensor)
        self.search_sensor.enable()

    def on_objects_sensed(self, stamped_objects):
        """Callback for a sensed object."""
        sos = Blank()
        sos.search_object_list = []
        sos.weights = []
        for x in range(len(stamped_objects.data.weights)):
            if stamped_objects.data.search_object_list[x].id in self.searched_ids:
                sos.search_object_list.append(stamped_objects.data.search_object_list[x])
                sos.weights.append(stamped_objects.data.weights[x])

        self.so_tracker.process_data(StampedData(stamped_objects.stamp, sos))
        self.agent.on_objects_sensed(stamped_objects)
