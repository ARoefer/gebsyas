import rospy
from gebsyas.actions import Action
from gebsyas.generic_motion_action import GenericMotionAction
from gebsyas.grasp_action import GraspAction, LetGoAction
from gebsyas.predicates import IsGrasped, Graspable, Above, PointingAt, ClearlyPerceived, IsControlled,InPosture
from gebsyas.data_structures import StampedData
from gebsyas.dl_reasoning import DLTop, DLExistsRA, DLDisjunction, DLRigidObject, DLRigidGMMObject
from gebsyas.observation_controller import ObservationController, run_observation_controller
from gebsyas.predicate_state_action import PSAction
from gebsyas.sensors import TopicSensor
from gebsyas.trackers import SearchObjectTracker
from gebsyas.utils import Blank, bb
from giskardpy.symengine_wrappers import translation3
from gop_gebsyas_msgs.msg import SearchObjectList as SearchObjectListMsg
from gop_gebsyas_msgs.msg import SearchResult     as SearchResultMsg
from gop_gebsyas_msgs.msg import StringArray      as StringArrayMsg

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
                context.log('Execution for grasping {} finished with {}'.format(Id, feedback))
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
        self.pub_found_id = rospy.Publisher('/search_result', SearchResultMsg, queue_size=1, tcp_nodelay=1)
        self.pub_search_request = rospy.Publisher('/search_request', StringArrayMsg, queue_size=1, tcp_nodelay=1)

    def execute(self, context):
        done = False
        filter = self.searched_ids
        context.display.begin_draw_cycle()
        self.initialize_search(context)
        data_state = context.agent.get_data_state()
        predicate_state = context.agent.get_predicate_state()
        robot = context.agent.robot
        #delivery_box = bb(width=0.33, height=0.14, length=0.31, mass=0.5, pose=robot.get_fk_expression('map', 'box_link'))

        #stupid_thing = bb(radius=0.035, height=0.2, mass=1.0, pose=(robot.gripper.pose * translation3(0,0,0.1)))
        print(context.agent.memory.keys())
        #posture = context.agent.memory['basic_stance']

        #self.execute_subaction(context, GenericMotionAction(InPosture.fp(context, robot.state.data, posture)))

        dl_rigid_objects = DLDisjunction(DLRigidObject, DLRigidGMMObject)

        observation_controller = ObservationController(context,
                                                data_state.dl_data_iterator(dl_rigid_objects),
                                                set(),
                                                3,
                                                context.log)
        observation_controller.init(context, 
                                    robot.get_fk_expression('map', 'base_link') * translation3(0.1, 0, 0),
                                    robot.camera)
        data_state.register_new_data_cb(dl_rigid_objects, observation_controller.add_new_obstacle)
        self.set_search_request(self.searched_ids)

        while not rospy.is_shutdown() and not len(self.searched_ids) == 0:
            context.log('New searched ids: {}'.format(', '.join(self.searched_ids)))
            observation_controller.reset_search()
            b_found_object, m_lf, t_log = run_observation_controller(robot, observation_controller, context.agent, 0.02, 0.9)
            if b_found_object:
                found_obj = observation_controller.get_current_object()
                found_id = found_obj.id
                result_msg = SearchResultMsg()
                result_msg.id = int(''.join([c for c in found_id if c.isdigit()]))
                result_msg.grasp = len({i for i in self.searched_ids if i in found_id}) > 0
                self.pub_found_id.publish(result_msg)
                
                context.log('Found thing is called {}. Pose:\n{}'.format(found_id, str(sorted(found_obj.gmm)[-1].pose)))

                if 'table' in found_id:
                    print('found a table')
                    found_obj.pose = sorted(found_obj.gmm)[-1].pose
                    del found_obj.gmm
                    found_obj.pose[2, 3] = found_obj.height * 0.5
                
                data_state.insert_data(StampedData(rospy.Time.now(), found_obj), found_id)

                if result_msg.grasp:
                    # self.execute_subaction(context, GenericMotionAction(Graspable.fp(context, robot.gripper, found_obj), {found_id}))
                    # self.execute_subaction(context, GraspAction(robot, robot.gripper, found_obj))

                    # context.log('IsControlled({}) = {}'.format(found_id, predicate_state.evaluate(context, IsControlled, (found_id, ))))
                    # #TODO: Drop into box
                    # self.execute_subaction(context, GenericMotionAction(Above.fp(context, data_state[found_id].data.data, delivery_box, context.agent)))
                    # self.execute_subaction(context, LetGoAction(robot, robot.gripper, found_obj))

                    #if self.sim_mode:
                    self.searched_ids = {i for i in self.searched_ids if i not in found_id}
                    #else:
                    #    self.searched_ids.remove(found_id)

                    if len(self.searched_ids) > 0:
                        self.set_search_request(self.searched_ids)

            else:
                context.log('observation controller should never return unsuccessfully unless it was terminated from the outside.')
            #rospy.sleep(0.3)
        data_state.deregister_new_data_cb(dl_rigid_objects, observation_controller.add_new_obstacle)
        observation_controller.stop()
        context.display.render()

    def set_search_request(self, objects):
        msg = StringArrayMsg()
        msg.strings.extend(objects)
        self.pub_search_request.publish(msg)

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
            if len(self.searched_ids) > 0  and max([s in stamped_objects.data.search_object_list[x].id for s in self.searched_ids]):
                sos.search_object_list.append(stamped_objects.data.search_object_list[x])
                sos.weights.append(stamped_objects.data.weights[x])

        self.so_tracker.process_data(StampedData(stamped_objects.stamp, sos))
        self.agent.on_objects_sensed(stamped_objects)
