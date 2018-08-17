import rospy
from gebsyas.actions import Action
from gebsyas.predicates import IsGrasped, Graspable, PointingAt
from gebsyas.dl_reasoning import DLTop, DLExistsRA
from gebsyas.predicate_state_action import PSAction

class SingleObjectSearchAction(Action):
    def __init__(self, dl_searched_class):
        super(SingleObjectSearchAction, self).__init__('SingleObjectSearch')
        self.dl_searched_class = dl_searched_class

    def execute(self, context):
        done = False
        filter = self.dl_searched_class
        context.display.begin_draw_cycle()
        while not rospy.is_shutdown() and not done:
            for Id, data in context.agent.data_state.dl_data_iterator(self.dl_searched_class):
                feedback = self.execute_subaction(context, PSAction({Graspable: {('gripper', Id): True}}))
                context.log('Execution for grapsing {} finished with {}'.format(Id, feedback))
                done = True
                break
            context.log('Known objects: {}'.format(', '.join([Id for Id in context.agent.data_state.dl_iterator(DLTop())])))
            rospy.sleep(0.3)
        context.display.render()