import rospy

from navigation_msgs.msg import NavToPoseActionGoal as ATPGoalMsg
from navigation_msgs.msg import NavToPoseActionResult as ATPResultMsg
from actionlib_msgs.msg import GoalID as GoalIDMsg
from actionlib_msgs.msg import GoalStatus as GoalStatusMsg

class ExternalNavigationClient(object):
    def __init__(self, prefix='nav_to_pose'):
        self.pub_set_nav_goal = rospy.Publisher('/{}/goal'.format(prefix), ATPGoalMsg, queue_size=0, tcp_nodelay=True)
        self.pub_cancel       = rospy.Publisher('/{}/cancel'.format(prefix), GoalIDMsg, queue_size=0, tcp_nodelay=True)
        self.sub_nav_result   = rospy.Subscriber('/{}/result'.format(prefix), ATPResultMsg, self.__handle_nav_result, queue_size=1)

    def __handle_nav_result(self, msg):
        if msg.status.status == GoalStatusMsg.SUCCEEDED or msg.status.status == GoalStatusMsg.PREEMPTED:
            if msg.status.status == GoalStatusMsg.SUCCEEDED:
                self.handle_success()
            else:
                self.handle_preemption()
        elif msg.status.status == GoalStatusMsg.ABORTED:
            self.handle_failure()

    def set_goal(self, x, y, theta):
        msg = ATPGoalMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.goal.pose.x = x
        msg.goal.pose.y = y
        msg.goal.pose.theta = theta
        self.pub_set_nav_goal.publish(msg)

    def cancel(self):
        self.pub_cancel.publish(GoalIDMsg())

    def handle_success(self):
        pass

    def handle_preemption(self):
        pass

    def handle_failure(self):
        pass

class BlockingNavigationClient(ExternalNavigationClient):
    def __init__(self, prefix='nav_to_pose'):
        super(BlockingNavigationClient, self).__init__(prefix)
        self._waiting = False

    def set_goal(self, x, y, theta):
        super(BlockingNavigationClient, self).set_goal(x, y, theta)

        self._waiting = True
        while not rospy.is_shutdown() and self._waiting:
            pass

    def handle_success(self):
        self._waiting = False

    def handle_preemption(self):
        self._waiting = False

    def handle_failure(self):
        self._waiting = False
        