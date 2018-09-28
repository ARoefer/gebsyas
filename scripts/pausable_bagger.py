#!/usr/bin/env python
import rosbag
import rospy
import sys
from datetime import datetime
from std_srvs.srv import *


class ROSBagger(object):
    def __init__(self, bag_file=None, *topics):
        self.recording = False
        self.index = 0 
        self.file  = bag_file if bag_file != None else 'rosbag_{}'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
        self.bag = None

        self.services = [rospy.Service('record', SetBool, self.srv_record)]

        self.subscribers = []
        for t in topics:
            def cb(msg):
                if self.recording:
                    self.bag.write(t, msg)

            self.subscribers.append(rospy.Subscriber(t, rospy.AnyMsg, callback=cb))

    def record(self):
        if self.bag == None:
            self.bag = rosbag.Bag('{}_{}.bag'.format(self.file, self.index), 'w')
        self.recording = True

    def stop(self):
        self.recording = False
        if self.bag != None:
            self.bag.close()
            self.bag = None

    def shutdown(self):
        self.stop()
        for srv in self.services:
            srv.shutdown()
        for sub in self.subscribers:
            sub.unregister()

    def srv_record(self, req):
        if req.data and not self.recording:
            self.record()
        elif not req.data and self.recording:
            self.stop()
        res = SetBoolResponse()
        res.success = True
        return res


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Name at least one topic to record or type -h for help.')
        exit(0)

    file_name = None
    if sys.argv[1][0] == '-':
        if sys.argv[1] == '-h':
            print('Usage: [-b FILE_NAME] TOPICS')
            exit(0)
        elif sys.argv[1] == '-b':
            if len(sys.argv) < 3:
                print('-b requires a file name')
                exit(0)
            file_name = sys.argv[2]
            topics = sys.argv[3:]
        else:
            print('Unknown option "{}". Use "-h" for help.'.format(sys.argv[1]))
            exit(0)
    else:
        topics = sys.argv[1:]

    if len(topics) == 0:
        print('Name at least one topic to record.')
        exit(0)

    rospy.init_node('pausable_bagger')

    bagger = ROSBagger(file_name, *topics)

    while not rospy.is_shutdown():
        pass

    bagger.shutdown()