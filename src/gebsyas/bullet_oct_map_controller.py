import rospy
from symengine import eye
from gebsyas.bullet_based_controller import InEqBulletController
from gop_gebsyas_msgs.msg import OctTreeBoxes as OctTreeBoxesMsg

class InEqBulletOctMapController(InEqBulletController):


    def init(self, soft_constraints):
        super(InEqBulletOctMapController, self).init(soft_constraints)
        self.cubes = set()
        self.oct_sub = rospy.Subscriber('/oct_cubes', OctTreeBoxesMsg, self.cb_oct_map, queue_size=1)
        self.extents = {}
        self.pose = eye(4)

    def cb_oct_map(self, msg):
        for x in range(len(msg.positions)):
            center  = msg.positions[x]
            coords  = (center.x, center.y, center.z)

            if coords not in self.cubes:
                extents = msg.extents[x]
                extents = (extents.x, extents.y, extents.z, 1)

                if extents[0] not in self.extents:
                    self.extents[extents[0]] = set()
                self.extents[extents[0]].add(coords)
                self.cubes.add(coords)

                self.simulator.create_box(extents[:3], coords, mass=0)


    def draw_tie_in(self):
        for extent, coords in self.extents:
            self.visualizer.draw_cube_batch('oct_map', self.pose, extent, coords)

    def stop(self):
        self.oct_sub.unregister()
        super(InEqBulletOctMapController, self).stop()