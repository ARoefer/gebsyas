import rospy
from symengine import eye
from gebsyas.bullet_based_controller import InEqBulletController
from gop_gebsyas_msgs.msg import OctTreeBoxes as OctTreeBoxesMsg
from gop_gebsyas_msgs.msg import AABB as AABBMsg
from gop_gebsyas_msgs.msg import AABBList as AABBListMsg
from gop_gebsyas_msgs.msg import GridUpdate as GridUpdateMsg
from iai_bullet_sim.basic_simulator import vec_add, vec_sub, vec_scale

class InEqBulletOctMapController(InEqBulletController):


    def init(self, soft_constraints):
        super(InEqBulletOctMapController, self).init(soft_constraints)
        self.cubes = set()
        self.aabb_publisher  = rospy.Publisher('/aabbs_to_exlude', AABBListMsg, queue_size=1, tcp_nodelay=True)
        self.focus_publisher = rospy.Publisher('/oct_map_focus', GridUpdateMsg, queue_size=1, tcp_nodelay=True)
        self.extents = {}
        self.pose = eye(4)
        self.oct_sub = rospy.Subscriber('/oct_cubes', OctTreeBoxesMsg, self.cb_oct_map, queue_size=1)

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

    
    def get_cmd(self, nWSR=None):
        cmd = super(InEqBulletOctMapController, self).get_cmd(nWSR)

        focus_msg = GridUpdateMsg()
        focus_msg.header.stamp = rospy.Time.now()

        for link, (m, b) in self.robot.collision_avoidance_links.items():
            aabb = self.bullet_bot.get_AABB(link)
            aabb_msg = AABBMsg()

            center  = vec_scale(vec_add(aabb.min, aabb.max), 0.5)
            extents = vec_sub(aabb.max, aabb.min)
            for n in 'xyz':
                setattr(aabb_msg.center,  n, getattr(center, n))
                setattr(aabb_msg.extents, n, getattr(extents, n) + b)
            focus_msg.ids.append(link)
            focus_msg.grid_resolution.append(b)
            focus_msg.bounding_boxes.append(aabb_msg)

        exclusion_msg = AABBListMsg()
        for Id, bullet_obj in self.allowed_objects.items():
            aabb = bullet_obj.get_AABB()
            aabb_msg = AABBMsg()
            center  = vec_scale(vec_add(aabb.min, aabb.max), 0.5)
            extents = vec_sub(aabb.max, aabb.min)
            for n in 'xyz':
                setattr(aabb_msg.center,  n, getattr(center, n))
                setattr(aabb_msg.extents, n, getattr(extents, n))
            exclusion_msg.ids.append(Id)
            exclusion_msg.bounding_boxes.append(aabb_msg)

        self.aabb_publisher.publish(exclusion_msg)
        self.focus_publisher.publish(focus_msg)
        return cmd

    def draw_tie_in(self):
        for extent, coords in self.extents:
            self.visualizer.draw_cube_batch('oct_map', self.pose, extent, coords)

    def stop(self):
        self.oct_sub.unregister()
        self.aabb_publisher.unregister()
        self.focus_publisher.unregister()
        super(InEqBulletOctMapController, self).stop()