import numpy as np

from kineverse.gradients.diff_logic    import create_pos, get_diff_symbol
from kineverse.gradients.gradient_math import point3, vector3, x_of, norm, cross, tan, GC
from kineverse.motion.min_qp_builder   import TypedQPBuilder as TQPB \
                                              GeomQPBuilder  as GQPB \
                                              SoftConstraint as SC
from kineverse.motion.integrator       import CommandIntegrator
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

class Camera(object):
    def __init__(self, pose, fov, aspect_ratio=1.0):
        self.pose         = pose
        self.fov          = fov
        self.aspect_ratio = aspect_ratio

class GaussianComponent(object):
    def __init__(self, obj_id, gc_id, position, covariance, nav_distance):
        self.obj_id   = obj_id
        self.id       = gc_id
        self.position = position
        self.nav_distance = nav_distance
        if type(covariance) is list or type(covariance) is tuple:
            if len(covariance) != 36:
                raise Exception('Failed to create covariance matrix from list. The list should have exactly 36 elements, but has {}'.format(len(covariance)))
            self.covariance = np.array(covariance).reshape((6, 6))
        else:
            self.covariance = covariance


max_view_cones = 10

class GaussianInspector(object):
    def __init__(self, km, camera, sym_loc_x, sym_loc_y, sym_loc_a, min_obs_fraction=0.2, max_obs_fraction=1.0, permitted_vc_overlap=0.2, visualizer=None):
        self.km     = km
        self.examined_view_cones = {}
        self.camera = camera
        self.state  = {}
        self.gc     = None

        localization_vars = {sym_loc_x, sym_loc_y, sym_loc_a}
        if len(camera.pose.free_symbols.intersection(localization_vars)) < 3:
            raise Exception('Provided localization symbols are not part of the canera pose equation. Missing symbols:\n {}'.format('\n '.join([str(s) for s in localization_vars.difference(camera.pose.free_symbols)])))

        self.sym_loc_x = sym_loc_x 
        self.sym_loc_y = sym_loc_y 
        self.sym_loc_a = sym_loc_a

        #self.visualizer = visualizer if visualizer is not None else ROSBPBVisualizer('/debug_vis', 'map')

        self.gauss_position = point3(*[create_pos('gaussian_{}'.format(d)) for d in ['x','y','z']])
        self.sym_gaussian_x, self.sym_gaussian_y, self.sym_gaussian_z = self.gauss_position[:3]
        self.sym_obj_radius = create_pos('object_radius')
        self.sym_eigen_vecs = [[create_pos('eigen_{}_{}'.format(x, d)) for d in ['x','y','z']] for x in range(3)]

        self.eigen_vecs = [vector3(*syms) for syms in self.sym_eigen_vecs]

        # Looking along axes of uncertainty
        view_dir = x_of(camera.pose) * vector3(1,0,0)

        sc_eigen_vecs = {'look along eigen {}'.format(x):
                            SC(-norm(cross(view_dir, e)),
                               -norm(cross(view_dir, e)),
                                norm(e),
                                GC(norm(cross(view_dir, e)))) for x, e in enumerate(self.eigen_vecs)}

        # Taking a good observation distance
        obj_dist = norm(pos_of(self.camera.pose) - self.gauss_position)
        if self.camera.aspect_ratio >= 1.0:
            cone_radius = tan(self.camera.fov / self.camera.aspect_ratio * 0.5)
        else:
            cone_radius = tan(self.camera.fov * 0.5)

        projected_radius = self.sym_obj_radius / (obj_dist * cone_radius + 1e-5) # Add a little offset to avoid div by zero

        camera_to_obj    = self.gauss_position - pos_of(self.camera.pose)
        camera_to_obj   /= norm(camera_to_obj) + 1e-3
        view_align       = dot(view_dir, camera_to_obj)
        sc_obs_distance  = {'observation distance': SC(min_obs_fraction - projected_radius,
                                                       max_obs_fraction - projected_radius, 
                                                       1, 
                                                       GC(projected_radius)),
                            'look at': SC(1 - view_align, 
                                          1 - view_align, 
                                          1, 
                                          GC(view_align))}

        # Avoiding overlap with another view cone
        cone_axis     = vector3(*[create_pos('view_{}'.format(d)) for d in ['x','y','z']])
        sc_avoid_cone = SC(-1, 
                           GC(permitted_vc_overlap - dot(cone_axis, view_dir)), 1, 
                           GC(dot(cone_axis, view_dir)))
        # Generate the derivatives beforehand to save time later
        sc_avoid_cone.upper.do_full_diff()
        sc_avoid_cone.expr.do_full_diff()

        self.cone_axes = [vector3(*[create_pos('view_{}_{}'.format(d, x)) for d in ['x','y','z']]) for x in range(max_view_cones)]
        self.sc_cone_constraints = {'cone constraint {}'.format(x): SC(sc_avoid_cone.lower, 
                                                                       sc_avoid_cone.upper.subs(subs),
                                                                       sc_avoid_cone.weight,
                                                                       sc_avoid_cone.expr.subs(subs)) 
                                    for x, subs in enumerate([{a: b for a, b in zip(cone_axis, axis)} 
                                                               for axis in self.cone_axes])}

        robot_symbols      = self.camera.pose.free_symbols
        controlled_symbols = {get_diff_symbol(s) for s in robot_symbols}
        all_sybols         = robot_symbols.union(controlled_symbols)
        self.world         = self.km.get_active_geometry(all_symbols)
        hard_constraints   = self.km.get_constraints_by_symbols(all_symbols)

        # Generating full constraint set
        to_remove = set()
        controlled_values = {}
        for k, c in constraints.items():
            if type(c.expr) is spw.Symbol and c.expr in controlled_symbols:
                weight = 0.01 if c.expr != roomba_joint.lin_vel and c.expr != roomba_joint.ang_vel else 0.2
                controlled_values[str(c.expr)] = ControlledValue(c.lower, c.upper, c.expr, weight)
                to_remove.add(k)

        for s in controlled_symbols:
            if str(s) not in controlled_values:
                controlled_values[str(s)] = ControlledValue(-1e9, 1e9, s, 0.01)

        constraints = {k: c for k, c in constraints.items() if k not in to_remove}

        soft_constraints = self.sc_eigen_vecs.copy()
        soft_constraints.update(self.sc_eigen_vecs)
        soft_constraints.update(self.sc_obs_distance)
        soft_constraints.update(self.sc_cone_constraints)
        self.collision_free_solver = TQPB(constraints, soft_constraints, controlled_values)
        self.collision_solver      = GQPB(self.world, constraints, soft_constraints, controlled_values, visualizer=visualizer)




    def set_gaussian_component(self, gc, obj_radius=0.2):    
        # Compute eigen vectors of covariance and write them to the state
        w, v = np.linalg.eig(gc.covariance[:3, :3])
        pos_eig = w * v
        for x, v in enumerate(self.eigen_vecs):
            for y, s in enumerate(v[:3]):
                self.state[s] = pos_eig[y, x]

        # Set position
        for x, s in enumerate(self.gauss_position[:3]):
            self.state[s] = gc.position[x]

        # Initialize view cone memory if not already present
        if gc.obj_id not in self.examined_view_cones:
            self.examined_view_cones[gc.obj_id] = {}

        if gc.id not in self.examined_view_cones[gc.obj_id]:
            self.examined_view_cones[gc.obj_id][gc.id] = []
        else:
            l = self.examined_view_cones[gc.obj_id][gc.id]
            if len(l) < max_view_cones:
                l += [vector3(0,0,0)] * (max_view_cones - len(l))
            for sym, val in zip(self.cone_axes, l):
                self.state[sym[0]] = val[0]
                self.state[sym[1]] = val[1]
                self.state[sym[2]] = val[2]

    # Return sorted list of resolved camera 6d poses.
    def get_view_poses(self, num_iterations=100, int_factor=0.25, samples=20, spread=3.0):
        if self.gc is None:
            raise Exception('Set a gaussion component before trying to solve for a view pose.')

        # Initialize samples
        # Create batch of integrators
        # Update batch in parallel
        # Rate results by their last errors
        # Add best to the explored view cones
        # Return sorted, rated results as 6D camera poses


