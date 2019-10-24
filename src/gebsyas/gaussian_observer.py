import numpy as np

from giskardpy.exceptions import QPSolverException
from kineverse.gradients.diff_logic    import create_pos, get_diff_symbol, erase_type, Symbol
from kineverse.gradients.gradient_math import point3, vector3, x_of, pos_of, dot, norm, cross, tan, sin, GC, spw
from kineverse.model.paths             import Path
from kineverse.model.geometry_model    import closest_distance_constraint
from kineverse.motion.min_qp_builder   import TypedQPBuilder as TQPB, \
                                              GeomQPBuilder  as GQPB, \
                                              SoftConstraint as SC, \
                                              ControlledValue
from kineverse.motion.integrator       import CommandIntegrator
from kineverse.type_sets               import symbolic_types

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
MAX_RESETS = 50

def subsample_list(l, n_elem):
    step = len(l) / float(n_elem - 1)
    return [l[int(np.round(x * step))] for x in range(n_elem - 1)] + [l[-1]]


class GaussianInspector(object):
    def __init__(self, km, camera, sym_loc_x, sym_loc_y, sym_loc_a, permitted_vc_overlap=0.2, collision_link_paths=[], visualizer=None, tilt_limit_min=0.4, tilt_limit_max=0.9):
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

        self.sym_min_dist = create_pos('min_obs_distance')
        self.sym_max_dist = create_pos('max_obs_distance')
        #self.visualizer = visualizer if visualizer is not None else ROSBPBVisualizer('/debug_vis', 'map')

        self.gauss_position = point3(*[create_pos('gaussian_{}'.format(d)) for d in ['x','y','z']])
        self.sym_gaussian_x, self.sym_gaussian_y, self.sym_gaussian_z = self.gauss_position[:3]
        self.sym_obj_radius = create_pos('object_radius')
        self.sym_eigen_vecs = [[create_pos('eigen_{}_{}'.format(x, d)) for d in ['x','y','z']] for x in range(3)]

        self.eigen_vecs = [vector3(*syms) for syms in self.sym_eigen_vecs]

        # Looking along axes of uncertainty
        view_dir = x_of(camera.pose)

        self.sc_eigen_vecs = {'look along eigen {}'.format(x):
                            SC(norm(e)-norm(cross(view_dir, e)),
                               norm(e)-norm(cross(view_dir, e)),
                                norm(e),
                                GC(norm(cross(view_dir, e)))) for x, e in enumerate(self.eigen_vecs)}
        total_eigen_vec_value = sum([c.weight for c in self.sc_eigen_vecs.values()])

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
        self.sc_obs_distance = {'observation distance': SC(self.sym_min_dist - obj_dist,
                                                           self.sym_max_dist - obj_dist, 
                                                           5 + total_eigen_vec_value, 
                                                           obj_dist),
                                'look at': SC(1 - view_align, 
                                              1 - view_align, 
                                              1 + total_eigen_vec_value, 
                                              GC(view_align)),
                                'limit_tilt': SC(-sin(tilt_limit_max) - view_dir[2],
                                                 -sin(tilt_limit_min) - view_dir[2],
                                                 5 + total_eigen_vec_value,
                                                 view_dir[2])}
        # 'observation distance': SC(min_obs_fraction - projected_radius,
        #                            max_obs_fraction - projected_radius, 
        #                            1, GC(projected_radius))


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
        all_symbols        = robot_symbols.union(controlled_symbols)
        self.world         = self.km.get_active_geometry(all_symbols)
        hard_constraints   = self.km.get_constraints_by_symbols(all_symbols)
        self.state = {s: 0.0 for s in robot_symbols}

        # Generating full constraint set
        to_remove = set()
        controlled_values = {}
        for k, c in hard_constraints.items():
            if type(c.expr) is spw.Symbol and c.expr in controlled_symbols and str(c.expr) not in controlled_values:
                weight = 0.01
                controlled_values[str(c.expr)] = ControlledValue(c.lower, c.upper, c.expr, weight)
                to_remove.add(k)

        self.js_alias = {s: str(Path(erase_type(s))[-1]) for s in robot_symbols if s not in localization_vars}

        for s in controlled_symbols:
            if str(s) not in controlled_values:
                controlled_values[str(s)] = ControlledValue(-1e9, 1e9, s, 0.01)

        hard_constraints = {k: c for k, c in hard_constraints.items() if k not in to_remove}
        for cp in collision_link_paths:
            for x in range(8):
                hard_constraints['{} collision_avoidance {}'.format(cp, x)] = closest_distance_constraint(self.km.get_data(cp + ('pose',)), spw.eye(4), cp, Path('anon/{}'.format(x)))
                #hard_constraints['{} collision_avoidance {}'.format(cp, x)].lower += 0.1

        soft_constraints = self.sc_obs_distance.copy()
        #soft_constraints.update(self.sc_eigen_vecs)
        #soft_constraints.update(self.sc_obs_distance)
        #soft_constraints.update(self.sc_cone_constraints)
        self.collision_free_solver = TQPB(hard_constraints, soft_constraints, controlled_values)
        self.collision_solver      = GQPB(self.world, hard_constraints, soft_constraints, controlled_values, visualizer=visualizer)
        self.visualizer = visualizer

    def set_observation_distance(self, min_dist, max_dist):
        self.state[self.sym_min_dist] = min_dist
        self.state[self.sym_max_dist] = max_dist

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
        
        self.gc = gc
        #print('State after setting gc:\n  {}'.format('\n  '.join(['{}: {}'.format(k,v) for k, v in self.state.items()])))

    # Return sorted list of resolved camera 6d poses.
    def get_view_poses(self, num_iterations=100, int_factor=0.25, samples=20, debug_trajectory=None, equilibrium=0.05):
        if self.gc is None:
            raise Exception('Set a gaussion component before trying to solve for a view pose.')

        if self.sym_min_dist not in self.state or self.sym_max_dist not in self.state:
            raise Exception('Set minimum and maximum observation distance before computing view poses.')

        # Initialize samples
        # Create batch of integrators
        # Update batch in parallel
        # Rate results by their last errors
        # Add best to the explored view cones
        # Return sorted, rated results as 6D camera poses

        integrators = []
        angle_step  = ((2 * np.pi) / samples)
        offset      = np.random.random() * angle_step
        for x in range(samples):
            state  = self.state.copy()
            angle  = offset + angle_step * x
            ring_width = self.state[self.sym_max_dist] - self.state[self.sym_min_dist]
            radius = np.random.normal(1.5, 0.3) #self.state[self.sym_max_dist] - ring_width * 0.5, ring_width * 0.5)  # self.state[self.sym_max_dist] - ring_width * 0.5
            state[self.sym_loc_x] = state[self.sym_gaussian_x] + np.cos(angle) * radius
            state[self.sym_loc_y] = state[self.sym_gaussian_y] + np.sin(angle) * radius
            state[self.sym_loc_a] = angle - np.pi
            integrators.append(CommandIntegrator(self.collision_solver, start_state=state, equilibrium=equilibrium))
            integrators[-1].restart('Integrator {}'.format(x))

        for x, i in enumerate(integrators):
            self.collision_solver.reset_solver()
            print('Optimizing sample {} for gc {} of object {}'.format(x, self.gc.id, self.gc.obj_id))
            resets = 0
            while resets < MAX_RESETS:
                try:
                    i.run(int_factor, num_iterations)
                    break
                except QPSolverException as e:
                    if e.message == 'INIT_FAILED_HOTSTART':
                        if resets < MAX_RESETS:
                            print('Solver reported a failed hot start. Retrying - Attempt {}/{}'.format(resets + 1, MAX_RESETS))
                            resets += 1
                            self.collision_solver.reset_solver()
                        else:
                            raise Exception('Solver hot start failed too often.')

        results = []
        constraints = [c for c in self.sc_eigen_vecs.values() + self.sc_obs_distance.values()]
        for i in integrators:
            state = i.state
            js = {jn: state[s] for s, jn in self.js_alias.items() if s in state}
            nav_pose = [state[self.sym_loc_x], state[self.sym_loc_y], state[self.sym_loc_a]]

            error = 0.0
            for c in constraints:
                lower = c.lower if type(c.lower) not in symbolic_types else c.lower.subs(state)
                upper = c.upper if type(c.upper) not in symbolic_types else c.upper.subs(state)
                error += max(lower, 0) - min(upper, 0)
            
            results.append((error, self.camera.pose.subs(i.state), js, nav_pose, i.recorder.data))

        out = sorted(results)
        if debug_trajectory is not None:
            for _, _, _, _, t in out:
                debug_trajectory.append({Symbol(k): subsample_list(v, min(len(v), 3)) for k, v in t.items()})
        return [t[:4] for t in out]



