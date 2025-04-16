import numpy as np
import math
import random
import time # For potential timing/debugging
from collections import namedtuple

# Optional, but highly recommended for performance:
try:
    from scipy.spatial import KDTree
    USE_KDTREE = True
except ImportError:
    USE_KDTREE = False
    print("Warning: scipy.spatial.KDTree not found. Falling back to slower nearest neighbor search.")

# Local Imports (assuming behavior_learner is passed in)
# from your_module import ObjectBehaviorLearner # If needed for type hinting

# --- RRT* Configuration ---
DEFAULT_STEP_SIZE = 0.5       # How far to extend tree in one step (adjust based on map scale)
DEFAULT_MAX_ITERATIONS = 5000 # Max attempts to grow the tree
DEFAULT_GOAL_SAMPLE_RATE = 0.1 # Probability of sampling the goal point directly
DEFAULT_SEARCH_RADIUS = 1.5   # Radius for finding neighbors for rewiring (RRT*) - tune this!
ROBOT_RADIUS = 0.3            # Collision radius for the robot (adjust!)

# --- Helper Classes ---
class Node:
    """Represents a node in the RRT* tree."""
    def __init__(self, x, y, cost=0.0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost  # Cost from start node
        self.parent = parent # Reference to parent Node object

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Node({self.x:.2f}, {self.y:.2f}, cost={self.cost:.2f})"

# Define a simple obstacle structure (replace with occupancy grid later)
Obstacle = namedtuple("Obstacle", ["x", "y", "radius"])


class PathPlannerRRTStar:
    def __init__(self, behavior_learner, config_space_bounds, step_size=DEFAULT_STEP_SIZE, max_iter=DEFAULT_MAX_ITERATIONS, goal_sample_rate=DEFAULT_GOAL_SAMPLE_RATE, search_radius=DEFAULT_SEARCH_RADIUS, robot_radius=ROBOT_RADIUS):
        """
        Initializes the RRT* planner.

        Args:
            behavior_learner: Instance of ObjectBehaviorLearner (for graph access).
            config_space_bounds (tuple): (min_x, max_x, min_y, max_y) defining the planning area.
            step_size (float): Max distance to extend tree in one step.
            max_iter (int): Maximum number of sampling iterations.
            goal_sample_rate (float): Probability (0-1) of sampling the goal directly.
            search_radius (float): Radius for neighbor search during RRT* rewiring.
            robot_radius (float): Collision radius of the robot.
        """
        self.behavior_learner = behavior_learner
        self.min_x, self.max_x, self.min_y, self.max_y = config_space_bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.robot_radius = robot_radius

        self.static_obstacles = [] # List to store static obstacles (e.g., Obstacle namedtuples)
        self.start_node = None
        self.goal_node = None
        self.nodes = [] # List storing all Node objects in the tree

        # For KDTree optimization
        self.kdtree = None

        print(f"RRT* Planner Initialized. Using KDTree: {USE_KDTREE}")
        print(f"Bounds: X({self.min_x}, {self.max_x}), Y({self.min_y}, {self.max_y})")


    def update_static_obstacles(self, obstacles_list):
        """
        Updates the list of static obstacles.

        Args:
            obstacles_list (list): List of Obstacle objects or similar structure.
                                   For now, assuming list of Obstacle(x, y, radius).
        """
        # TODO: Adapt this if using an occupancy grid map from sensor data
        self.static_obstacles = obstacles_list
        print(f"Updated static obstacles: {len(self.static_obstacles)} obstacles loaded.")


    def _distance(self, node1, node2):
        """Calculates Euclidean distance between two nodes or points."""
        p1 = node1 if isinstance(node1, Node) else Node(node1[0], node1[1])
        p2 = node2 if isinstance(node2, Node) else Node(node2[0], node2[1])
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_random_point(self):
        """Samples a random point within the configuration space."""
        if random.random() < self.goal_sample_rate and self.goal_node:
            # Bias towards goal
            return (self.goal_node.x, self.goal_node.y)
        else:
            # Sample random point
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            return (x, y)

    def _get_nearest_node(self, point):
        """Finds the node in the tree nearest to the given point."""
        if not self.nodes:
            return None

        target_point = np.array(point)

        if USE_KDTREE and self.kdtree:
            # Use KDTree for fast lookup
            distances, indices = self.kdtree.query(target_point, k=1)
            # KDTree might return multiple indices if distances are equal, take the first
            nearest_idx = indices[0] if isinstance(indices, np.ndarray) else indices
            return self.nodes[nearest_idx]
        else:
            # Manual search (slower)
            min_dist = float('inf')
            nearest = None
            for node in self.nodes:
                dist = self._distance(node, point)
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            return nearest

    def _find_near_nodes(self, target_node):
        """Finds all nodes within search_radius of the target_node."""
        if not self.nodes:
            return []

        target_point = np.array([target_node.x, target_node.y])
        radius_sq = self.search_radius ** 2

        if USE_KDTREE and self.kdtree:
            # Use KDTree for faster radius search
            indices = self.kdtree.query_ball_point(target_point, r=self.search_radius)
            near_nodes = [self.nodes[i] for i in indices]
        else:
            # Manual search (slower)
            near_nodes = []
            for node in self.nodes:
                dist_sq = (node.x - target_node.x)**2 + (node.y - target_node.y)**2
                if dist_sq <= radius_sq:
                    near_nodes.append(node)
        return near_nodes


    def _steer(self, from_node, to_point):
        """Steers from from_node towards to_point by step_size."""
        dist = self._distance(from_node, to_point)
        if dist <= self.step_size:
            # Already close enough or exact point
            return Node(to_point[0], to_point[1]) # Return as a potential node
        else:
            # Move step_size towards the point
            theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
            new_x = from_node.x + self.step_size * math.cos(theta)
            new_y = from_node.y + self.step_size * math.sin(theta)
            return Node(new_x, new_y) # Return as a potential node


    def _is_collision(self, node1, node2, dynamic_obstacles_pred):
        """
        Checks if the path segment between node1 and node2 collides with obstacles.

        Args:
            node1 (Node): Start node of the segment.
            node2 (Node): End node of the segment.
            dynamic_obstacles_pred (dict): Predicted trajectories, e.g.,
                                           {obj_id: [(t0, x0, y0, r0), (t1, x1, y1, r1), ...]}
                                           where r is the obstacle radius.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        p1 = np.array([node1.x, node1.y])
        p2 = np.array([node2.x, node2.y])
        segment_vec = p2 - p1
        segment_len = np.linalg.norm(segment_vec)

        if segment_len < 1e-6: # Avoid division by zero for very short segments
            return False # No movement, no collision

        # --- Static Obstacle Check ---
        # TODO: Replace with Occupancy Grid check if using a map
        for obs in self.static_obstacles:
            obs_center = np.array([obs.x, obs.y])
            total_radius_sq = (self.robot_radius + obs.radius)**2

            # Check endpoint collision
            if np.sum((p2 - obs_center)**2) <= total_radius_sq:
                 return True # Endpoint collides

            # Check line segment collision (simplified point-segment distance check)
            # Project obstacle center onto the line defined by the segment
            line_dir = segment_vec / segment_len
            ap = obs_center - p1
            proj_len = np.dot(ap, line_dir)

            if 0 <= proj_len <= segment_len:
                # Projection falls within the segment
                closest_point_on_segment = p1 + proj_len * line_dir
                dist_sq = np.sum((obs_center - closest_point_on_segment)**2)
                if dist_sq <= total_radius_sq:
                    return True # Collision with segment
            else:
                # Closest point is one of the endpoints (already checked p2, check p1 implicitly by checking distance below)
                dist_sq = np.sum((p1 - obs_center)**2) if proj_len < 0 else np.sum((p2 - obs_center)**2)
                if dist_sq <= total_radius_sq:
                     # This check is somewhat redundant given endpoint checks but safe to keep
                     # Primarily handles case where projection is outside but endpoint is close
                     pass # Endpoint check will handle this

        # --- Dynamic Obstacle Check (Simplified) ---
        # TODO: Implement more robust dynamic collision checking
        # This simplified version checks predicted positions at discrete steps
        # A better approach uses continuous collision detection or time estimation
        num_steps = max(2, int(segment_len / (self.step_size * 0.5))) # Check multiple points along segment
        for i in range(num_steps + 1):
            t_interp = i / num_steps
            robot_pos = p1 + t_interp * segment_vec
            # Estimate time (very crude: assumes constant robot velocity)
            # estimated_time = segment_len * t_interp / average_robot_speed # Needs robot speed!
            # For now, let's just check against all *predicted* points as static obstacles
            # THIS IS A MAJOR SIMPLIFICATION - DOES NOT ACCOUNT FOR TIMING
            for obj_id, trajectory in dynamic_obstacles_pred.items():
                for (_time, obs_x, obs_y, obs_radius) in trajectory: # Assuming trajectory includes radius
                    obs_center = np.array([obs_x, obs_y])
                    total_radius_sq = (self.robot_radius + obs_radius)**2
                    dist_sq = np.sum((robot_pos - obs_center)**2)
                    if dist_sq <= total_radius_sq:
                        #print(f"Collision detected with predicted obstacle {obj_id} at robot pos {robot_pos}")
                        return True # Collision

        return False # No collision detected


    def _calculate_cost(self, from_node, to_node):
        """Calculates cost from the start node to to_node via from_node."""
        return from_node.cost + self._distance(from_node, to_node)

    def _reconstruct_path(self, goal_node_final):
        """Traces the path back from the goal to the start."""
        path = []
        current = goal_node_final
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1] # Return reversed path (start to goal)

    def predict_dynamic_obstacles(self, robot_pose, prediction_horizon_secs, time_step):
        """
        Predicts future positions of relevant dynamic objects using the behavior learner.
        Needs concrete implementation based on ObjectBehaviorLearner details.
        """
        # TODO: Implement actual prediction logic
        # 1. Identify relevant objects near potential path area (how to define this?)
        # 2. Get sufficient state history for these objects.
        # 3. Loop time steps up to prediction_horizon_secs:
        #    a. Prepare input tensor for behavior_learner.predict()
        #    b. Get predicted states.
        #    c. Store predicted (t, x, y, radius) - Need object radius info!
        # 4. Return dictionary: {obj_id: [(t0, x0, y0, r0), (t1, x1, y1, r1), ...]}
        print("Placeholder: Predicting dynamic obstacles...")
        # Example structure, assuming radius 0.2 for all dynamic objects
        # dynamic_predictions = {
        #    'obj_1': [(0.5*i, 1.0 + 0.5*i, 2.0, 0.2) for i in range(int(prediction_horizon_secs/time_step)+1)],
        #    'obj_2': [(0.5*i, 5.0 - 0.3*i, 3.5 + 0.1*i, 0.2) for i in range(int(prediction_horizon_secs/time_step)+1)]
        # }
        return {} # Return empty for now

    def get_goal_position(self, target, current_graph):
        """Resolves target (ID or coords) to coordinates."""
        if isinstance(target, (list, tuple, np.ndarray)) and len(target) >= 2:
            return tuple(target[:2])
        elif isinstance(target, str):
             if current_graph and target in current_graph.nodes:
                  node_data = current_graph.nodes[target]
                  state = node_data.get('state')
                  if state and len(state) >= 2:
                       print(f"Target '{target}' resolved to graph coords: {tuple(state[:2])}")
                       return tuple(state[:2])
             # Optional: Check predefined named locations?
             # elif target == "kitchen_area": return (10.0, 5.0) # Example
        print(f"Warning: Could not resolve target '{target}' to coordinates.")
        return None

    def plan_path(self, start_pose, goal_target):
        """
        Plans a path using RRT* algorithm.

        Args:
            start_pose (tuple): (x, y) or (x, y, theta) of the robot.
            goal_target (str or tuple): Object ID, named location, or (x, y) coordinates.

        Returns:
            list or None: A list of (x, y) waypoints from start to goal, or None if no path found.
        """
        start_time = time.time()

        # --- Initialization ---
        self.start_node = Node(start_pose[0], start_pose[1], cost=0.0, parent=None)
        goal_pos = self.get_goal_position(goal_target, self.behavior_learner.object_graph)
        if goal_pos is None:
            print("RRT* Error: Goal position could not be determined.")
            return None
        self.goal_node = Node(goal_pos[0], goal_pos[1]) # Goal position node (not yet in tree)

        self.nodes = [self.start_node]
        if USE_KDTREE:
            self.kdtree = KDTree([[self.start_node.x, self.start_node.y]])

        best_goal_node = None # Keep track of the node closest to goal with lowest cost

        # --- Predict Dynamic Obstacles ---
        # TODO: Define horizon and time step appropriately
        dynamic_preds = self.predict_dynamic_obstacles(start_pose, 5.0, 0.5)

        # --- RRT* Main Loop ---
        for i in range(self.max_iter):
            rnd_point = self._get_random_point()
            nearest_node = self._get_nearest_node(rnd_point)
            if nearest_node is None: continue # Should not happen after adding start node

            new_potential_node = self._steer(nearest_node, rnd_point)

            # Check collision for the edge from nearest_node to new_potential_node
            if not self._is_collision(nearest_node, new_potential_node, dynamic_preds):
                # Find neighbors and optimal parent (ChooseParent in RRT* literature)
                near_nodes = self._find_near_nodes(new_potential_node)
                min_cost_node = nearest_node # Start with nearest as potential parent
                min_new_cost = self._calculate_cost(nearest_node, new_potential_node)

                for near_node in near_nodes:
                    cost_via_near = self._calculate_cost(near_node, new_potential_node)
                    if cost_via_near < min_new_cost and not self._is_collision(near_node, new_potential_node, dynamic_preds):
                        min_cost_node = near_node
                        min_new_cost = cost_via_near

                # Add the new node to the tree
                new_node = Node(new_potential_node.x, new_potential_node.y, cost=min_new_cost, parent=min_cost_node)
                self.nodes.append(new_node)
                if USE_KDTREE:
                    # Rebuild KDTree or use incremental update if library supports it
                    self.kdtree = KDTree([[n.x, n.y] for n in self.nodes]) # Simple rebuild

                # Rewire neighbors (Rewire in RRT* literature)
                for near_node in near_nodes:
                    if near_node == min_cost_node: continue # Skip the parent

                    cost_via_new = self._calculate_cost(new_node, near_node)
                    if cost_via_new < near_node.cost and not self._is_collision(new_node, near_node, dynamic_preds):
                        near_node.parent = new_node
                        near_node.cost = cost_via_new
                        # Note: Need to propagate cost updates if implementing complex cost functions

                # Check if goal reached
                dist_to_goal = self._distance(new_node, self.goal_node)
                if dist_to_goal <= self.step_size:
                    # Potentially connect directly to goal
                    if not self._is_collision(new_node, self.goal_node, dynamic_preds):
                        final_cost = self._calculate_cost(new_node, self.goal_node)
                        # If this is the first goal connection or cheaper than previous best
                        if best_goal_node is None or final_cost < best_goal_node.cost:
                             # Create a final node representing the goal connected to the tree
                             goal_connected_node = Node(self.goal_node.x, self.goal_node.y, cost=final_cost, parent=new_node)
                             best_goal_node = goal_connected_node
                             # Optional: Early exit if a good enough path is found
                             # if final_cost < some_threshold: break

            if i % 500 == 0:
                 print(f"RRT* Iteration: {i}/{self.max_iter}, Tree Size: {len(self.nodes)}")


        # --- Path Reconstruction ---
        elapsed_time = time.time() - start_time
        print(f"RRT* Planning finished in {elapsed_time:.2f} seconds.")

        if best_goal_node:
            print(f"Path found to goal with cost {best_goal_node.cost:.2f}.")
            path = self._reconstruct_path(best_goal_node)
            return path
        else:
            # If direct connection failed, find node closest to goal
            print("Direct path to goal not found or collision detected. Finding closest node.")
            closest_node_to_goal = self._get_nearest_node((self.goal_node.x, self.goal_node.y))
            if closest_node_to_goal:
                 path = self._reconstruct_path(closest_node_to_goal)
                 print(f"Returning path to closest node ({len(path)} waypoints).")
                 return path
            else:
                 print("RRT* Error: No path found, tree is empty (should not happen).")
                 return None

# --- Example Usage (within RobotNavigator or robot_loop) ---
# Assume 'behavior_learner' instance exists
# Assume 'config_bounds' = (0, 20, 0, 15) # Example map boundaries

# planner = PathPlannerRRTStar(behavior_learner, config_bounds)

# Define some simple static obstacles for testing
# obstacles = [
#     Obstacle(x=5, y=5, radius=1.0),
#     Obstacle(x=10, y=8, radius=1.5),
#     Obstacle(x=15, y=10, radius=1.0)
# ]
# planner.update_static_obstacles(obstacles)

# In RobotNavigator.map_actions_to_robot_commands for action 'go_to':
# current_pose = get_robot_state() # e.g., (1.0, 1.0, 0.0)
# target = instruction.get("target") # e.g., "obj_2" or (18.0, 12.0)
# waypoints = planner.plan_path(current_pose, target)
# if waypoints:
#     # Add move_to commands using the waypoints
#     for wp in waypoints:
#          robot_commands.append(("move_to", wp))
# else:
#     print(f"Path planning failed for target {target}")
