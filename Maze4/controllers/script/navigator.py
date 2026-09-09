import math
import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt, label

# This file is the robot brain for Maze 4.
# Mission states
STATE_EXPLORE   = 0   # Explore and look for pillars.
STATE_GO_BLUE   = 1   # Go to the blue pillar.
STATE_GO_YELLOW = 2   # After blue, go to yellow.
STATE_FINISHED  = 3   # Mission is finished.

class AutonomousNavigator:
    """
    Autonomous exploration brain for a differential-drive robot.
    Operates using Cost-Aware Dijkstra, Frontier Heuristics, and Blended Pure Pursuit.
    """

    def __init__(self, motors, wheel_track, map_size, resolution):
        # Save motor objects.
        # Hardware
        self.fl, self.fr, self.rl, self.rr = motors
        self.WHEEL_TRACK  = 0.2794
        self.WHEEL_RADIUS = 0.0436

        # Map values
        self.MAP_SIZE   = map_size
        self.RESOLUTION = resolution

        # Tuning values
        # This padding is small because the maze paths are narrow.
        self.SAFETY_PADDING_M = -0.070

        physical_radius_m = self.WHEEL_TRACK / 2.0
        self.ROBOT_RADIUS_CELLS = int(math.ceil(
            (physical_radius_m + self.SAFETY_PADDING_M) / self.RESOLUTION
        ))
        print(f"[Navigator] ROBOT_RADIUS_CELLS = {self.ROBOT_RADIUS_CELLS} "
              f"({self.ROBOT_RADIUS_CELLS * self.RESOLUTION * 100:.1f} cm half-width inflation)")

        self.LOOKAHEAD_DISTANCE  = 0.15
        self.ARRIVAL_THRESHOLD   = 0.15
        # Pillars use their own distance limits.
        self.PILLAR_ARRIVAL_THRESHOLD = 0.40
        self.PILLAR_MISSION_HANDOFF_THRESHOLD = 0.60
        self.BLUE_PILLAR_REACHED_THRESHOLD = 0.45
        self.YELLOW_PILLAR_REACHED_THRESHOLD = 0.45
        self.PILLAR_SURFACE_IGNORE_M = 0.14
        self.PILLAR_LINE_START_IGNORE_M = 0.06
        self.PILLAR_APPROACH_RADIUS = 0.8

        self.WALL_PENALTY_WEIGHT = 15.0

        # Frontier score values.
        self.W_SIZE = 0.1
        self.W_DIST = 3.0
        self.BLEND_THRESHOLD = 1.2

        # Speed limits
        self.MAX_LINEAR_SPEED  = 0.15
        self.MAX_ANGULAR_SPEED = 1.2
        self.MAX_MOTOR_RADS    = 15.0
        self.K_P_OMEGA         = 1.8

        # Robot safety box size
        self.ENVELOPE_FRONT  =  0.175
        self.ENVELOPE_REAR   = -0.175
        self.ENVELOPE_SIDE   =  0.1397

        # Robot state
        # current_path is the path and path_index is where we are on it.
        self.current_path            = []
        self.path_index              = 0
        self.target_frontier_world   = None
        self.last_plan_time          = 0.0
        self.blacklisted_frontiers   = set()
        self.FRONTIER_BLACKLIST_RADIUS_M = 0.14
        self.FRONTIER_BLACKLIST_RADIUS_CELLS = max(1, int(math.ceil(
            self.FRONTIER_BLACKLIST_RADIUS_M / self.RESOLUTION
        )))

        self.consecutive_stuck_ticks = 0
        # Recovery mode is used when the robot gets stuck.
        self.RECOVERY_MODE           = False
        self.recovery_start_time     = 0.0
        self.recovery_backoff_duration = 0.8
        self.recovery_turn_duration = 1.0

        self.mission_state      = STATE_EXPLORE
        self.mission_target_pos = None
        # This says why the current path was made.
        self.mission_path_intent = None
        self._blue_departure_path = []

        self.last_inflated = None

        self.last_stuck_time  = 0.0
        self.last_stuck_xy    = (0.0, 0.0)

        self.local_frontier_picks = 0
        self.visited_frontier_count = 0

        self.permanently_unreachable: set = set()
        # Failed frontiers get a short cooldown.
        self._relax_fail_timestamps: dict = {}
        self.RELAX_FAIL_COOLDOWN_S = 4.0

        self.YELLOW_SCAN_TRIGGER   = 5
        # If yellow is missing, the robot can spin to scan for it.
        self.YELLOW_SCAN_DURATION  = 6.5
        self.yellow_scan_active    = False
        self.yellow_scan_start     = 0.0
        self.yellow_scan_bl_count  = 0

    def world_to_grid(self, wx, wy):
        # Convert world metres to grid cells.
        gx = int(wx / self.RESOLUTION) + self.MAP_SIZE // 2
        gy = int(wy / self.RESOLUTION) + self.MAP_SIZE // 2
        return gx, gy

    def grid_to_world(self, gx, gy):
        # Convert grid cells back to world metres.
        wx = (gx - self.MAP_SIZE // 2) * self.RESOLUTION
        wy = (gy - self.MAP_SIZE // 2) * self.RESOLUTION
        return wx, wy

    @staticmethod
    def normalize_angle(angle):
        # Keep angles between -pi and pi.
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def get_costmap(self, grid, green_floor=None, extra_blocked=None):
        # Build the cost map used for path planning.
        walls = grid > 0.65
        if extra_blocked is not None:
            walls = walls | extra_blocked
        dist_field = distance_transform_edt(~walls)
        inflated_bool = dist_field <= self.ROBOT_RADIUS_CELLS

        # Green floor is blocked and nearby cells get extra cost.
        GREEN_SOFT_MARGIN_CELLS = 4
        green_dist = None
        if green_floor is not None:
            inflated_bool = inflated_bool | green_floor
            green_dist = distance_transform_edt(~green_floor)

        self.last_inflated = inflated_bool
        cost_map = np.full_like(grid, 50.0, dtype=np.float32)
        cost_map[inflated_bool] = np.inf

        safe_free_mask = (~inflated_bool) & (grid < 0.0)
        clearance = dist_field[safe_free_mask] - self.ROBOT_RADIUS_CELLS
        clearance = np.maximum(clearance, 0.01)

        cost_map[safe_free_mask] = 1.0 + (self.WALL_PENALTY_WEIGHT / (clearance ** 2))

        if green_dist is not None:
            green_penalty_mask = (green_dist <= GREEN_SOFT_MARGIN_CELLS) & safe_free_mask
            safe_green_dist = np.maximum(green_dist[green_penalty_mask], 0.1)
            cost_map[green_penalty_mask] += 200.0 / safe_green_dist

        return inflated_bool, cost_map


    # Reference for costmap inflation code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
    def _path_endpoint_near_unexplored_frontier(self, grid, path, radius_m=0.25):
        # Check if the path ends near unexplored space.
        if not path:
            return False

        col, row = self.world_to_grid(path[-1][0], path[-1][1])
        radius_cells = int(math.ceil(radius_m / self.RESOLUTION))

        row_min = max(0, row - radius_cells)
        row_max = min(self.MAP_SIZE, row + radius_cells + 1)
        col_min = max(0, col - radius_cells)
        col_max = min(self.MAP_SIZE, col + radius_cells + 1)

        free_cells = grid < 0.0
        unknown_cells = (grid > 0.4) & (grid < 0.6)

        frontier_mask = np.zeros_like(grid, dtype=bool)
        # Frontier means free space next to unknown space.
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            frontier_mask |= free_cells & np.roll(unknown_cells, shift=(dy, dx), axis=(0, 1))

        return bool(np.any(frontier_mask[row_min:row_max, col_min:col_max]))

    def _direct_pillar_path(self, start_w, pillar_w, grid, cost_map,
                            allow_frontier_endpoint=False):
        # Try to make a path to a pillar.
        path = self.plan_path_to_pillar(start_w, pillar_w, cost_map)
        if path and not allow_frontier_endpoint:
            if self._path_endpoint_near_unexplored_frontier(grid, path):
                msg_key = (round(pillar_w[0], 2), round(pillar_w[1], 2))
                if getattr(self, '_last_deferred_direct_pillar_key', None) != msg_key:
                    self._last_deferred_direct_pillar_key = msg_key
                    print("[Navigator] MISSION: Direct pillar approach borders unexplored frontier — exploring area first.")
                return []
        return path

    def _add_blacklisted_frontier(self, col, row):
        # Remember a frontier we do not want to use again.
        self.blacklisted_frontiers.add((int(col), int(row)))

    def _frontier_is_blacklisted(self, col, row):
        # Nearby blacklisted cells count as the same frontier.
        if not self.blacklisted_frontiers:
            return False
        radius_sq = self.FRONTIER_BLACKLIST_RADIUS_CELLS ** 2
        for b_col, b_row in self.blacklisted_frontiers:
            if (col - b_col) ** 2 + (row - b_row) ** 2 <= radius_sq:
                return True
        return False

    def _frontier_region_is_blacklisted(self, xs, ys):
        # Check if a whole frontier region is blacklisted.
        if not self.blacklisted_frontiers or xs.size == 0:
            return False
        radius_sq = self.FRONTIER_BLACKLIST_RADIUS_CELLS ** 2
        for b_col, b_row in self.blacklisted_frontiers:
            if np.any((xs - b_col) ** 2 + (ys - b_row) ** 2 <= radius_sq):
                return True
        return False

    def _blacklist_current_frontier(self, label=""):
        # Blacklist the frontier after we reach it.
        if self.target_frontier_world is None:
            return
        col, row = self.world_to_grid(
            self.target_frontier_world[0],
            self.target_frontier_world[1]
        )
        self._add_blacklisted_frontier(col, row)
        self.visited_frontier_count += 1
        self.local_frontier_picks   = 0
        tag = f" ({label})" if label else ""
        print(f"[Navigator] Frontier visited & blacklisted{tag}. "
              f"Total visited: {self.visited_frontier_count}, "
              f"blacklisted: {len(self.blacklisted_frontiers)}")
        self.target_frontier_world = None

    def _mark_permanently_unreachable(self, col, row, current_time):
        # Mark a frontier as not reachable.
        self.permanently_unreachable.add((int(col), int(row)))
        self._relax_fail_timestamps[(int(col), int(row))] = current_time

    def _is_relax_cooldown_active(self, col, row):
        # Check if this frontier is still cooling down.
        ts = self._relax_fail_timestamps.get((int(col), int(row)))
        if ts is None:
            return False
        return True

    def _purge_relax_cooldowns(self, current_time):
        # Remove old cooldowns.
        expired = [
            k for k, ts in self._relax_fail_timestamps.items()
            if current_time - ts > self.RELAX_FAIL_COOLDOWN_S
        ]
        for k in expired:
            del self._relax_fail_timestamps[k]

    def _frontier_region_is_permanently_unreachable(self, xs, ys):
        # Skip regions that are marked unreachable.
        if not self.permanently_unreachable or xs.size == 0:
            return False
        radius_sq = self.FRONTIER_BLACKLIST_RADIUS_CELLS ** 2
        for p_col, p_row in self.permanently_unreachable:
            if np.any((xs - p_col) ** 2 + (ys - p_row) ** 2 <= radius_sq):
                return True
        return False

    def find_best_frontier(self, grid, inflated_map, rx, ry, force_global=False, ignore_blacklist=False):
        # Find the best frontier to explore next.
        free_cells    = grid < 0.0
        unknown_cells = (grid > 0.4) & (grid < 0.6)

        frontier_mask = np.zeros_like(grid, dtype=bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            frontier_mask |= free_cells & np.roll(unknown_cells, shift=(dy, dx), axis=(0, 1))

        labeled_array, num_features = label(frontier_mask)
        if num_features == 0:
            return None

        robot_col, robot_row = self.world_to_grid(rx, ry)

        # If we are stuck locally, prefer farther frontiers.
        GLOBAL_ESCAPE_RADIUS_M  = max(0.8, self.ARRIVAL_THRESHOLD * 4.0)
        GLOBAL_LOCAL_PENALTY    = 1e5
        W_SIZE_GLOBAL           = 1.0

        best_frontier = None
        best_utility  = -np.inf

        for region_id in range(1, num_features + 1):
            region_coords = np.where(labeled_array == region_id)
            ys, xs = region_coords[0], region_coords[1]
            size = len(ys)

            if not ignore_blacklist and self._frontier_region_is_blacklisted(xs, ys):
                continue
            if self._frontier_region_is_permanently_unreachable(xs, ys):
                continue

            cy = float(np.mean(ys))
            cx = float(np.mean(xs))

            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            order = np.argsort(d2)

            # Pick a safe cell near the middle of the frontier.
            rep_col, rep_row = None, None
            for idx in order:
                cc, rr = int(xs[idx]), int(ys[idx])
                if inflated_map[rr, cc]:
                    continue
                if self._is_relax_cooldown_active(cc, rr):
                    continue
                if not ignore_blacklist and self._frontier_is_blacklisted(cc, rr):
                    continue
                rep_col, rep_row = cc, rr
                break

            if rep_col is None:
                continue

            dist_meters = math.hypot(
                (rep_col - robot_col) * self.RESOLUTION,
                (rep_row - robot_row) * self.RESOLUTION
            )

            if dist_meters < (self.ARRIVAL_THRESHOLD + 0.2):
                continue

            if force_global:
                local_penalty = (GLOBAL_LOCAL_PENALTY if dist_meters < GLOBAL_ESCAPE_RADIUS_M else 0.0)
                utility = (W_SIZE_GLOBAL * size) - local_penalty
            else:
                utility = (self.W_SIZE * size) - (self.W_DIST * dist_meters)

            if utility > best_utility:
                best_utility  = utility
                best_frontier = self.grid_to_world(rep_col, rep_row)

        if (best_frontier is None and len(self.blacklisted_frontiers) > 0 and not ignore_blacklist):
            print("[Navigator] All frontiers are already visited/unreachable.")

        return best_frontier

    def find_frontiers_toward(self, grid, inflated_map, target_wx, target_wy, ignore_blacklist=False):
        # Find frontiers that move us toward a target.
        free_cells    = grid < 0.0
        unknown_cells = (grid > 0.4) & (grid < 0.6)

        frontier_mask = np.zeros_like(grid, dtype=bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            frontier_mask |= free_cells & np.roll(unknown_cells, shift=(dy, dx), axis=(0, 1))

        labeled_array, num_features = label(frontier_mask)
        if num_features == 0:
            return []

        target_col, target_row = self.world_to_grid(target_wx, target_wy)
        candidates = []

        for region_id in range(1, num_features + 1):
            region_coords = np.where(labeled_array == region_id)
            ys, xs = region_coords[0], region_coords[1]
            if xs.size == 0:
                continue

            cy = float(np.mean(ys))
            cx = float(np.mean(xs))

            if not ignore_blacklist and self._frontier_region_is_blacklisted(xs, ys):
                continue

            if self._frontier_region_is_permanently_unreachable(xs, ys):
                continue

            rep_col, rep_row = None, None
            d2_to_centroid = (xs - cx) ** 2 + (ys - cy) ** 2
            order = np.argsort(d2_to_centroid)
            for idx in order:
                rr, cc = int(ys[idx]), int(xs[idx])
                if inflated_map[rr, cc]:
                    continue
                if not ignore_blacklist and self._frontier_is_blacklisted(cc, rr):
                    continue
                if self._is_relax_cooldown_active(cc, rr):
                    continue
                rep_col, rep_row = cc, rr
                break
            if rep_col is None:
                continue

            d = math.hypot(
                (rep_col - target_col) * self.RESOLUTION,
                (rep_row - target_row) * self.RESOLUTION
            )
            candidates.append((d, self.grid_to_world(rep_col, rep_row)))

        candidates.sort(key=lambda x: x[0])
        return [pt for _, pt in candidates]

    def find_frontier_toward(self, grid, inflated_map, target_wx, target_wy, ignore_blacklist=False):
        # Return only the closest target-facing frontier.
        lst = self.find_frontiers_toward(
            grid, inflated_map, target_wx, target_wy,
            ignore_blacklist=ignore_blacklist
        )
        return lst[0] if lst else None



    # Reference for frontier selection code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
    def plan_path_dijkstra(self, start_w, goal_w, cost_map):
        # Dijkstra path planner using the cost map.
        start_col, start_row = self.world_to_grid(start_w[0], start_w[1])
        goal_col,  goal_row  = self.world_to_grid(goal_w[0],  goal_w[1])

        local_cost = cost_map.copy()

        # Loosen the start and goal area so inflation does not trap us.
        R = self.ROBOT_RADIUS_CELLS + 1
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                row, col = start_row + dy, start_col + dx
                if 0 <= col < self.MAP_SIZE and 0 <= row < self.MAP_SIZE:
                    if local_cost[row, col] == np.inf:
                        local_cost[row, col] = 1.0

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                row, col = goal_row + dy, goal_col + dx
                if 0 <= col < self.MAP_SIZE and 0 <= row < self.MAP_SIZE:
                    if local_cost[row, col] == np.inf:
                        local_cost[row, col] = 1.0

        if (not (0 <= goal_col < self.MAP_SIZE and 0 <= goal_row < self.MAP_SIZE)
                or local_cost[goal_row, goal_col] == np.inf):
            return []

        queue = []
        heapq.heappush(queue, (0.0, start_col, start_row))

        # This stores where each cell came from for rebuilding the path.
        came_from   = {(start_col, start_row): None}
        cost_so_far = {(start_col, start_row): 0.0}

        max_iter = self.MAP_SIZE * self.MAP_SIZE

        for _ in range(max_iter):
            if not queue:
                break

            current_cost, col, row = heapq.heappop(queue)

            if col == goal_col and row == goal_row:
                path = []
                curr = (col, row)
                while curr is not None:
                    path.append(self.grid_to_world(curr[0], curr[1]))
                    curr = came_from[curr]
                return path[::-1]

            for dc, dr, step_dist in (
                (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
                (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414), ( 1,  1, 1.414)
            ):
                nc, nr = col + dc, row + dr
                if 0 <= nc < self.MAP_SIZE and 0 <= nr < self.MAP_SIZE:
                    edge_cost = local_cost[nr, nc]
                    if edge_cost == np.inf:
                        continue

                    new_cost = cost_so_far[(col, row)] + (edge_cost * step_dist)

                    if (nc, nr) not in cost_so_far or new_cost < cost_so_far[(nc, nr)]:
                        cost_so_far[(nc, nr)] = new_cost
                        heapq.heappush(queue, (new_cost, nc, nr))
                        came_from[(nc, nr)] = (col, row)

        return []


    # Reference for Dijkstra path planning code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
    def check_hardware_safety_envelope(self, lidar_ranges, lidar_fov):
        # Check if something is inside the robot safety box.
        num_beams = len(lidar_ranges)
        if num_beams == 0:
            return False

        for i, r in enumerate(lidar_ranges):
            if not math.isfinite(r) or r <= 0.05 or r > 3.5:
                continue

            beam_angle = (lidar_fov / 2.0) - (i / num_beams) * lidar_fov
            local_x = r * math.cos(beam_angle)
            local_y = r * math.sin(beam_angle)

            # Convert lidar point to robot x/y and check the box.
            if (self.ENVELOPE_REAR <= local_x <= self.ENVELOPE_FRONT and
                    -self.ENVELOPE_SIDE <= local_y <= self.ENVELOPE_SIDE):
                if math.hypot(local_x, local_y) > 0.21:
                    continue
                return True
        return False

    def _direct_drive_to(self, rx, ry, rtheta, target_wx, target_wy):
        # Drive directly to a nearby target.
        dist = math.hypot(target_wx - rx, target_wy - ry)
        if dist < self.ARRIVAL_THRESHOLD:
            self.stop_robot()
            return True

        dist_bucket = int(dist * 10)
        if dist_bucket != getattr(self, '_last_direct_dist_bucket', -1):
            self._last_direct_dist_bucket = dist_bucket
            print(f"[Navigator] MISSION: Direct drive to target (dist={dist:.1f}m).")

        alpha = self.normalize_angle(
            math.atan2(target_wy - ry, target_wx - rx) - rtheta
        )

        # Slow down if the target is not in front.
        v_target     = self.MAX_LINEAR_SPEED * max(0.0, math.cos(alpha))
        omega_target = max(-self.MAX_ANGULAR_SPEED,
                           min(self.MAX_ANGULAR_SPEED, self.K_P_OMEGA * alpha))
        if abs(alpha) > 1.0:
            v_target = 0.0

        half_track  = self.WHEEL_TRACK / 2.0
        v_left      = v_target - (omega_target * half_track)
        v_right     = v_target + (omega_target * half_track)
        rads_left   = max(-self.MAX_MOTOR_RADS,
                          min(self.MAX_MOTOR_RADS, v_left  / self.WHEEL_RADIUS))
        rads_right  = max(-self.MAX_MOTOR_RADS,
                          min(self.MAX_MOTOR_RADS, v_right / self.WHEEL_RADIUS))

        self.fl.setVelocity(rads_left);  self.rl.setVelocity(rads_left)
        self.fr.setVelocity(rads_right); self.rr.setVelocity(rads_right)
        return False

    def _straight_line_is_clear(self, start_w, target_w, cost_map, goal_ignore_m=None, start_ignore_m=None):
        # Check if a straight line to the target is clear.
        start_col, start_row = self.world_to_grid(start_w[0], start_w[1])
        goal_col, goal_row = self.world_to_grid(target_w[0], target_w[1])
        hard_blocked = getattr(self, '_hard_blocked_mask', None)

        dx = abs(goal_col - start_col)
        dy = abs(goal_row - start_row)
        sx = 1 if start_col < goal_col else -1
        sy = 1 if start_row < goal_row else -1
        err = dx - dy

        goal_ignore_m = self.PILLAR_ARRIVAL_THRESHOLD if goal_ignore_m is None else goal_ignore_m
        start_ignore_m = (self.ROBOT_RADIUS_CELLS + 1) * self.RESOLUTION if start_ignore_m is None else start_ignore_m

        col, row = start_col, start_row
        while True:
            if not (0 <= col < self.MAP_SIZE and 0 <= row < self.MAP_SIZE):
                return False

            wx, wy = self.grid_to_world(col, row)
            dist_from_start = math.hypot(wx - start_w[0], wy - start_w[1])
            dist_to_goal = math.hypot(wx - target_w[0], wy - target_w[1])

            if (hard_blocked is not None
                    and dist_from_start > start_ignore_m
                    and hard_blocked[row, col]):
                # Camera obstacles also block direct movement.
                return False

            if dist_from_start > start_ignore_m and dist_to_goal > goal_ignore_m:
                if cost_map[row, col] == np.inf:
                    return False

            if col == goal_col and row == goal_row:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                col += sx
            if e2 < dx:
                err += dx
                row += sy

        return True

    def _pillar_arrival_is_clear(self, start_w, pillar_w, grid, cost_map):
        # Check if the pillar approach is clear.
        return self._straight_line_is_clear(
            start_w, pillar_w, cost_map,
            goal_ignore_m=self.PILLAR_SURFACE_IGNORE_M,
            start_ignore_m=self.PILLAR_LINE_START_IGNORE_M
        )

    def _try_direct_or_replan_to_pillar(self, rx, ry, rtheta, target_w, cost_map, color_label):
        # Try direct driving first, then replan if blocked.
        if self._straight_line_is_clear(
                (rx, ry), target_w, cost_map,
                goal_ignore_m=self.PILLAR_SURFACE_IGNORE_M,
                start_ignore_m=self.PILLAR_LINE_START_IGNORE_M):
            return self._direct_drive_to(rx, ry, rtheta, target_w[0], target_w[1])

        replanned = self.plan_path_to_pillar((rx, ry), target_w, cost_map)
        if replanned:
            print(f"[Navigator] GO_{color_label}: Direct approach blocked — "
                  "routing around obstacle.")
            self.current_path = replanned
            self.path_index = 0
            self.mission_path_intent = f"direct_{color_label.lower()}"
            return False

        print(f"[Navigator] GO_{color_label}: Direct approach blocked by obstacle — stopping.")
        self.stop_robot()
        return False

    def _pillar_approach_goal_cells(self, world_pos, cost_map, max_radius_m=None):
        # Make possible goal cells around a pillar.
        goal_col, goal_row = self.world_to_grid(world_pos[0], world_pos[1])
        max_radius_m = self.PILLAR_APPROACH_RADIUS if max_radius_m is None else max_radius_m
        max_radius_cells = int(math.ceil(max_radius_m / self.RESOLUTION))

        candidates = []
        row_min = max(0, goal_row - max_radius_cells)
        row_max = min(self.MAP_SIZE - 1, goal_row + max_radius_cells)
        col_min = max(0, goal_col - max_radius_cells)
        col_max = min(self.MAP_SIZE - 1, goal_col + max_radius_cells)

        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                if cost_map[row, col] == np.inf:
                    continue

                dist_cells = math.hypot(col - goal_col, row - goal_row)
                if dist_cells <= 0.0:
                    continue

                dist_m = dist_cells * self.RESOLUTION
                if dist_m <= max_radius_m:
                    candidates.append((dist_m, col, row))

        # Try closest safe cells first.
        candidates.sort(key=lambda item: item[0])
        return candidates

    def plan_path_to_any_goal_cell(self, start_w, goal_cells, cost_map):
        # Plan to any safe goal cell from a list.
        if not goal_cells:
            return []

        start_col, start_row = self.world_to_grid(start_w[0], start_w[1])
        local_cost = cost_map.copy()

        # Loosen the robot start area.
        R = self.ROBOT_RADIUS_CELLS + 1
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                row, col = start_row + dy, start_col + dx
                if 0 <= col < self.MAP_SIZE and 0 <= row < self.MAP_SIZE:
                    if local_cost[row, col] == np.inf:
                        local_cost[row, col] = 1.0

        valid_goals = {
            (col, row)
            for col, row in goal_cells
            if (0 <= col < self.MAP_SIZE and 0 <= row < self.MAP_SIZE
                    and local_cost[row, col] != np.inf)
        }
        if not valid_goals:
            return []

        queue = []
        heapq.heappush(queue, (0.0, start_col, start_row))

        # Stop when we reach any valid goal.
        came_from   = {(start_col, start_row): None}
        cost_so_far = {(start_col, start_row): 0.0}
        max_iter    = self.MAP_SIZE * self.MAP_SIZE

        for _ in range(max_iter):
            if not queue:
                break

            current_cost, col, row = heapq.heappop(queue)
            if current_cost > cost_so_far.get((col, row), np.inf):
                continue

            if (col, row) in valid_goals:
                path = []
                curr = (col, row)
                while curr is not None:
                    path.append(self.grid_to_world(curr[0], curr[1]))
                    curr = came_from[curr]
                return path[::-1]

            for dc, dr, step_dist in (
                (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
                (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414), ( 1,  1, 1.414)
            ):
                nc, nr = col + dc, row + dr
                if 0 <= nc < self.MAP_SIZE and 0 <= nr < self.MAP_SIZE:
                    edge_cost = local_cost[nr, nc]
                    if edge_cost == np.inf:
                        continue

                    new_cost = cost_so_far[(col, row)] + (edge_cost * step_dist)
                    if (nc, nr) not in cost_so_far or new_cost < cost_so_far[(nc, nr)]:
                        cost_so_far[(nc, nr)] = new_cost
                        heapq.heappush(queue, (new_cost, nc, nr))
                        came_from[(nc, nr)] = (col, row)

        return []

    def plan_path_to_pillar(self, start_w, pillar_w, cost_map):
        # Do not ignore real obstacles near the pillar.
        candidates = self._pillar_approach_goal_cells(pillar_w, cost_map)
        if not candidates:
            return []

        preferred_cells = [
            (col, row)
            for dist_m, col, row in candidates
            if dist_m <= self.PILLAR_ARRIVAL_THRESHOLD
        ]
        path = self.plan_path_to_any_goal_cell(start_w, preferred_cells, cost_map)
        if path:
            return path

        # If close cells fail, try a wider approach.
        fallback_cells = [(col, row) for _, col, row in candidates]
        return self.plan_path_to_any_goal_cell(start_w, fallback_cells, cost_map)

    def find_nearest_navigable_goal(self, world_pos, cost_map):
        # Find the nearest safe goal near a point.
        candidates = self._pillar_approach_goal_cells(world_pos, cost_map, max_radius_m=0.70)
        if candidates:
            _, col, row = candidates[0]
            return self.grid_to_world(col, row)
        return world_pos


    # Reference for pillar approach path helper code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
    def pure_pursuit_drive(self, rx, ry, rtheta):
        # Follow the path using pure pursuit.
        if not self.current_path:
            return True

        # Skip waypoints we already reached.
        while self.path_index < len(self.current_path) - 1:
            wx, wy = self.current_path[self.path_index]
            if math.hypot(wx - rx, wy - ry) < self.LOOKAHEAD_DISTANCE:
                self.path_index += 1
            else:
                break

        if self.path_index >= len(self.current_path):
            self.path_index = len(self.current_path) - 1

        target = self.current_path[-1]
        # Pick a point ahead on the path.
        for pt in self.current_path[self.path_index:]:
            if math.hypot(pt[0] - rx, pt[1] - ry) >= self.LOOKAHEAD_DISTANCE:
                target = pt
                break

        alpha = self.normalize_angle(
            math.atan2(target[1] - ry, target[0] - rx) - rtheta
        )

        v_target = self.MAX_LINEAR_SPEED * max(0.0, math.cos(alpha))

        omega_target = self.K_P_OMEGA * alpha
        omega_target = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, omega_target))

        if abs(alpha) > 1.0:
            # Turn first if the target is too far to the side.
            v_target = 0.0

        half_track = self.WHEEL_TRACK / 2.0

        v_left  = v_target - (omega_target * half_track)
        v_right = v_target + (omega_target * half_track)

        rads_left  = v_left  / self.WHEEL_RADIUS
        rads_right = v_right / self.WHEEL_RADIUS

        rads_left  = max(-self.MAX_MOTOR_RADS, min(self.MAX_MOTOR_RADS, rads_left))
        rads_right = max(-self.MAX_MOTOR_RADS, min(self.MAX_MOTOR_RADS, rads_right))

        self.fl.setVelocity(rads_left)
        self.rl.setVelocity(rads_left)
        self.fr.setVelocity(rads_right)
        self.rr.setVelocity(rads_right)

        final_wx, final_wy = self.current_path[-1]
        return math.hypot(final_wx - rx, final_wy - ry) < self.ARRIVAL_THRESHOLD

    def stop_robot(self):
        # Stop all wheels.
        self.fl.setVelocity(0.0)
        self.rl.setVelocity(0.0)
        self.fr.setVelocity(0.0)
        self.rr.setVelocity(0.0)

    def start_active_recovery(self, current_time, backoff_duration=0.8, turn_duration=1.0):
        # Start a backup and turn recovery move.
        self.RECOVERY_MODE = True
        self.recovery_start_time = current_time
        self.recovery_backoff_duration = backoff_duration
        self.recovery_turn_duration = turn_duration
        self.consecutive_stuck_ticks = 0
        self.current_path = []
        self.path_index = 0
        self.stop_robot()



    # Reference for path following and recovery code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
    # Main brain loop
    def step_brain(
            self, current_time, grid, rx, ry, rtheta, lidar_ranges, lidar_fov,
            mission_pillars=None, green_floor=None, extra_blocked=None):
        # One full navigation update.
        inflated_bool, cost_map = self.get_costmap(
            grid,
            green_floor=green_floor,
            extra_blocked=extra_blocked,
        )
        self._hard_blocked_mask = extra_blocked

        self._purge_relax_cooldowns(current_time)

        # Check for obstacles near the current path.
        LOOKAHEAD_WP = 15
        if self.current_path and self.path_index < len(self.current_path):
            # Start a few waypoints ahead so we do not panic about our own cell.
            start_check = min(self.path_index + 3, len(self.current_path))
            end = min(self.path_index + LOOKAHEAD_WP, len(self.current_path))
            
            # If the path ahead is blocked, clear the path and replan.
            for wp_wx, wp_wy in self.current_path[start_check:end]:
                wp_col, wp_row = self.world_to_grid(wp_wx, wp_wy)
                if 0 <= wp_col < self.MAP_SIZE and 0 <= wp_row < self.MAP_SIZE:
                    cell_cost = cost_map[wp_row, wp_col]
                    
                    # High cost means the path is too close to an obstacle.
                    if cell_cost == np.inf or cell_cost > 40.0:
                        print(f"[Navigator] Proximity Alert: Floating wall/obstacle too close to path (Cost: {cell_cost:.1f}). Replanning further away.")
                        self.current_path = []
                        self.path_index   = 0
                        # Reset intent so mission paths replan correctly.
                        if self.mission_state in (STATE_GO_BLUE, STATE_GO_YELLOW):
                            self.mission_path_intent = None
                        self.stop_robot()
                        break
        _pillars_early = mission_pillars or {}
        # Store the current mission target for display.
        if self.mission_state == STATE_GO_BLUE and _pillars_early.get('blue'):
            self.mission_target_pos = (
                _pillars_early['blue']['x'], _pillars_early['blue']['y']
            )
        elif self.mission_state == STATE_GO_YELLOW and _pillars_early.get('yellow'):
            self.mission_target_pos = (
                _pillars_early['yellow']['x'], _pillars_early['yellow']['y']
            )
        elif self.mission_state not in (STATE_GO_BLUE, STATE_GO_YELLOW):
            self.mission_target_pos = None

        if self.current_path:
            safety_hit = self.check_hardware_safety_envelope(lidar_ranges, lidar_fov)

            if not safety_hit:
                # Virtual bumper checks the map in front of the robot.
                forward_cells = self.ROBOT_RADIUS_CELLS + 2
                rx_idx, ry_idx = self.world_to_grid(rx, ry)
                hx, hy = math.cos(rtheta), math.sin(rtheta)
                for d in range(2, forward_cells):
                    cx = int(rx_idx + hx * d)
                    cy = int(ry_idx + hy * d)
                    if 0 <= cx < self.MAP_SIZE and 0 <= cy < self.MAP_SIZE:
                        if grid[cy, cx] > 0.85:
                            safety_hit = True
                            print("[Navigator] Virtual Bumper Hit! (Floating obstacle detected)")
                            break

            if safety_hit:
                self._safety_breach_count = getattr(self, '_safety_breach_count', 0) + 1
            else:
                self._safety_breach_count = 0

            last_replan = getattr(self, '_last_safety_replan_time', -1e9)
            # Wait for a few hits before forcing a replan.
            if self._safety_breach_count >= 3 and (current_time - last_replan) > 2.5:
                print("[Navigator] Safety envelope breached — emergency brake & replan.")
                self.current_path = []
                self.path_index   = 0
                self.stop_robot()
                self._last_safety_replan_time = current_time
                self._safety_breach_count = 0

        if self.current_path and not getattr(self, 'RECOVERY_MODE', False):
            # If the robot is not moving, start recovery.
            dist_moved   = math.hypot(rx - self.last_stuck_xy[0], ry - self.last_stuck_xy[1])
            time_elapsed = current_time - self.last_stuck_time

            if time_elapsed >= 1.2:
                if dist_moved < 0.03:
                    self.consecutive_stuck_ticks += 1
                    print(f"[Navigator] Potential slip detected ({self.consecutive_stuck_ticks}/3)")
                else:
                    self.consecutive_stuck_ticks = 0
                
                self.last_stuck_time = current_time
                self.last_stuck_xy   = (rx, ry)

            if self.consecutive_stuck_ticks >= 3:
                print("[Navigator] WATCHDOG: Detected wheel slip! Entering active recovery.")
                self.start_active_recovery(current_time)
                return
        elif not getattr(self, 'RECOVERY_MODE', False):
            self.last_stuck_time = current_time
            self.last_stuck_xy   = (rx, ry)

        if getattr(self, 'RECOVERY_MODE', False):
            # Recovery backs up, turns, then returns to normal planning.
            time_in_recovery = current_time - self.recovery_start_time
            linear_rads  = self.MAX_LINEAR_SPEED / self.WHEEL_RADIUS
            angular_rads = (self.MAX_ANGULAR_SPEED * (self.WHEEL_TRACK / 2.0)) / self.WHEEL_RADIUS

            if time_in_recovery < self.recovery_backoff_duration:
                v_rads = -linear_rads
                self.fl.setVelocity(v_rads); self.rl.setVelocity(v_rads)
                self.fr.setVelocity(v_rads); self.rr.setVelocity(v_rads)
                return
            elif time_in_recovery < (self.recovery_backoff_duration + self.recovery_turn_duration):
                self.fl.setVelocity(angular_rads);  self.rl.setVelocity(angular_rads)
                self.fr.setVelocity(-angular_rads); self.rr.setVelocity(-angular_rads)
                return
            else:
                print("[Navigator] Recovery maneuver complete. Resetting for new plan.")
                self.RECOVERY_MODE = False
                self.consecutive_stuck_ticks = 0
                self.stop_robot()

        pillars = mission_pillars or {}

        if (self.mission_state == STATE_EXPLORE
                and pillars.get('blue') is not None
                and pillars.get('yellow') is None):
            # If blue is found but yellow is not, spin to scan around.
            bl_count = len(self.blacklisted_frontiers)

            if (not self.yellow_scan_active
                    and bl_count >= self.YELLOW_SCAN_TRIGGER
                    and bl_count > self.yellow_scan_bl_count):
                print(f"[Navigator] YELLOW SCAN: starting 360° sweep "
                      f"(blacklisted={bl_count}).")
                self.yellow_scan_active  = True
                self.yellow_scan_start   = current_time
                self.yellow_scan_bl_count = bl_count
                self.current_path        = []
                self.path_index          = 0
                self.stop_robot()

            if self.yellow_scan_active:
                elapsed = current_time - self.yellow_scan_start
                if elapsed < self.YELLOW_SCAN_DURATION:
                    scan_rads = (self.MAX_ANGULAR_SPEED * 0.5) * (self.WHEEL_TRACK / 2.0) / self.WHEEL_RADIUS
                    self.fl.setVelocity(-scan_rads); self.rl.setVelocity(-scan_rads)
                    self.fr.setVelocity( scan_rads); self.rr.setVelocity( scan_rads)
                    return
                else:
                    print("[Navigator] YELLOW SCAN: complete. Resuming exploration.")
                    self.yellow_scan_active = False
                    self.yellow_scan_bl_count = len(self.blacklisted_frontiers)
                    self.stop_robot()

        if self.mission_state == STATE_FINISHED:
            # Mission done, so stop.
            self.stop_robot()
            return

        if self.mission_state == STATE_EXPLORE:
            # Explore until the pillars are found.
            blue_data   = pillars.get('blue')
            yellow_data = pillars.get('yellow')
            if blue_data is not None and yellow_data is not None:
                blue_raw  = (blue_data['x'], blue_data['y'])
                dist_to_blue = math.hypot(rx - blue_data['x'], ry - blue_data['y'])
                
                if dist_to_blue < self.PILLAR_MISSION_HANDOFF_THRESHOLD:
                    # If blue is close, go to it now.
                    print("[Navigator] MISSION: Blue is already nearby. Switching to STATE_GO_BLUE.")
                    self.mission_state       = STATE_GO_BLUE
                    self.current_path        = []
                    self.path_index          = 0
                    self.mission_path_intent = None
                    self.target_frontier_world = None
                    return
                elif not self.current_path:
                    test_path = self._direct_pillar_path(
                        (rx, ry), blue_raw, grid, cost_map,
                        allow_frontier_endpoint=True
                    )
                    if test_path or dist_to_blue < self.PILLAR_APPROACH_RADIUS:
                        # Both pillars are known, so start the mission.
                        print("[Navigator] MISSION: Both pillars found; Blue is reachable or nearby. Switching to STATE_GO_BLUE.")
                        self.mission_state       = STATE_GO_BLUE
                        self.current_path        = test_path
                        self.path_index          = 0
                        self.mission_path_intent = 'direct_blue'
                        self.target_frontier_world = None
                    else:
                        last_log = getattr(self, '_last_blue_unreachable_log_time', -1e9)
                        if current_time - last_log > 2.0:
                            self._last_blue_unreachable_log_time = current_time
                            print("[Navigator] MISSION: Both pillars known, but Blue has no confident mapped route yet. Exploring route-opening frontiers first.")
                            
        if self.mission_state == STATE_GO_BLUE:
            # First go to the blue pillar.
            blue_data = pillars.get('blue')
            if blue_data is None:
                # If blue is missing, go back to exploring.
                self.mission_state       = STATE_EXPLORE
                self.mission_path_intent = None
            else:
                blue_raw     = (blue_data['x'], blue_data['y'])
                dist_to_blue = math.hypot(rx - blue_data['x'], ry - blue_data['y'])
                self.mission_target_pos = blue_raw

                if dist_to_blue < self.BLUE_PILLAR_REACHED_THRESHOLD:
                    # Save a small path to back away from blue.
                    if self.current_path:
                        tail_end   = min(self.path_index + 1, len(self.current_path))
                        tail_start = max(0, tail_end - 10)
                        self._blue_departure_path = list(reversed(
                            self.current_path[tail_start:tail_end]
                        ))
                    print("[Navigator] MISSION: ★ BLUE pillar REACHED! "
                          "Switching to STATE_GO_YELLOW.")
                    self.mission_state       = STATE_GO_YELLOW
                    self.mission_target_pos  = None
                    self.current_path        = []
                    self.path_index          = 0
                    self.mission_path_intent = None
                    return
                elif dist_to_blue < self.PILLAR_ARRIVAL_THRESHOLD:
                    # Being close is not enough if the way is blocked.
                    last_log = getattr(self, '_last_blocked_blue_arrival_log_time', -1e9)
                    if current_time - last_log > 1.0:
                        self._last_blocked_blue_arrival_log_time = current_time
                        print("[Navigator] GO_BLUE: Close to blue, but approach is blocked/unknown — "
                              "not counting as reached.")

                if self.current_path:
                    # Clear paths that no longer make sense for blue.
                    last_wp = self.current_path[-1]
                    intent  = self.mission_path_intent
                    if intent == 'direct_blue':
                        if math.hypot(last_wp[0] - blue_data['x'],
                                      last_wp[1] - blue_data['y']) > (self.PILLAR_APPROACH_RADIUS + 0.05):
                            self.current_path        = []
                            self.mission_path_intent = None
                    elif intent != 'frontier_blue':
                        self.current_path        = []
                        self.mission_path_intent = None

                if not self.current_path:
                    # First try a routed path to blue.
                    direct = self._direct_pillar_path(
                        (rx, ry), blue_raw, grid, cost_map,
                        allow_frontier_endpoint=False
                    )
                    if not direct:
                        # Try again with a less strict check.
                        direct = self._direct_pillar_path(
                            (rx, ry), blue_raw, grid, cost_map,
                            allow_frontier_endpoint=True
                        )

                    if direct:
                        print("[Navigator] GO_BLUE: Routed path to Blue found — driving to pillar.")
                        self.current_path        = direct
                        self.path_index          = 0
                        self.mission_path_intent = 'direct_blue'
                        self.target_frontier_world = None
                    else:
                        # If no blue path exists, explore toward blue.
                        chosen_pt   = None
                        chosen_path = None

                        candidates = self.find_frontiers_toward(
                            grid, inflated_bool,
                            blue_data['x'], blue_data['y']
                        )
                        if not candidates:
                            # If needed, also try old blacklisted frontiers.
                            candidates = self.find_frontiers_toward(
                                grid, inflated_bool,
                                blue_data['x'], blue_data['y'],
                                ignore_blacklist=True
                            )
                        for cand_pt in candidates[:15]:
                            # Only test a few candidates to keep it fast.
                            cand_path = self.plan_path_dijkstra(
                                (rx, ry), cand_pt, cost_map
                            )
                            if cand_path:
                                chosen_pt, chosen_path = cand_pt, cand_path
                                break

                        # Last option is any frontier toward blue.
                        if chosen_path is None:
                            any_pt = self.find_frontier_toward(
                                grid, inflated_bool,
                                blue_data['x'], blue_data['y'],
                                ignore_blacklist=True
                            )
                            if any_pt is not None:
                                any_path = self.plan_path_dijkstra(
                                    (rx, ry), any_pt, cost_map
                                )
                                if any_path:
                                    chosen_pt, chosen_path = any_pt, any_path

                        if chosen_path:
                            print("[Navigator] GO_BLUE: No direct route to Blue yet — "
                                  "exploring frontier toward Blue.")
                            self.current_path          = chosen_path
                            self.path_index            = 0
                            self.mission_path_intent   = 'frontier_blue'
                            self.target_frontier_world = chosen_pt
                        else:
                            print("[Navigator] GO_BLUE: No path to Blue or toward Blue — waiting.")
                            self.stop_robot()

                # When close, use direct driving only if it is clear.
                if dist_to_blue < self.PILLAR_MISSION_HANDOFF_THRESHOLD:
                    # Make sure the direct line to blue is clear.
                    if self._pillar_arrival_is_clear((rx, ry), blue_raw, grid, cost_map):
                        arrived = self._direct_drive_to(rx, ry, rtheta, blue_raw[0], blue_raw[1])
                        if arrived or dist_to_blue < self.BLUE_PILLAR_REACHED_THRESHOLD:
                            if self.current_path:
                                tail_end   = min(self.path_index + 1, len(self.current_path))
                                tail_start = max(0, tail_end - 10)
                                self._blue_departure_path = list(reversed(self.current_path[tail_start:tail_end]))
                            print("[Navigator] MISSION: ★ BLUE pillar REACHED (direct proximity)! Switching to STATE_GO_YELLOW.")
                            self.mission_state       = STATE_GO_YELLOW
                            self.mission_target_pos  = None
                            self.current_path        = []
                            self.path_index          = 0
                            self.mission_path_intent = None
                        return
                    else:
                        last_log = getattr(self, '_last_handoff_blocked_blue', -1e9)
                        if current_time - last_log > 2.0:
                            print(f"[Navigator] GO_BLUE: Inside handoff radius ({dist_to_blue:.2f}m), but approach is blocked. Sticking to Dijkstra path.")
                            self._last_handoff_blocked_blue = current_time

                if self.current_path:
                    # Follow the current blue path.
                    arrived = self.pure_pursuit_drive(rx, ry, rtheta)
                    if arrived and self.mission_path_intent in ('direct_blue', 'forced_direct_blue'):
                        self.current_path        = []
                        self.path_index          = 0
                        self.mission_path_intent = None
                    elif arrived and self.mission_path_intent == 'frontier_blue':
                        self._blacklist_current_frontier("go_blue")
                        self.current_path        = []
                        self.path_index          = 0
                        self.mission_path_intent = None
                else:
                    # If there is no path, only move if blue is close and clear.
                    if (dist_to_blue < self.PILLAR_APPROACH_RADIUS
                            and self._pillar_arrival_is_clear((rx, ry), blue_raw, grid, cost_map)):
                        self._direct_drive_to(rx, ry, rtheta, blue_raw[0], blue_raw[1])
                    else:
                        self.stop_robot()
                return

        if self.mission_state == STATE_GO_YELLOW:
            # After blue, go to the yellow pillar.
            yellow_data = pillars.get('yellow')
            if yellow_data is None:
                # If yellow is missing, explore again.
                self.mission_state       = STATE_EXPLORE
                self.mission_path_intent = None
            else:
                yellow_raw     = (yellow_data['x'], yellow_data['y'])
                dist_to_yellow = math.hypot(rx - yellow_data['x'], ry - yellow_data['y'])
                self.mission_target_pos = yellow_raw

                # If we reached yellow, finish.
                if dist_to_yellow < self.YELLOW_PILLAR_REACHED_THRESHOLD:
                    print("[Navigator] MISSION: ★★ YELLOW pillar REACHED! "
                          "MISSION SUCCESSFUL ★★")
                    self.mission_state       = STATE_FINISHED
                    self.mission_target_pos  = None
                    self.mission_path_intent = None
                    self.current_path        = []
                    self.path_index          = 0
                    self.stop_robot()
                    return

                # When close, direct drive only if clear.
                if dist_to_yellow < self.PILLAR_MISSION_HANDOFF_THRESHOLD:
                    # Check that the direct line to yellow is clear.
                    if self._pillar_arrival_is_clear((rx, ry), yellow_raw, grid, cost_map):
                        arrived = self._direct_drive_to(rx, ry, rtheta, yellow_raw[0], yellow_raw[1])
                        if arrived or dist_to_yellow < self.YELLOW_PILLAR_REACHED_THRESHOLD:
                            print("[Navigator] MISSION: ★★ YELLOW pillar REACHED (direct proximity)! MISSION SUCCESSFUL ★★")
                            self.mission_state       = STATE_FINISHED
                            self.mission_target_pos  = None
                            self.current_path        = []
                            self.path_index          = 0
                            self.mission_path_intent = None
                            self.stop_robot()
                        return
                    else:
                        last_log = getattr(self, '_last_handoff_blocked_yellow', -1e9)
                        if current_time - last_log > 2.0:
                            print(f"[Navigator] GO_YELLOW: Inside handoff radius ({dist_to_yellow:.2f}m), "
                                  "approach blocked — routing via Dijkstra.")
                            self._last_handoff_blocked_yellow = current_time

                        # Continue to path planning below.
                # Remove old paths that no longer fit.
                if self.current_path:
                    last_wp = self.current_path[-1]
                    intent  = self.mission_path_intent
                    if intent == 'direct_yellow':
                        if math.hypot(last_wp[0] - yellow_data['x'],
                                      last_wp[1] - yellow_data['y']) > (self.PILLAR_APPROACH_RADIUS + 0.05):
                            self.current_path        = []
                            self.mission_path_intent = None
                    elif intent not in ('frontier_yellow', 'depart_blue'):
                        self.current_path        = []
                        self.mission_path_intent = None

                # Make a new yellow path if needed.
                if not self.current_path:
                    if self._blue_departure_path:
                        # Back away from blue first.
                        print("[Navigator] GO_YELLOW: Backing away from Blue first.")
                        self.current_path         = self._blue_departure_path
                        self._blue_departure_path = []
                        self.path_index           = 0
                        self.mission_path_intent  = 'depart_blue'
                    else:
                        # First try a routed path to yellow.
                        direct = self._direct_pillar_path(
                            (rx, ry), yellow_raw, grid, cost_map,
                            allow_frontier_endpoint=False
                        )
                        if not direct:
                            # Try again with a less strict check.
                            direct = self._direct_pillar_path(
                                (rx, ry), yellow_raw, grid, cost_map,
                                allow_frontier_endpoint=True
                            )

                        if direct:
                            print("[Navigator] GO_YELLOW: Routed path to Yellow found — driving to pillar.")
                            self.current_path        = direct
                            self.path_index          = 0
                            self.mission_path_intent = 'direct_yellow'
                            self.target_frontier_world = None
                        else:
                            # If no yellow path exists, explore toward yellow.
                            chosen_pt   = None
                            chosen_path = None

                            candidates = self.find_frontiers_toward(
                                grid, inflated_bool,
                                yellow_data['x'], yellow_data['y']
                            )
                            if not candidates:
                                # Try again while ignoring the blacklist.
                                candidates = self.find_frontiers_toward(
                                    grid, inflated_bool,
                                    yellow_data['x'], yellow_data['y'],
                                    ignore_blacklist=True
                                )
                            for cand_pt in candidates[:15]:
                                # Check only a few choices to keep it fast.
                                cand_path = self.plan_path_dijkstra(
                                    (rx, ry), cand_pt, cost_map
                                )
                                if cand_path:
                                    chosen_pt, chosen_path = cand_pt, cand_path
                                    break

                            # Last option is any frontier toward yellow.
                            if chosen_path is None:
                                any_pt = self.find_frontier_toward(
                                    grid, inflated_bool,
                                    yellow_data['x'], yellow_data['y'],
                                    ignore_blacklist=True
                                )
                                if any_pt is not None:
                                    any_path = self.plan_path_dijkstra(
                                        (rx, ry), any_pt, cost_map
                                    )
                                    if any_path:
                                        chosen_pt, chosen_path = any_pt, any_path

                            if chosen_path:
                                print("[Navigator] GO_YELLOW: No direct route to Yellow yet — "
                                      "exploring frontier toward Yellow.")
                                self.current_path          = chosen_path
                                self.path_index            = 0
                                self.mission_path_intent   = 'frontier_yellow'
                                self.target_frontier_world = chosen_pt
                            else:
                                print("[Navigator] GO_YELLOW: No path to Yellow or toward Yellow — waiting.")
                                self.stop_robot()

                if self.current_path:
                    # Drive the current yellow path.
                    arrived = self.pure_pursuit_drive(rx, ry, rtheta)
                    if arrived and self.mission_path_intent == 'depart_blue':
                        self.current_path        = []
                        self.path_index          = 0
                        self.mission_path_intent = None
                    elif arrived and self.mission_path_intent == 'direct_yellow':
                        # Check again if yellow is really reached.
                        if dist_to_yellow < self.YELLOW_PILLAR_REACHED_THRESHOLD:
                            print("[Navigator] MISSION: ★★ YELLOW pillar REACHED (path end)! MISSION SUCCESSFUL ★★")
                            self.mission_state       = STATE_FINISHED
                            self.mission_target_pos  = None
                            self.current_path        = []
                            self.path_index          = 0
                            self.mission_path_intent = None
                            self.stop_robot()
                            return
                        self.current_path        = []
                        self.path_index          = 0
                        self.mission_path_intent = None
                    elif arrived and self.mission_path_intent == 'frontier_yellow':
                        self._blacklist_current_frontier("go_yellow")
                        self.current_path        = []
                        self.path_index          = 0
                        self.mission_path_intent = None
                else:
                    self.stop_robot()
                return

        if not self.current_path:
            # Normal exploration path planning.
            blue_known   = pillars.get('blue')   is not None
            yellow_known = pillars.get('yellow') is not None
            biased_pt    = None
            
            if blue_known and yellow_known:
                # If both pillars are known, explore toward blue.
                blue_pos  = pillars['blue']
                biased_pt = self.find_frontier_toward(
                    grid, inflated_bool, blue_pos['x'], blue_pos['y']
                )
                if biased_pt is None:
                    biased_pt = self.find_frontier_toward(
                        grid, inflated_bool, blue_pos['x'], blue_pos['y'],
                        ignore_blacklist=True
                    )

            elif blue_known and not yellow_known:
                # If only blue is known, explore near blue but not too close.
                blue_pos  = pillars['blue']
                all_cands = self.find_frontiers_toward(
                    grid, inflated_bool, blue_pos['x'], blue_pos['y']
                )
                for cand in reversed(all_cands):
                    if math.hypot(cand[0] - rx, cand[1] - ry) < 0.3:
                        continue
                    test_path = self.plan_path_dijkstra((rx, ry), cand, cost_map)
                    if test_path:
                        biased_pt = cand
                        break

            force_global = (self.local_frontier_picks >= 6)

            if biased_pt is not None:
                # Use mission-related frontier if we found one.
                self.target_frontier_world = biased_pt
                self.local_frontier_picks  = 0
            else:
                self.target_frontier_world = self.find_best_frontier(
                    grid, inflated_bool, rx, ry, force_global=force_global
                )
                self.local_frontier_picks += 1

            if self.target_frontier_world is None:
                # If nothing is left, try ignoring blacklist once.
                if self.blacklisted_frontiers:
                    relaxed_pt = self.find_best_frontier(
                        grid, inflated_bool, rx, ry,
                        force_global=True,
                        ignore_blacklist=True
                    )
                    if relaxed_pt is not None:
                        self.target_frontier_world = relaxed_pt
                        self.local_frontier_picks = 0

            if self.target_frontier_world is None:
                # If there are no frontiers, try mission fallback paths.
                if blue_known and yellow_known:
                    blue_pos = pillars['blue']
                    yellow_pos = pillars['yellow']
                    
                    dist_to_yellow = math.hypot(rx - yellow_pos['x'], ry - yellow_pos['y'])
                    if dist_to_yellow < self.PILLAR_MISSION_HANDOFF_THRESHOLD:
                        # If yellow is already close, go to yellow.
                        self.mission_state       = STATE_GO_YELLOW
                        self.current_path        = []
                        self.path_index          = 0
                        self.mission_path_intent = None
                        self.target_frontier_world = None
                        return

                    fallback_direct = self._direct_pillar_path(
                        (rx, ry), (blue_pos['x'], blue_pos['y']),
                        grid, cost_map, allow_frontier_endpoint=True
                    )
                    if fallback_direct:
                        self.mission_state       = STATE_GO_BLUE
                        self.current_path        = fallback_direct
                        self.path_index          = 0
                        self.mission_path_intent = 'direct_blue'
                        self.target_frontier_world = None
                        self.local_frontier_picks = 0
                        return
                self.stop_robot()
                return

            self.current_path = self.plan_path_dijkstra(
                (rx, ry), self.target_frontier_world, cost_map
            )
            self.path_index     = 0
            self.last_plan_time = current_time

            if not self.current_path:
                # Count failed plans so we can stop retrying bad frontiers.
                bad_col, bad_row = self.world_to_grid(
                    self.target_frontier_world[0],
                    self.target_frontier_world[1]
                )
                
                retry_count = getattr(self, '_frontier_retry_counts', {})
                key = (bad_col, bad_row)
                retry_count[key] = retry_count.get(key, 0) + 1
                self._frontier_retry_counts = retry_count
                
                if retry_count[key] >= 4:
                    self._add_blacklisted_frontier(bad_col, bad_row)
                    self._mark_permanently_unreachable(bad_col, bad_row, current_time)
                
                self.target_frontier_world = None
                self.local_frontier_picks  = 0
                self.stop_robot()
                return

        # Drive along the selected path.
        arrived = self.pure_pursuit_drive(rx, ry, rtheta)
        if arrived:
            self._blacklist_current_frontier("explore")
            self.current_path = []
            self.path_index   = 0
