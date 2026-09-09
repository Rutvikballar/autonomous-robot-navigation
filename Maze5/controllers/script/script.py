from controller import Robot, Keyboard, Camera
import math
import numpy as np
import pygame
import colorsys
import cv2


# Import the navigator class from the other file.
from navigator import AutonomousNavigator


try:
    from config import * # noqa: F403,F401
except ImportError:
    pass

# --- INIT ---
# Main robot object from Webots.
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Keyboard is enabled in case we want to test manual driving.
keyboard = Keyboard()
keyboard.enable(timestep)

# --- CONSTANTS ---
# Robot size and map settings.
WHEEL_RADIUS = globals().get("WHEEL_RADIUS", 0.0425)
WHEEL_TRACK = globals().get("WHEEL_TRACK", 0.194)

MAP_SIZE = globals().get("MAP_SIZE", 600)
RESOLUTION = globals().get("RESOLUTION", 0.02)

# --- LOG-ODDS CONSTANTS ---
# Positive values mean wall, negative values mean free space.
L_OCCUPIED = globals().get("L_OCCUPIED", 0.85)
L_FREE = globals().get("L_FREE", -0.15)
L_MAX = globals().get("L_MAX", 5.0)
L_MIN = globals().get("L_MIN", -5.0)
L_WALL_LOCK = globals().get("L_WALL_LOCK", 3.2)

# --- PILLAR DETECTION CONSTANTS ---
# These values help the camera find the blue and yellow pillars.
CAMERA_HEIGHT_METERS = globals().get("CAMERA_HEIGHT_METERS", 0.2)
PILLAR_RADIUS_METERS = globals().get("PILLAR_RADIUS_METERS", 0.1)
DISTANCE_DISCREPANCY_THRESHOLD = globals().get("DISTANCE_DISCREPANCY_THRESHOLD", 0.5)
MAX_PILLAR_WIDTH = globals().get("MAX_PILLAR_WIDTH", 0.5)

BLUE_HSV_LOWER = globals().get("BLUE_HSV_LOWER", (0.55, 0.5, 0.5))
BLUE_HSV_UPPER = globals().get("BLUE_HSV_UPPER", (0.75, 1.0, 1.0))
YELLOW_HSV_LOWER = globals().get("YELLOW_HSV_LOWER", (0.1, 0.5, 0.5))
YELLOW_HSV_UPPER = globals().get("YELLOW_HSV_UPPER", (0.25, 1.0, 1.0))
GREEN_HSV_LOWER = globals().get("GREEN_HSV_LOWER", (0.25, 0.3, 0.2))
GREEN_HSV_UPPER = globals().get("GREEN_HSV_UPPER", (0.45, 1.0, 1.0))


# --- SCAN MATCHING (MAP-BASED POSE CORRECTION) ---
# This helps correct small position errors while the robot is mapping.
SCAN_MATCH_INTERVAL = globals().get("SCAN_MATCH_INTERVAL", 5)
SCAN_MATCH_BEAM_STEP = globals().get("SCAN_MATCH_BEAM_STEP", 6)
SCAN_MATCH_DXY = globals().get("SCAN_MATCH_DXY", np.array([-0.03, 0.0, 0.03]))
SCAN_MATCH_DTH = globals().get("SCAN_MATCH_DTH", np.deg2rad(np.array([-3.0, 0.0, 3.0])))
SCAN_MATCH_MIN_POINTS = globals().get("SCAN_MATCH_MIN_POINTS", 20)
SCAN_MATCH_MIN_KNOWN_RATIO = globals().get("SCAN_MATCH_MIN_KNOWN_RATIO", 0.15)
SCAN_MATCH_MIN_IMPROVEMENT = globals().get("SCAN_MATCH_MIN_IMPROVEMENT", 0.04)
SCAN_MATCH_KNOWN_EPS = globals().get("SCAN_MATCH_KNOWN_EPS", 0.2)

# --- ROTATION GATING ---
# We skip mapping when the robot is turning too much.
ROTATION_GATING_THRESHOLD = globals().get("ROTATION_GATING_THRESHOLD", 0.05)

# Give sensors a few steps to settle before using them.
SENSOR_WARMUP_STEPS = globals().get("SENSOR_WARMUP_STEPS", 15)

# --- GRID ---
# These arrays store the map and extra danger layers.
grid = np.ones((MAP_SIZE, MAP_SIZE)) * 0.5
locked_walls = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
lidar_only_walls = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
green_floor = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
floating_obstacles = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
floating_obstacle_memory = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
floating_back_padding = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
floating_detection_dots = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
floating_detection_age = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
floating_settled_zone = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
low_bump_votes = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)

# We only trust a pillar after seeing it a few times.
mission_pillars = {
    'blue': None,
    'yellow': None
}
pillar_candidates = {
    'blue': [],
    'yellow': []
}
PILLAR_REQUIRED_DETECTIONS = globals().get("PILLAR_REQUIRED_DETECTIONS", 2)

# Last movement values, used by the tilt safety check.
odom_dx = 0.0
odom_dy = 0.0
odom_dtheta = 0.0

# --- TILT / MISSED-OBSTACLE FALLBACK ---
# If the robot tilts, we mark that spot as blocked.
TILT_BASELINE_WARMUP_STEPS = globals().get("TILT_BASELINE_WARMUP_STEPS", SENSOR_WARMUP_STEPS)
TILT_TRIGGER_DEG = globals().get("TILT_TRIGGER_DEG", 27.0)
TILT_RELEASE_DEG = globals().get("TILT_RELEASE_DEG", 18.0)
TILT_CONFIRM_STEPS = globals().get("TILT_CONFIRM_STEPS", 3)
NOSE_UP_TRIGGER_DEG = globals().get("NOSE_UP_TRIGGER_DEG", 5.0)
TILT_MARK_COOLDOWN_STEPS = globals().get("TILT_MARK_COOLDOWN_STEPS", 20)
TILT_FRONT_CONTACT_M = globals().get("TILT_FRONT_CONTACT_M", 0.22)
TILT_SIDE_CONTACT_M = globals().get("TILT_SIDE_CONTACT_M", 0.18)
TILT_PATCH_LENGTH_M = globals().get("TILT_PATCH_LENGTH_M", 0.16)
TILT_PATCH_WIDTH_M = globals().get("TILT_PATCH_WIDTH_M", 0.24)

tilt_baseline_roll = 0.0
tilt_baseline_pitch = 0.0
tilt_baseline_samples = 0
tilt_over_threshold_steps = 0
tilt_latched = False
last_tilt_mark_step = -10**9

# --- PYGAME VISUALIZATION ---
# Pygame is used to show the map while the robot is running.
pygame.init()
SCREEN_SIZE = globals().get("SCREEN_SIZE", 600)
PYGAME_TARGET_FPS = globals().get("PYGAME_TARGET_FPS", 20.0)
PYGAME_MIN_FRAME_DT = globals().get("PYGAME_MIN_FRAME_DT", 1.0 / PYGAME_TARGET_FPS)
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('RosBot Mapping (Pygame)')

# --- MOTORS ---
# Get all four wheel motors.
fl = robot.getDevice('fl_wheel_joint')
fr = robot.getDevice('fr_wheel_joint')
rl = robot.getDevice('rl_wheel_joint')
rr = robot.getDevice('rr_wheel_joint')

for m in [fl, fr, rl, rr]:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# INITIALIZE NAVIGATOR
# The navigator decides where the robot should go.
motors = (fl, fr, rl, rr)
brain = AutonomousNavigator(motors, WHEEL_TRACK, MAP_SIZE, RESOLUTION)

# --- ENCODERS ---
# Encoders tell us how much each wheel moved.
fl_enc = robot.getDevice('front left wheel motor sensor')
fr_enc = robot.getDevice('front right wheel motor sensor')
rl_enc = robot.getDevice('rear left wheel motor sensor')
rr_enc = robot.getDevice('rear right wheel motor sensor')

for enc in [fl_enc, fr_enc, rl_enc, rr_enc]:
    enc.enable(timestep)

# --- IMU ---
imu = robot.getDevice('imu inertial_unit')
imu.enable(timestep)

# --- LIDAR ---
# Lidar is used to find walls and free space.
lidar = robot.getDevice('laser')
lidar.enable(timestep)
lidar.enablePointCloud()




# --- CAMERA ---
# RGB camera finds colours and depth camera gives distance.
camera_rgb = robot.getDevice('camera rgb')
camera_rgb.enable(timestep)

camera_depth = robot.getDevice('camera depth')
if camera_depth:
    camera_depth.enable(timestep)

camera_width = camera_rgb.getWidth()
camera_height = camera_rgb.getHeight()
camera_fov_horizontal = camera_rgb.getFov()
camera_fov_vertical = 2 * math.atan(math.tan(camera_fov_horizontal / 2) * (camera_height / camera_width))

# --- INIT POSE ---
# Start position of the robot in our map.
x, y, theta = 0.0, 0.0, 0.0

# Step once so sensor values are ready.
robot.step(timestep)

prev_fl = fl_enc.getValue()
prev_fr = fr_enc.getValue()
prev_rl = rl_enc.getValue()
prev_rr = rr_enc.getValue()

# --- HELPERS ---
# Convert real position to grid position.
def world_to_grid(x, y):
    gx = int(x / RESOLUTION) + MAP_SIZE // 2
    gy = int(y / RESOLUTION) + MAP_SIZE // 2
    return gx, gy

# Keep angle values in a normal range.
def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def begin_tilt_recovery(label):
    """Immediately enter the same active recovery used for a confirmed slip."""
    brain.start_active_recovery(robot.getTime())
    print(f"[TiltGuard] {label}: immediate active recovery triggered.")

def mark_tilt_contact_patch(robot_x, robot_y, robot_theta, local_x, local_y, label):
    """Stamp a conservative blocked patch where a missed obstacle likely lifted the robot."""
    global grid, locked_walls

    # Make a small blocked patch where the robot probably touched the obstacle.
    half_len = TILT_PATCH_LENGTH_M / 2.0
    half_wid = TILT_PATCH_WIDTH_M / 2.0
    sample_step = max(RESOLUTION * 0.5, 0.01)

    along = np.arange(-half_len, half_len + sample_step, sample_step)
    across = np.arange(-half_wid, half_wid + sample_step, sample_step)
    aa, bb = np.meshgrid(along, across)

    patch_local_x = local_x + aa.ravel()
    patch_local_y = local_y + bb.ravel()

    cos_th = math.cos(robot_theta)
    sin_th = math.sin(robot_theta)
    patch_world_x = robot_x + patch_local_x * cos_th - patch_local_y * sin_th
    patch_world_y = robot_y + patch_local_x * sin_th + patch_local_y * cos_th

    gx = (patch_world_x / RESOLUTION + MAP_SIZE // 2).astype(np.int32)
    gy = (patch_world_y / RESOLUTION + MAP_SIZE // 2).astype(np.int32)
    valid = (gx >= 0) & (gx < MAP_SIZE) & (gy >= 0) & (gy < MAP_SIZE)
    if not np.any(valid):
        return

    gx = gx[valid]
    gy = gy[valid]
    # Lock this patch so the robot remembers it as blocked.
    grid[gy, gx] = L_MAX
    locked_walls[gy, gx] = True
    begin_tilt_recovery(label)
    print(f"[TiltGuard] {label}: marked missed-obstacle patch at probable contact zone.")

def detect_tilted_contact_obstacle(robot_x, robot_y, robot_theta):
    """
    Use sustained pitch/roll deviation from the startup baseline as evidence that
    the robot climbed onto an obstacle the mapper failed to represent.
    """
    global tilt_baseline_roll, tilt_baseline_pitch, tilt_baseline_samples
    global tilt_over_threshold_steps, tilt_latched, last_tilt_mark_step

    roll, pitch, _ = imu.getRollPitchYaw()

    # First learn the normal flat angle of the robot.
    if tilt_baseline_samples < TILT_BASELINE_WARMUP_STEPS:
        n = tilt_baseline_samples
        tilt_baseline_roll = ((tilt_baseline_roll * n) + roll) / (n + 1)
        tilt_baseline_pitch = ((tilt_baseline_pitch * n) + pitch) / (n + 1)
        tilt_baseline_samples += 1
        return

    roll_delta = normalize_angle(roll - tilt_baseline_roll)
    pitch_delta = normalize_angle(pitch - tilt_baseline_pitch)
    abs_roll_deg = abs(math.degrees(roll_delta))
    abs_pitch_deg = abs(math.degrees(pitch_delta))
    max_tilt_deg = max(abs_roll_deg, abs_pitch_deg)

    motion_along_heading = odom_dx * math.cos(robot_theta) + odom_dy * math.sin(robot_theta)

    # If the front lifts up, mark a small obstacle in front.
    if (
        abs_pitch_deg >= NOSE_UP_TRIGGER_DEG
        and abs_pitch_deg >= abs_roll_deg
        and motion_along_heading >= -1e-4
        and not tilt_latched
        and (step_count - last_tilt_mark_step) >= TILT_MARK_COOLDOWN_STEPS
    ):
        mark_tilt_contact_patch(
            robot_x,
            robot_y,
            robot_theta,
            TILT_FRONT_CONTACT_M,
            0.0,
            f"nose-up {math.degrees(pitch_delta):.1f} deg",
        )
        last_tilt_mark_step = step_count
        tilt_latched = True
        return

    if max_tilt_deg >= TILT_TRIGGER_DEG:
        tilt_over_threshold_steps += 1
    else:
        tilt_over_threshold_steps = 0

    if max_tilt_deg <= TILT_RELEASE_DEG:
        tilt_latched = False

    if (
        tilt_latched
        or tilt_over_threshold_steps < TILT_CONFIRM_STEPS
        or (step_count - last_tilt_mark_step) < TILT_MARK_COOLDOWN_STEPS
    ):
        return

    if abs_pitch_deg >= abs_roll_deg:
        # Pitch usually means front or back contact.
        local_x = TILT_FRONT_CONTACT_M if motion_along_heading >= -1e-4 else -TILT_FRONT_CONTACT_M
        local_y = 0.0
        label = f"pitch {math.degrees(pitch_delta):.1f} deg"
    else:
        # Roll usually means left or right contact.
        local_x = 0.0
        local_y = TILT_SIDE_CONTACT_M if roll_delta >= 0.0 else -TILT_SIDE_CONTACT_M
        label = f"roll {math.degrees(roll_delta):.1f} deg"

    mark_tilt_contact_patch(robot_x, robot_y, robot_theta, local_x, local_y, label)
    last_tilt_mark_step = step_count
    tilt_latched = True

def rgb_to_hsv(r, g, b):
    # Convert RGB values into HSV.
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

def is_color_in_range(hsv, lower, upper):
    # Check if the colour is inside the range we want.
    return (lower[0] <= hsv[0] <= upper[0]) and (lower[1] <= hsv[1] <= upper[1]) and (lower[2] <= hsv[2] <= upper[2])

def bresenham_cells(x0, y0, x1, y1):
    # Get all grid cells on a straight line.
    x, y = x0, y0
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    cells_x = []
    cells_y = []

    while True:
        cells_x.append(x)
        cells_y.append(y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.asarray(cells_x, dtype=np.int32), np.asarray(cells_y, dtype=np.int32)

def score_pose_candidate(cx, cy, ctheta, d, angles):
    # Check how well a guessed robot pose matches the map.
    obs_x = cx + d * np.cos(ctheta + angles)
    obs_y = cy + d * np.sin(ctheta + angles)
    gx = (obs_x / RESOLUTION + MAP_SIZE // 2).astype(np.int32)
    gy = (obs_y / RESOLUTION + MAP_SIZE // 2).astype(np.int32)

    in_bounds = (gx >= 0) & (gx < MAP_SIZE) & (gy >= 0) & (gy < MAP_SIZE)
    if np.count_nonzero(in_bounds) < SCAN_MATCH_MIN_POINTS:
        return -np.inf, 0.0

    cells = grid[gy[in_bounds], gx[in_bounds]]
    known_ratio = float(np.mean(np.abs(cells - 0.5) > SCAN_MATCH_KNOWN_EPS))
    score = float(np.mean(cells) + 0.2 * known_ratio)
    return score, known_ratio

def compute_scan_match_correction(ranges, fov, res):
    # Try small position corrections and keep the best one.
    beam_indices = np.arange(0, res, SCAN_MATCH_BEAM_STEP)
    d = ranges[beam_indices]
    valid = (d > 0.1) & (d < 7.0)
    d = d[valid]
    beam_indices = beam_indices[valid]

    if d.size < SCAN_MATCH_MIN_POINTS:
        return None

    angles = (fov / 2.0) - (beam_indices / res) * fov
    base_score, _ = score_pose_candidate(x, y, theta, d, angles)
    best_score = base_score
    best_delta = (0.0, 0.0, 0.0)

    for dtheta in SCAN_MATCH_DTH:
        for dx in SCAN_MATCH_DXY:
            for dy in SCAN_MATCH_DXY:
                if dx == 0.0 and dy == 0.0 and dtheta == 0.0:
                    continue
                trial_x = x + dx
                trial_y = y + dy
                trial_theta = theta + dtheta
                score, known_ratio = score_pose_candidate(trial_x, trial_y, trial_theta, d, angles)
                if known_ratio < SCAN_MATCH_MIN_KNOWN_RATIO:
                    continue
                if score > best_score:
                    best_score = score
                    best_delta = (dx, dy, dtheta)

    if best_score - base_score >= SCAN_MATCH_MIN_IMPROVEMENT:
        return best_delta
    return None


# Reference for scan matching code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
class Particle:
    # Small helper class for testing robot position.
    def __init__(self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

    def predict(self, dx, dy, dtheta, noise=[0.01, 0.01, 0.02]):
        # Add a bit of random noise to the movement.
        self.x += dx + np.random.normal(0, noise[0])
        self.y += dy + np.random.normal(0, noise[1])
        self.theta += dtheta + np.random.normal(0, noise[2])

    def update_weight(self, lidar_ranges, lidar_angles, occupancy_grid):
        score = 0
        # Check some lidar beams against the current map.
        for i in range(0, len(lidar_ranges), 10): # Use every 10th beam to make it faster.
            r = lidar_ranges[i]
            if r >= 7.0 or r <= 0.1: continue
            
            # Work out where this lidar hit would be.
            tx = self.x + r * math.cos(self.theta + lidar_angles[i])
            ty = self.y + r * math.sin(self.theta + lidar_angles[i])
            
            gx, gy = world_to_grid(tx, ty)
            
            if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                # Give more score if the hit lands on a wall.
                if occupancy_grid[gy, gx] > 0:
                    score += 1
        
        self.weight = score + 1e-6 # Stop the weight becoming zero.

# --- ODOMETRY ---
def update_odometry():
    # Update where we think the robot is.
    global x, y, theta
    global prev_fl, prev_fr, prev_rl, prev_rr
    global odom_dx, odom_dy, odom_dtheta

    # Get current encoder values.
    vals = [fl_enc.getValue(), fr_enc.getValue(), rl_enc.getValue(), rr_enc.getValue()]
    # Skip if a sensor value is not ready.
    if any(math.isnan(v) for v in vals): return

    # Calculate how far the left and right wheels moved.
    d_left = WHEEL_RADIUS * ((vals[0] - prev_fl) + (vals[2] - prev_rl)) / 2.0
    d_right = WHEEL_RADIUS * ((vals[1] - prev_fr) + (vals[3] - prev_rr)) / 2.0
    
    # Save values for the next loop.
    prev_fl, prev_fr, prev_rl, prev_rr = vals

    # Calculate the robot movement.
    d_center = (d_left + d_right) / 2.0
    
    # Use IMU yaw as the robot heading.
    new_theta = imu.getRollPitchYaw()[2]
    
    # Use an average angle so turns are smoother.
    avg_theta = theta + (new_theta - theta) / 2.0
    
    old_x, old_y, old_theta = x, y, theta
    x += d_center * math.cos(avg_theta)
    y += d_center * math.sin(avg_theta)
    theta = new_theta
    odom_dx = x - old_x
    odom_dy = y - old_y
    odom_dtheta = normalize_angle(theta - old_theta)


# --- LIDAR MAP ---
def update_map():
    # Update the map using lidar readings.
    global grid, x, y, theta, locked_walls, lidar_only_walls

    if step_count < SENSOR_WARMUP_STEPS:
        return

    # Get lidar data.
    ranges = np.array(lidar.getRangeImage())
    fov = lidar.getFov()
    res = len(ranges)

    # Slightly correct the pose if the scan matches the map better.
    if step_count % SCAN_MATCH_INTERVAL == 0:
        correction = compute_scan_match_correction(ranges, fov, res)
        if correction is not None:
            x += correction[0]
            y += correction[1]
            theta = normalize_angle(theta + correction[2])

    # Get robot grid position.
    gx_robot, gy_robot = world_to_grid(x, y)

    # Use every lidar beam.
    lidar_step = 1
    indices = np.arange(0, res, lidar_step)
    d = ranges[indices]

    # Ignore bad distances.
    mask = (d > 0.1) & (d < 7.0)
    d = d[mask]
    indices = indices[mask]

    if d.size == 0:
        return

    # Calculate where each lidar beam hits.
    angles = (fov / 2.0) - (indices / res) * fov
    obs_x = x + d * np.cos(theta + angles)
    obs_y = y + d * np.sin(theta + angles)

    # Convert hit points into grid cells.
    gx_hits = (obs_x / RESOLUTION + MAP_SIZE // 2).astype(int)
    gy_hits = (obs_y / RESOLUTION + MAP_SIZE // 2).astype(int)

    # Mark lidar hit points as occupied.
    valid_hits = (gx_hits >= 0) & (gx_hits < MAP_SIZE) & (gy_hits >= 0) & (gy_hits < MAP_SIZE)
    if np.any(valid_hits):
        hit_x = gx_hits[valid_hits]
        hit_y = gy_hits[valid_hits]
        unlocked_hit_mask = ~locked_walls[hit_y, hit_x]
        if np.any(unlocked_hit_mask):
            hit_x = hit_x[unlocked_hit_mask]
            hit_y = hit_y[unlocked_hit_mask]
            np.add.at(grid, (hit_y, hit_x), L_OCCUPIED)
            new_lock = (grid[hit_y, hit_x] >= L_WALL_LOCK)
            locked_walls[hit_y, hit_x] |= new_lock
            lidar_only_walls[hit_y, hit_x] |= new_lock


    # Mark cells before the hit as free space.
    n_valid = int(np.count_nonzero(valid_hits))
    if n_valid > 0:
        rx_g = int(gx_robot)
        ry_g = int(gy_robot)

        for hx, hy in zip(gx_hits[valid_hits], gy_hits[valid_hits]):
            ray_x, ray_y = bresenham_cells(rx_g, ry_g, int(hx), int(hy))
            if ray_x.size <= 1:
                continue

            ray_x = ray_x[:-1]
            ray_y = ray_y[:-1]

            in_b = (ray_x >= 0) & (ray_x < MAP_SIZE) & (ray_y >= 0) & (ray_y < MAP_SIZE)
            ray_x = ray_x[in_b]
            ray_y = ray_y[in_b]
            if ray_x.size == 0:
                continue

            # Stop clearing when we hit a real lidar wall.
            hit_lidar_wall = lidar_only_walls[ray_y, ray_x]
            if np.any(hit_lidar_wall):
                first_lock = int(np.argmax(hit_lidar_wall))
                ray_x = ray_x[:first_lock]
                ray_y = ray_y[:first_lock]

            if ray_x.size > 0:
                free_update = ~locked_walls[ray_y, ray_x]
                if np.any(free_update):
                    np.add.at(grid, (ray_y[free_update], ray_x[free_update]), L_FREE)

    # Keep map values inside the allowed range.
    grid = np.clip(grid, L_MIN, L_MAX)

    # Fill tiny unknown gaps inside open space.
    for _ in range(2):
        pad = np.pad(grid, 1, mode='edge')
        neighbor_free = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                neighbor_free += (pad[dy:dy+MAP_SIZE, dx:dx+MAP_SIZE] < -0.5).astype(np.float32)
        gap = (grid > -0.5) & (grid < 0.3) & (neighbor_free >= 4) & (~locked_walls)
        grid[gap] = -1.0


# Reference for lidar mapping code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
def detect_and_mark_pillars(robot_x, robot_y, robot_theta):
    # Find blue or yellow pillars and add them to the map.
    global grid, locked_walls, lidar_only_walls
    global mission_pillars, pillar_candidates, PILLAR_REQUIRED_DETECTIONS

    if all(p is not None for p in mission_pillars.values()):
        return

    image = camera_rgb.getImage()
    if not image:
        return

    blue_pixels, yellow_pixels = [], []
    sample_step = 5

    # Sample some pixels instead of every pixel to save time.
    for px in range(0, camera_width, sample_step):
        for py in range(0, camera_height, sample_step):
            r = Camera.imageGetRed(image, camera_width, px, py)
            g = Camera.imageGetGreen(image, camera_width, px, py)
            b = Camera.imageGetBlue(image, camera_width, px, py)

            hsv = rgb_to_hsv(r, g, b)

            if is_color_in_range(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER):
                blue_pixels.append((px, py))
            elif is_color_in_range(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER):
                yellow_pixels.append((px, py))

    detected_color, pixels = None, []
    min_pixels = 10

    if len(blue_pixels) > len(yellow_pixels) and len(blue_pixels) > min_pixels:
        detected_color, pixels = "blue", blue_pixels
    elif len(yellow_pixels) > len(blue_pixels) and len(yellow_pixels) > min_pixels:
        detected_color, pixels = "yellow", yellow_pixels
    else:
        return

    if mission_pillars[detected_color] is not None:
        return

    # Estimate distance using the lowest coloured pixel.
    base_pixel = max(pixels, key=lambda p: p[1])
    pixel_y_from_center = base_pixel[1] - (camera_height / 2.0)

    angle_down = (
        pixel_y_from_center / (camera_height / 2.0)
    ) * (camera_fov_vertical / 2.0)

    if angle_down <= 0.05:
        return

    camera_dist_estimate = CAMERA_HEIGHT_METERS / math.tan(angle_down)

    # Estimate the left/right angle of the coloured object.
    avg_pixel_x = sum(p[0] for p in pixels) / len(pixels)
    angle_horizontal = (0.5 - (avg_pixel_x / camera_width)) * camera_fov_horizontal

    lidar_points = lidar.getPointCloud()
    cone_angle = math.radians(2.0)

    candidate_indices = []

    # Find lidar points in the same direction as the colour blob.
    for i, p in enumerate(lidar_points):
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            continue

        p_angle = math.atan2(p.y, p.x)

        # Compare angles safely.
        angle_diff = math.atan2(
            math.sin(p_angle - angle_horizontal),
            math.cos(p_angle - angle_horizontal)
        )

        if abs(angle_diff) < cone_angle:
            candidate_indices.append(i)

    if not candidate_indices:
        return

    def point_dist(pt):
        return math.sqrt(pt.x ** 2 + pt.y ** 2)

    def point_gap(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    candidate_indices.sort(key=lambda i: point_dist(lidar_points[i]))

    # Use the middle point so one noisy edge does not fool us.
    best_index = candidate_indices[len(candidate_indices) // 2]
    best_pt = lidar_points[best_index]
    lidar_dist = point_dist(best_pt)

    if abs(camera_dist_estimate - lidar_dist) > DISTANCE_DISCREPANCY_THRESHOLD:
        print(
            f"Ignored {detected_color}: "
            f"Mismatch Cam {camera_dist_estimate:.2f}m vs LiDAR {lidar_dist:.2f}m"
        )
        return

    num_pts = len(lidar_points)
    left_idx, right_idx = best_index, best_index
    max_search = int(num_pts / 8)

    # Find the left edge of the object.
    for _ in range(max_search):
        prev = (left_idx - 1) % num_pts

        p_curr = lidar_points[left_idx]
        p_prev = lidar_points[prev]

        if not (
            math.isfinite(p_prev.x)
            and math.isfinite(p_prev.y)
            and math.isfinite(p_curr.x)
            and math.isfinite(p_curr.y)
        ):
            break

        d_curr = point_dist(p_curr)
        d_prev = point_dist(p_prev)
        gap = point_gap(p_curr, p_prev)

        if abs(d_curr - d_prev) > 0.2 or gap > 0.15:
            break

        left_idx = prev

    # Find the right edge of the object.
    for _ in range(max_search):
        nxt = (right_idx + 1) % num_pts

        p_curr = lidar_points[right_idx]
        p_next = lidar_points[nxt]

        if not (
            math.isfinite(p_next.x)
            and math.isfinite(p_next.y)
            and math.isfinite(p_curr.x)
            and math.isfinite(p_curr.y)
        ):
            break

        d_curr = point_dist(p_curr)
        d_next = point_dist(p_next)
        gap = point_gap(p_curr, p_next)

        if abs(d_curr - d_next) > 0.2 or gap > 0.15:
            break

        right_idx = nxt

    p_l = lidar_points[left_idx]
    p_r = lidar_points[right_idx]

    width = point_gap(p_l, p_r)

    # Check if the width looks like a pillar.
    if width < 0.06 or width > 0.35:
        print(
            f"Ignored {detected_color}: "
            f"Object width invalid ({width:.2f}m). Likely a wall."
        )
        return

    # Check if the object is curved, not flat.
    if width < 0.01:
        return

    numerator = abs(
        (p_r.x - p_l.x) * (p_l.y - best_pt.y)
        - (p_l.x - best_pt.x) * (p_r.y - p_l.y)
    )

    bulge = numerator / width

    if bulge < 0.0015:
        print(
            f"Ignored {detected_color}: "
            f"Object is flat. Bulge: {bulge:.3f}m. Likely a colored wall/poster."
        )
        return

    pillar_global_angle = robot_theta + angle_horizontal

    # Lidar sees the front of the pillar, so add the radius to estimate the centre.
    pillar_center_dist = lidar_dist + PILLAR_RADIUS_METERS
    pillar_x = robot_x + pillar_center_dist * math.cos(pillar_global_angle)
    pillar_y = robot_y + pillar_center_dist * math.sin(pillar_global_angle)

    pillar_candidates[detected_color].append((pillar_x, pillar_y))

    print(
        f"CANDIDATE {detected_color.upper()} PILLAR at "
        f"({pillar_x:.2f}, {pillar_y:.2f}) "
        f"[Surface: {lidar_dist:.2f}m, Center: {pillar_center_dist:.2f}m, "
        f"Width: {width:.2f}m, Bulge: {bulge:.3f}m]"
    )

    if len(pillar_candidates[detected_color]) >= PILLAR_REQUIRED_DETECTIONS:
        # Average all sightings so the pillar position is more stable.
        avg_x = sum(p[0] for p in pillar_candidates[detected_color]) / len(
            pillar_candidates[detected_color]
        )
        avg_y = sum(p[1] for p in pillar_candidates[detected_color]) / len(
            pillar_candidates[detected_color]
        )

        print(f"CONFIRMED {detected_color.upper()} PILLAR at ({avg_x:.2f}, {avg_y:.2f})")

        mission_pillars[detected_color] = {
            "x": avg_x,
            "y": avg_y
        }
        clear_floating_padding_near_pillars()

        mx, my = world_to_grid(avg_x, avg_y)
        rad = int(PILLAR_RADIUS_METERS / RESOLUTION)

        # Mark the pillar as a small solid circle on the map.
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dx ** 2 + dy ** 2 <= rad ** 2:
                    tx = mx + dx
                    ty = my + dy

                    if 0 <= tx < MAP_SIZE and 0 <= ty < MAP_SIZE:
                        grid[ty][tx] = L_MAX
                        locked_walls[ty][tx] = True
                        lidar_only_walls[ty][tx] = True


# Reference for pillar detection code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
def detect_green_floor(robot_x, robot_y, robot_theta):
    # Detect green floor and mark it as dangerous.
    global green_floor

    if step_count % 3 != 0:
        return

    image = camera_rgb.getImage()
    if not image or not camera_depth:
        return

    depth_array = camera_depth.getRangeImage()
    if not depth_array:
        return

    img_array = np.frombuffer(image, np.uint8).reshape((camera_height, camera_width, 4))
    # Only use the lower part of the image where floor is visible.
    y_start = int(camera_height * 0.35)
    bgr_img = img_array[y_start:, :, 0:3].copy()

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # Use a wider green range because lighting changes the colour a bit.
    lower_green = np.array([38, 55, 35])
    upper_green = np.array([81, 255, 255])

    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    valid_y_local, valid_x = np.where(green_mask > 0)
    if valid_x.size == 0:
        return

    valid_y = valid_y_local + y_start

    # Use depth to turn pixels into robot positions.
    depth_indices = valid_y * camera_width + valid_x
    z_forward = np.array([depth_array[i] for i in depth_indices], dtype=np.float32)

    valid_depth = np.isfinite(z_forward) & (z_forward > 0.05) & (z_forward < 3.0)

    valid_x = valid_x[valid_depth]
    valid_y = valid_y[valid_depth]
    z_forward = z_forward[valid_depth]

    if valid_x.size == 0:
        return

    cx = camera_width / 2.0
    fx = cx / math.tan(camera_fov_horizontal / 2.0)

    ray_x_norm = (valid_x - cx) / fx
    rob_x = z_forward
    rob_y = -(ray_x_norm * z_forward)

    cos_th = math.cos(robot_theta)
    sin_th = math.sin(robot_theta)

    point_x = robot_x + rob_x * cos_th - rob_y * sin_th
    point_y = robot_y + rob_x * sin_th + rob_y * cos_th

    gx = (point_x / RESOLUTION + MAP_SIZE // 2).astype(np.int32)
    gy = (point_y / RESOLUTION + MAP_SIZE // 2).astype(np.int32)

    bounds_mask = (gx >= 0) & (gx < MAP_SIZE) & (gy >= 0) & (gy < MAP_SIZE)
    gx = gx[bounds_mask]
    gy = gy[bounds_mask]

    wall_mask = ~locked_walls[gy, gx]
    gx = gx[wall_mask]
    gy = gy[wall_mask]

    green_floor[gy, gx] = True

    # Fill tiny holes so the green strip is not broken.
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            tx = np.clip(gx + ox, 0, MAP_SIZE - 1)
            ty = np.clip(gy + oy, 0, MAP_SIZE - 1)
            green_floor[ty, tx] = True




# Reference for green floor detection code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
# --- FLOATING OBSTACLE DETECTION ---
# Precompute camera sample points so each frame is faster.
FLOAT_SAMPLE_STEP = 4
_px, _py = np.meshgrid(
    np.arange(0, camera_width, FLOAT_SAMPLE_STEP),
    np.arange(0, camera_height, FLOAT_SAMPLE_STEP)
)
_px_flat = _px.flatten()
_py_flat = _py.flatten()
_indices = _py_flat * camera_width + _px_flat

FLOAT_MIN_HEIGHT = 0.02
FLOAT_MAX_HEIGHT = 0.55
FLOAT_MAX_DEPTH = 3.0
# Anything below this height is too low for the robot to pass under.
ROBOT_CLEARANCE_HEIGHT_METERS = globals().get("ROBOT_CLEARANCE_HEIGHT_METERS", 0.19)
BELOW_OFFSET_PIXELS = 32
DEPTH_GAP_THRESHOLD = 0.25
FLOAT_DEBUG = True
FLOAT_RGB_COLOR_TOL = globals().get("FLOAT_RGB_COLOR_TOL", 32.0)
FLOAT_RGB_DEPTH_TOL = globals().get("FLOAT_RGB_DEPTH_TOL", 0.35)
FLOAT_RGB_COMPONENT_PAD_PX = globals().get("FLOAT_RGB_COMPONENT_PAD_PX", 8)
FLOAT_RGB_MIN_FUSED_PIXELS = globals().get("FLOAT_RGB_MIN_FUSED_PIXELS", 12)

# --- FLOATING-WALL LINE COMPLETION ---
# These values help turn floating obstacle dots into a wall line.
FLOAT_LINE_INTERVAL = 30
FLOAT_LINE_CLUSTER_DILATE = 4
FLOAT_LINE_MIN_POINTS = 5
FLOAT_LINE_MIN_LINEARITY = 0.75
FLOAT_LINE_MAX_RAY_CELLS = 24
FLOAT_LINE_MIN_SPAN_CELLS = 6
FLOAT_LINE_MIN_LONG_SIDE_CELLS = 16
FLOAT_LINE_MIN_LONG_SIDE_RATIO = 2.5
FLOAT_LINE_OUTLIER_TRIM_PCT = 5.0
FLOAT_LINE_CARDINAL_SNAP_DEG = 18.0
FLOAT_LINE_THICKNESS = 1
FLOAT_BACK_PADDING_CELLS = globals().get("FLOAT_BACK_PADDING_CELLS", 12)
FLOAT_BACK_PADDING_BRUSH_CELLS = globals().get("FLOAT_BACK_PADDING_BRUSH_CELLS", 1)
FLOAT_BACK_PADDING_END_TRIM_CELLS = globals().get("FLOAT_BACK_PADDING_END_TRIM_CELLS", 4)
FLOAT_PILLAR_PADDING_CLEAR_RADIUS_METERS = globals().get(
    "FLOAT_PILLAR_PADDING_CLEAR_RADIUS_METERS", 0.45
)
FLOAT_SETTLE_RADIUS = 6
FLOAT_DOT_TTL = 45

FLOAT_LINE_MIN_TOTAL_CELLS = 13
FLOAT_LINE_REQUIRE_ANCHOR = False


LOW_BUMP_INTERVAL = globals().get("LOW_BUMP_INTERVAL", 5)
LOW_BUMP_SAMPLE_STEP = globals().get("LOW_BUMP_SAMPLE_STEP", 4)
LOW_BUMP_MIN_HEIGHT = globals().get("LOW_BUMP_MIN_HEIGHT", 0.03)
LOW_BUMP_MAX_HEIGHT = globals().get("LOW_BUMP_MAX_HEIGHT", 0.10)
LOW_BUMP_MIN_DEPTH = globals().get("LOW_BUMP_MIN_DEPTH", 0.10)
LOW_BUMP_MAX_DEPTH = globals().get("LOW_BUMP_MAX_DEPTH", 2.5)
LOW_BUMP_MIN_COMPONENT_AREA = globals().get("LOW_BUMP_MIN_COMPONENT_AREA", 2)
LOW_BUMP_MIN_IMAGE_COMPONENT_PIXELS = globals().get("LOW_BUMP_MIN_IMAGE_COMPONENT_PIXELS", 10)
LOW_BUMP_CONFIRM_FRAMES = globals().get("LOW_BUMP_CONFIRM_FRAMES", 2)

_bump_px, _bump_py = np.meshgrid(
    np.arange(0, camera_width, LOW_BUMP_SAMPLE_STEP),
    np.arange(0, camera_height, LOW_BUMP_SAMPLE_STEP)
)
_bump_px_flat = _bump_px.flatten()
_bump_py_flat = _bump_py.flatten()
_bump_indices = _bump_py_flat * camera_width + _bump_px_flat


def fuse_floating_rgb_depth(seed_x, seed_y, seed_z, depth_img, bgr_img, cy, fy):
    """
    Grow sparse floating-wall depth seeds into denser RGB-consistent image patches.
    Depth still provides geometry; RGB mainly recovers pixels skipped by sparse sampling.
    """
    if seed_x.size == 0:
        return seed_x, seed_y, seed_z

    # Grow the detected depth points using similar colour and depth.
    seed_mask = np.zeros((camera_height, camera_width), dtype=np.uint8)
    seed_mask[seed_y, seed_x] = 1
    seed_kernel_size = max(3, FLOAT_SAMPLE_STEP + 1)
    seed_kernel = np.ones((seed_kernel_size, seed_kernel_size), dtype=np.uint8)
    seed_regions = cv2.dilate(seed_mask, seed_kernel, iterations=1)
    num_seed_labels, seed_labels, stats, _ = cv2.connectedComponentsWithStats(seed_regions, 8)

    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    rows = np.arange(camera_height, dtype=np.float32)[:, None]
    with np.errstate(invalid='ignore'):
        point_height_img = CAMERA_HEIGHT_METERS - (((rows - cy) / fy) * depth_img)
    valid_dense_depth = (
        np.isfinite(depth_img)
        & (depth_img > 0.10)
        & (depth_img < FLOAT_MAX_DEPTH)
        & (point_height_img > FLOAT_MIN_HEIGHT)
        & (point_height_img < FLOAT_MAX_HEIGHT)
    )

    fused_mask = seed_mask.astype(bool)

    for label_id in range(1, num_seed_labels):
        seed_region = (seed_labels == label_id) & seed_mask.astype(bool)
        local_seed_y, local_seed_x = np.where(seed_region)
        if local_seed_x.size == 0:
            continue

        x0 = max(0, stats[label_id, cv2.CC_STAT_LEFT] - FLOAT_RGB_COMPONENT_PAD_PX)
        y0 = max(0, stats[label_id, cv2.CC_STAT_TOP] - FLOAT_RGB_COMPONENT_PAD_PX)
        x1 = min(
            camera_width,
            stats[label_id, cv2.CC_STAT_LEFT]
            + stats[label_id, cv2.CC_STAT_WIDTH]
            + FLOAT_RGB_COMPONENT_PAD_PX,
        )
        y1 = min(
            camera_height,
            stats[label_id, cv2.CC_STAT_TOP]
            + stats[label_id, cv2.CC_STAT_HEIGHT]
            + FLOAT_RGB_COMPONENT_PAD_PX,
        )

        seed_lab = np.median(lab_img[local_seed_y, local_seed_x], axis=0)
        seed_depth = float(np.median(depth_img[local_seed_y, local_seed_x]))

        lab_roi = lab_img[y0:y1, x0:x1]
        depth_roi = depth_img[y0:y1, x0:x1]
        valid_roi = valid_dense_depth[y0:y1, x0:x1]
        color_dist = np.linalg.norm(lab_roi - seed_lab, axis=2)
        candidate_roi = (
            valid_roi
            & (color_dist <= FLOAT_RGB_COLOR_TOL)
            & (np.abs(depth_roi - seed_depth) <= FLOAT_RGB_DEPTH_TOL)
        )

        num_labels, labels = cv2.connectedComponents(candidate_roi.astype(np.uint8), 8)
        if num_labels <= 1:
            continue

        seed_y_roi = local_seed_y - y0
        seed_x_roi = local_seed_x - x0
        in_roi = (
            (seed_y_roi >= 0) & (seed_y_roi < (y1 - y0))
            & (seed_x_roi >= 0) & (seed_x_roi < (x1 - x0))
        )
        touched_labels = np.unique(labels[seed_y_roi[in_roi], seed_x_roi[in_roi]])
        touched_labels = touched_labels[touched_labels > 0]
        if touched_labels.size == 0:
            continue

        selected = np.isin(labels, touched_labels)
        if np.count_nonzero(selected) >= FLOAT_RGB_MIN_FUSED_PIXELS:
            fused_mask[y0:y1, x0:x1] |= selected

    fused_y, fused_x = np.where(fused_mask & valid_dense_depth)
    if fused_x.size < seed_x.size:
        return seed_x, seed_y, seed_z
    return fused_x.astype(np.int32), fused_y.astype(np.int32), depth_img[fused_y, fused_x]



# Reference for RGB/depth fusion code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
def detect_low_bumps(robot_x, robot_y, robot_theta):
    # Detect small bumps that lidar may miss.
    global grid, locked_walls, lidar_only_walls, low_bump_votes

    if not camera_depth:
        return

    depth_array = camera_depth.getRangeImage()
    if not depth_array:
        return

    cx = camera_width / 2.0
    cy = camera_height / 2.0
    fx = cx / math.tan(camera_fov_horizontal / 2.0)
    fy = cy / math.tan(camera_fov_vertical / 2.0)

    z_forward = np.array([depth_array[i] for i in _bump_indices], dtype=np.float32)
    # Use lower image pixels because bumps are near the floor.
    valid_depth = (
        np.isfinite(z_forward)
        & (z_forward > LOW_BUMP_MIN_DEPTH)
        & (z_forward < LOW_BUMP_MAX_DEPTH)
        & (_bump_py_flat > cy)
    )

    if np.count_nonzero(valid_depth) == 0:
        return

    px_valid = _bump_px_flat[valid_depth]
    py_valid = _bump_py_flat[valid_depth]
    z_valid = z_forward[valid_depth]

    ray_y_norm = (py_valid - cy) / fy
    y_down = ray_y_norm * z_valid
    point_height = CAMERA_HEIGHT_METERS - y_down

    # Keep only heights that look like a small bump.
    bump_height = (
        (point_height > LOW_BUMP_MIN_HEIGHT)
        & (point_height < LOW_BUMP_MAX_HEIGHT)
    )

    if np.count_nonzero(bump_height) == 0:
        return

    bump_mask_img = np.zeros(_bump_px.shape, dtype=np.uint8)
    bump_rows = (py_valid[bump_height] // LOW_BUMP_SAMPLE_STEP).astype(np.int32)
    bump_cols = (px_valid[bump_height] // LOW_BUMP_SAMPLE_STEP).astype(np.int32)
    bump_mask_img[bump_rows, bump_cols] = 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bump_mask_img, 8)
    coherent_mask_img = np.zeros_like(bump_mask_img)
    # Remove tiny random dots.
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= LOW_BUMP_MIN_IMAGE_COMPONENT_PIXELS:
            coherent_mask_img[labels == label_id] = 1

    if not np.any(coherent_mask_img):
        return

    coherent_sample_mask = coherent_mask_img[
        (py_valid // LOW_BUMP_SAMPLE_STEP).astype(np.int32),
        (px_valid // LOW_BUMP_SAMPLE_STEP).astype(np.int32)
    ].astype(bool)
    bump_height &= coherent_sample_mask

    if np.count_nonzero(bump_height) == 0:
        return

    px_bump = px_valid[bump_height]
    z_bump = z_valid[bump_height]

    ray_x_norm = (px_bump - cx) / fx
    x_right = ray_x_norm * z_bump

    rob_x = z_bump
    rob_y = -x_right

    # Convert robot-local points into map points.
    cos_th = math.cos(robot_theta)
    sin_th = math.sin(robot_theta)

    point_x = robot_x + rob_x * cos_th - rob_y * sin_th
    point_y = robot_y + rob_x * sin_th + rob_y * cos_th

    gx = (point_x / RESOLUTION + MAP_SIZE // 2).astype(np.int32)
    gy = (point_y / RESOLUTION + MAP_SIZE // 2).astype(np.int32)

    bounds_mask = (gx >= 0) & (gx < MAP_SIZE) & (gy >= 0) & (gy < MAP_SIZE)
    gx = gx[bounds_mask]
    gy = gy[bounds_mask]

    if gx.size == 0:
        return

    candidate_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    candidate_mask[gy, gx] = 1

    if LOW_BUMP_MIN_COMPONENT_AREA > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
        cleaned_mask = np.zeros_like(candidate_mask)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= LOW_BUMP_MIN_COMPONENT_AREA:
                cleaned_mask[labels == label_id] = 1
        map_y, map_x = np.where(cleaned_mask > 0)
    else:
        map_y, map_x = np.where(candidate_mask > 0)

    if map_x.size == 0:
        return

    current_bump_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
    current_bump_mask[map_y, map_x] = True
    # Use votes so one bad frame does not create a wall.
    stale_votes = (low_bump_votes > 0) & (~current_bump_mask)
    low_bump_votes[stale_votes] -= 1

    low_bump_votes[map_y, map_x] = np.minimum(
        low_bump_votes[map_y, map_x] + 1,
        LOW_BUMP_CONFIRM_FRAMES
    )

    confirmed = low_bump_votes[map_y, map_x] >= LOW_BUMP_CONFIRM_FRAMES
    if np.count_nonzero(confirmed) == 0:
        return

    map_x = map_x[confirmed]
    map_y = map_y[confirmed]

    # Confirmed bumps are added as walls.
    grid[map_y, map_x] = L_MAX
    locked_walls[map_y, map_x] = True
    lidar_only_walls[map_y, map_x] = True


# Reference for low bump detection code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
def detect_floating_obstacles(robot_x, robot_y, robot_theta):
    # Detect obstacles that are above the floor but still block the robot.
    global grid, locked_walls, floating_obstacles, floating_obstacle_memory
    global floating_detection_dots, floating_detection_age

    if step_count % 4 != 0:
        return

    # Keep recent dots for a short time so they can connect into lines.
    live_dots = floating_detection_age > 0
    floating_detection_age[live_dots] -= 1
    floating_detection_dots[floating_detection_age == 0] = False
    floating_obstacles[:, :] = False

    if not camera_depth:
        return

    depth_array = camera_depth.getRangeImage()
    if not depth_array:
        return

    cx = camera_width / 2.0
    cy = camera_height / 2.0
    fx = cx / math.tan(camera_fov_horizontal / 2.0)
    fy = cy / math.tan(camera_fov_vertical / 2.0)

    z_forward = np.array([depth_array[i] for i in _indices], dtype=np.float32)
    valid_mask = np.isfinite(z_forward) & (z_forward > 0.10) & (z_forward < FLOAT_MAX_DEPTH)

    px_valid = _px_flat[valid_mask]
    py_valid = _py_flat[valid_mask]
    z_forward = z_forward[valid_mask]

    if z_forward.size == 0:
        return

    ray_y_norm = (py_valid - cy) / fy
    y_down = ray_y_norm * z_forward
    point_height = CAMERA_HEIGHT_METERS - y_down

    # Keep only points in the height range we care about.
    height_mask = (point_height > FLOAT_MIN_HEIGHT) & (point_height < FLOAT_MAX_HEIGHT)
    if np.count_nonzero(height_mask) == 0:
        return

    px_h = px_valid[height_mask]
    py_h = py_valid[height_mask]
    z_h = z_forward[height_mask]

    py_below = np.clip(py_h + BELOW_OFFSET_PIXELS, 0, camera_height - 1)
    below_indices = py_below * camera_width + px_h
    z_below = np.array([depth_array[i] for i in below_indices], dtype=np.float32)

    valid_below = np.isfinite(z_below) & (z_below > 0.10) & (z_below < 4.0)
    # If the point below is farther away, the object is floating.
    free_under = valid_below & (z_below > z_h + DEPTH_GAP_THRESHOLD)

    px_obs = px_h[free_under]
    z_obs = z_h[free_under]
    obs_height = point_height[height_mask][free_under]

    if z_obs.size == 0:
        return

    ray_x_norm = (px_obs - cx) / fx
    x_right = ray_x_norm * z_obs

    rob_x = z_obs
    rob_y = -x_right

    cos_th = math.cos(robot_theta)
    sin_th = math.sin(robot_theta)

    point_x = robot_x + rob_x * cos_th - rob_y * sin_th
    point_y = robot_y + rob_x * sin_th + rob_y * cos_th

    gx = (point_x / RESOLUTION + MAP_SIZE // 2).astype(np.int32)
    gy = (point_y / RESOLUTION + MAP_SIZE // 2).astype(np.int32)

    bounds_mask = (gx >= 0) & (gx < MAP_SIZE) & (gy >= 0) & (gy < MAP_SIZE)
    gx = gx[bounds_mask]
    gy = gy[bounds_mask]
    obs_height = obs_height[bounds_mask]

    if gx.size == 0:
        return

    in_settled = floating_settled_zone[gy, gx]
    gx = gx[~in_settled]
    gy = gy[~in_settled]
    obs_height = obs_height[~in_settled]
    if gx.size == 0:
        return

    candidate_mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    candidate_mask[gy, gx] = 1
    candidate_min_height = np.full((MAP_SIZE, MAP_SIZE), np.inf, dtype=np.float32)
    np.minimum.at(candidate_min_height, (gy, gx), obs_height.astype(np.float32))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
    cleaned_mask = np.zeros_like(candidate_mask)
    MIN_FLOATING_COMPONENT_AREA = 3
    # Keep only blobs that are low enough to hit the robot.
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < MIN_FLOATING_COMPONENT_AREA:
            continue

        component = labels == label_id
        lowest_seen_height = np.min(candidate_min_height[component])
        if lowest_seen_height > ROBOT_CLEARANCE_HEIGHT_METERS:
            continue

        cleaned_mask[component] = 1

    map_y, map_x = np.where(cleaned_mask > 0)
    if map_x.size == 0:
        return

    # Save it in the current map and also in memory.
    floating_obstacles[map_y, map_x] = True
    floating_detection_dots[map_y, map_x] = True
    floating_detection_age[map_y, map_x] = FLOAT_DOT_TTL
    floating_obstacle_memory[map_y, map_x] = True
    grid[map_y, map_x] = L_MAX
    locked_walls[map_y, map_x] = True


# Reference for floating obstacle code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
def _snap_to_cardinal(direction, tolerance_deg):
    # Snap almost-straight wall lines to horizontal or vertical.
    angle = math.atan2(direction[1], direction[0]) % math.pi
    tol = math.radians(tolerance_deg)
    if angle < tol or angle > math.pi - tol:
        return np.array([1.0, 0.0])
    if abs(angle - math.pi / 2.0) < tol:
        return np.array([0.0, 1.0])
    return direction

def _mark_cell_thick(ix, iy, thickness, line_track_mask=None):
    # Mark a cell and nearby cells as blocked.
    global grid, locked_walls, floating_obstacle_memory
    x0 = max(0, ix - thickness)
    x1 = min(MAP_SIZE, ix + thickness + 1)
    y0 = max(0, iy - thickness)
    y1 = min(MAP_SIZE, iy + thickness + 1)
    grid[y0:y1, x0:x1] = L_MAX
    locked_walls[y0:y1, x0:x1] = True
    floating_obstacle_memory[y0:y1, x0:x1] = True
    if line_track_mask is not None:
        line_track_mask[y0:y1, x0:x1] = True

def _mark_floating_back_padding_cell(ix, iy, brush_cells):
    # Add extra safety padding behind a floating wall.
    global floating_back_padding
    if not (0 <= ix < MAP_SIZE and 0 <= iy < MAP_SIZE):
        return
    x0 = max(0, ix - brush_cells)
    x1 = min(MAP_SIZE, ix + brush_cells + 1)
    y0 = max(0, iy - brush_cells)
    y1 = min(MAP_SIZE, iy + brush_cells + 1)
    floating_back_padding[y0:y1, x0:x1] = True

def clear_floating_padding_near_pillars():
    # Remove padding near pillars so the robot can still reach them.
    global floating_back_padding

    clear_radius_cells = int(math.ceil(
        FLOAT_PILLAR_PADDING_CLEAR_RADIUS_METERS / RESOLUTION
    ))
    if clear_radius_cells <= 0:
        return

    for pillar in mission_pillars.values():
        if pillar is None:
            continue

        center_x, center_y = world_to_grid(pillar["x"], pillar["y"])
        x0 = max(0, center_x - clear_radius_cells)
        x1 = min(MAP_SIZE, center_x + clear_radius_cells + 1)
        y0 = max(0, center_y - clear_radius_cells)
        y1 = min(MAP_SIZE, center_y + clear_radius_cells + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        clear_mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2
                      <= clear_radius_cells ** 2)
        floating_back_padding[y0:y1, x0:x1][clear_mask] = False

def _unique_ordered_cells(cells):
    # Remove duplicate cells but keep the same order.
    ordered = []
    seen = set()
    for ix, iy in cells:
        key = (int(ix), int(iy))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered

def _add_floating_back_padding(line_cells, direction):
    """
    Add thick clearance only behind a floating wall, opposite the robot side.
    The first/last cells are trimmed so the padding does not cap the ends.
    """
    global floating_back_padding

    line_cells = _unique_ordered_cells(line_cells)
    if not line_cells:
        return

    # Put padding on the side away from the robot.
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return
    normal /= norm

    robot_col, robot_row = world_to_grid(x, y)
    midpoint = np.mean(np.asarray(line_cells, dtype=np.float64), axis=0)
    robot_vec = np.array([robot_col - midpoint[0], robot_row - midpoint[1]], dtype=np.float64)
    if float(np.linalg.norm(robot_vec)) > 1e-9 and float(np.dot(normal, robot_vec)) >= 0.0:
        normal = -normal

    trim = min(FLOAT_BACK_PADDING_END_TRIM_CELLS, max(0, (len(line_cells) - 1) // 3))
    usable_cells = line_cells[trim:len(line_cells) - trim] if trim > 0 else line_cells
    if not usable_cells:
        usable_cells = line_cells

    first_offset = FLOAT_LINE_THICKNESS + 1
    last_offset = FLOAT_LINE_THICKNESS + FLOAT_BACK_PADDING_CELLS
    # Start padding just behind the wall line.
    for cx_pt, cy_pt in usable_cells:
        for offset in range(first_offset, last_offset + 1):
            px = int(round(cx_pt + normal[0] * offset))
            py = int(round(cy_pt + normal[1] * offset))
            _mark_floating_back_padding_cell(px, py, FLOAT_BACK_PADDING_BRUSH_CELLS)

    floating_back_padding[floating_obstacle_memory] = False
    clear_floating_padding_near_pillars()

def _scout_extension(start_x, start_y, dx, dy, max_steps, walls_snapshot):
    """
    Walk from (start_x,start_y) in direction (dx,dy).
    Return (cells_path, found_wall) WITHOUT writing anything.
    cells_path stops just before the wall hit (or at max_steps).
    """
    norm = max(abs(dx), abs(dy))
    if norm < 1e-9:
        return [], False
    sx = dx / norm
    sy = dy / norm

    x_pos, y_pos = float(start_x), float(start_y)
    path = []
    found_wall = False
    for _ in range(max_steps):
        x_pos += sx
        y_pos += sy
        ix, iy = int(round(x_pos)), int(round(y_pos))
        if not (0 <= ix < MAP_SIZE and 0 <= iy < MAP_SIZE):
            break
        if walls_snapshot[iy, ix]:
            found_wall = True
            break
        path.append((ix, iy))

    return path, found_wall

def connect_floating_to_walls():
    """
    Scout-then-commit pattern:
      1. Build cluster, find principal axis & extremes.
      2. Scout the within-cluster fill and both extensions WITHOUT drawing.
      3. Decide: total drawn length must be >= MIN_TOTAL_CELLS,
                 and (if REQUIRE_ANCHOR) at least one extension must hit a wall.
      4. Only then commit + settle. Otherwise, leave the dots for next time.
    """
    global grid, locked_walls, floating_obstacle_memory, floating_back_padding
    global floating_detection_dots, floating_detection_age, floating_settled_zone

    active_dots = floating_detection_dots & ~floating_settled_zone
    if not np.any(active_dots):
        return

    real_walls_mask = lidar_only_walls

    # Dilate dots so close detections become one object.
    mask = active_dots.astype(np.uint8)
    k = FLOAT_LINE_CLUSTER_DILATE
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    num_labels, labels = cv2.connectedComponents(dilated)

    new_cells_total = 0
    rejected_short = 0
    rejected_short_side = 0
    rejected_no_anchor = 0

    for label_id in range(1, num_labels):
        cluster = (labels == label_id) & mask.astype(bool)
        ys, xs = np.where(cluster)
        if xs.size < FLOAT_LINE_MIN_POINTS:
            continue

        # Find the main direction of the dot cluster.
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        total = eigvals[0] + eigvals[1]
        if total < 1e-6:
            continue
        linearity = eigvals[1] / total
        if linearity < FLOAT_LINE_MIN_LINEARITY:
            continue

        direction = eigvecs[:, -1]
        direction = _snap_to_cardinal(direction, FLOAT_LINE_CARDINAL_SNAP_DEG)

        proj = centered @ direction
        # Remove extreme noisy points.
        if FLOAT_LINE_OUTLIER_TRIM_PCT > 0 and pts.shape[0] >= 10:
            lo = np.percentile(proj, FLOAT_LINE_OUTLIER_TRIM_PCT)
            hi = np.percentile(proj, 100.0 - FLOAT_LINE_OUTLIER_TRIM_PCT)
            keep = (proj >= lo) & (proj <= hi)
            if int(np.count_nonzero(keep)) >= FLOAT_LINE_MIN_POINTS:
                pts_t = pts[keep]
                proj_t = proj[keep]
            else:
                pts_t = pts
                proj_t = proj
        else:
            pts_t = pts
            proj_t = proj

        far_pos_idx = int(np.argmax(proj_t))
        far_neg_idx = int(np.argmin(proj_t))
        far_pos = (int(pts_t[far_pos_idx, 0]), int(pts_t[far_pos_idx, 1]))
        far_neg = (int(pts_t[far_neg_idx, 0]), int(pts_t[far_neg_idx, 1]))

        # Ignore clusters that are too short.
        span = math.hypot(far_pos[0] - far_neg[0], far_pos[1] - far_neg[1])
        if span < FLOAT_LINE_MIN_SPAN_CELLS:
            continue

        # Only extend the long side of the wall.
        minor_direction = np.array([-direction[1], direction[0]])
        centered_t = pts_t - pts_t.mean(axis=0)
        major_proj = centered_t @ direction
        minor_proj = centered_t @ minor_direction
        major_span = float(np.max(major_proj) - np.min(major_proj))
        minor_span = max(1.0, float(np.max(minor_proj) - np.min(minor_proj)))
        if (major_span < FLOAT_LINE_MIN_LONG_SIDE_CELLS
                or major_span < minor_span * FLOAT_LINE_MIN_LONG_SIDE_RATIO):
            rejected_short_side += 1
            continue


        # Check the line first without writing to the map.
        # Get the line inside the detected cluster.
        rx, ry = bresenham_cells(far_neg[0], far_neg[1], far_pos[0], far_pos[1])
        valid = (rx >= 0) & (rx < MAP_SIZE) & (ry >= 0) & (ry < MAP_SIZE)
        rx = rx[valid]
        ry = ry[valid]
        within_cluster_cells = []
        if rx.size > 0:
            wall_hits = real_walls_mask[ry, rx]
            if np.any(wall_hits):
                first_hit = int(np.argmax(wall_hits))
                rx = rx[:first_hit]
                ry = ry[:first_hit]
            within_cluster_cells = list(zip(rx.tolist(), ry.tolist()))

        # Check both ends of the line.
        ext_pos_path, anchored_pos = _scout_extension(
            far_pos[0], far_pos[1],
            direction[0], direction[1],
            FLOAT_LINE_MAX_RAY_CELLS, real_walls_mask
        )
        ext_neg_path, anchored_neg = _scout_extension(
            far_neg[0], far_neg[1],
            -direction[0], -direction[1],
            FLOAT_LINE_MAX_RAY_CELLS, real_walls_mask
        )


        # Decide if this line is good enough.
        # Only use extensions that connect to a wall.
        ext_pos_use = ext_pos_path if anchored_pos else []
        ext_neg_use = ext_neg_path if anchored_neg else []

        total_cells = len(within_cluster_cells) + len(ext_pos_use) + len(ext_neg_use)
        any_anchor = anchored_pos or anchored_neg

        if FLOAT_LINE_REQUIRE_ANCHOR and not any_anchor:
            rejected_no_anchor += 1
            continue

        if total_cells < FLOAT_LINE_MIN_TOTAL_CELLS:
            rejected_short += 1
            continue


        # Now write the accepted line to the map.
        line_track = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)

        # Bad clusters were skipped before this point.
        for cx_pt, cy_pt in within_cluster_cells:
            _mark_cell_thick(int(cx_pt), int(cy_pt), FLOAT_LINE_THICKNESS, line_track)
        for cx_pt, cy_pt in ext_pos_use:
            _mark_cell_thick(int(cx_pt), int(cy_pt), FLOAT_LINE_THICKNESS, line_track)
        for cx_pt, cy_pt in ext_neg_use:
            _mark_cell_thick(int(cx_pt), int(cy_pt), FLOAT_LINE_THICKNESS, line_track)

        ordered_line_cells = (
            list(reversed(ext_neg_use))
            + within_cluster_cells
            + ext_pos_use
        )
        _add_floating_back_padding(ordered_line_cells, direction)

        new_cells_total += total_cells

        # Mark this area as done so it is not processed again.
        if np.any(line_track):
            r = FLOAT_SETTLE_RADIUS
            dil_kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
            settled = cv2.dilate(line_track.astype(np.uint8), dil_kernel, iterations=1).astype(bool)
            floating_settled_zone |= settled
            floating_detection_dots[settled] = False
            floating_detection_age[settled] = 0

    if FLOAT_DEBUG and (new_cells_total > 0 or rejected_short > 0
                        or rejected_short_side > 0 or rejected_no_anchor > 0):
        print("FLOAT LINE DEBUG:",
              "new_cells=", new_cells_total,
              "rejected_short=", rejected_short,
              "rejected_short_side=", rejected_short_side,
              "rejected_no_anchor=", rejected_no_anchor)

    if new_cells_total > 0:
        if 'brain' in globals():
            # Clear old frontier choices because the map changed.
            brain.blacklisted_frontiers.clear()
            brain.permanently_unreachable.clear()
            if hasattr(brain, '_frontier_retry_counts'):
                brain._frontier_retry_counts.clear()
            print("[Navigator] Map topology changed — cleared frontier blacklist cache.")


# Reference for floating wall line completion and padding code above: OpenAI ChatGPT/Codex helped implement our own ideas. All ideas, testing, performance tuning, and final decisions are ours.
def draw_map_pygame():
    # Draw the live map window.
    # Draw the base map first.
    normalized = np.clip((grid - L_MIN) / (L_MAX - L_MIN), 0.0, 1.0)
    gray_vals = (255 - (normalized * 255)).astype(np.uint8)
    display_pixels = np.stack((gray_vals,) * 3, axis=-1)

    visible_float_padding = (
        floating_back_padding
        & (~floating_obstacle_memory)
        & (~floating_obstacles)
        & (~green_floor)
    )
    # Use different colours for each map layer.
    display_pixels[visible_float_padding] = [255, 0, 0]
    display_pixels[green_floor] = [0, 255, 0]
    display_pixels[floating_obstacle_memory] = [255, 0, 255]
    display_pixels[floating_obstacles] = [255, 0, 255]

    # Show the inflated safety area from the navigator.
    if 'brain' in globals() and getattr(brain, 'last_inflated', None) is not None:
        overlay_color = np.array([57, 255, 20])  # Bright colour for safety area.
        opacity = 0.40  # Keep it see-through.

        original_pixels = display_pixels[brain.last_inflated]
        blended = (original_pixels * (1.0 - opacity)) + (overlay_color * opacity)
        display_pixels[brain.last_inflated] = blended.astype(np.uint8)

    display_pixels[visible_float_padding] = [255, 0, 0]
    display_pixels[green_floor] = [0, 255, 0]
    display_pixels[floating_obstacle_memory] = [255, 0, 255]
    display_pixels[floating_obstacles] = [255, 0, 255]

    surf = pygame.surfarray.make_surface(display_pixels.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surf, (SCREEN_SIZE, SCREEN_SIZE))
    scaled = pygame.transform.flip(scaled, False, True)
    screen.blit(scaled, (0, 0))

    # Draw the robot.
    gx, gy = world_to_grid(x, y)
    if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
        px = int(gx * (SCREEN_SIZE / MAP_SIZE))
        py = SCREEN_SIZE - int(gy * (SCREEN_SIZE / MAP_SIZE))
        pygame.draw.circle(screen, (0, 255, 0), (px, py), 5)
        nose_len = 12
        nx = int(px + nose_len * math.cos(theta))
        ny = int(py - nose_len * math.sin(theta))
        pygame.draw.line(screen, (255, 255, 255), (px, py), (nx, ny), 2)

    # Draw the mission pillars.
    for color, data in mission_pillars.items():
        if data:
            pgx, pgy = world_to_grid(data['x'], data['y'])
            if 0 <= pgx < MAP_SIZE and 0 <= pgy < MAP_SIZE:
                ppx = int(pgx * (SCREEN_SIZE / MAP_SIZE))
                ppy = SCREEN_SIZE - int(pgy * (SCREEN_SIZE / MAP_SIZE))
                p_color = (0, 0, 255) if color == 'blue' else (255, 255, 0)
                pygame.draw.circle(screen, p_color, (ppx, ppy), 12)

    # Draw navigator path and target.
    if 'brain' in globals():
        # Draw the planned path.
        if brain.current_path:
            path_pixels = []
            for wx, wy in brain.current_path:
                pgx, pgy = world_to_grid(wx, wy)
                if 0 <= pgx < MAP_SIZE and 0 <= pgy < MAP_SIZE:
                    ppx = int(pgx * (SCREEN_SIZE / MAP_SIZE))
                    ppy = SCREEN_SIZE - int(pgy * (SCREEN_SIZE / MAP_SIZE))
                    path_pixels.append((ppx, ppy))
            
            if len(path_pixels) > 1:
                pygame.draw.lines(screen, (0, 255, 255), False, path_pixels, 3)

        # Draw a direct target line if one exists.
        direct_target = getattr(brain, 'direct_drive_target_world', None)
        if direct_target:
            start_gx, start_gy = world_to_grid(x, y)
            end_gx, end_gy = world_to_grid(direct_target[0], direct_target[1])
            if (0 <= start_gx < MAP_SIZE and 0 <= start_gy < MAP_SIZE
                    and 0 <= end_gx < MAP_SIZE and 0 <= end_gy < MAP_SIZE):
                start_px = int(start_gx * (SCREEN_SIZE / MAP_SIZE))
                start_py = SCREEN_SIZE - int(start_gy * (SCREEN_SIZE / MAP_SIZE))
                end_px = int(end_gx * (SCREEN_SIZE / MAP_SIZE))
                end_py = SCREEN_SIZE - int(end_gy * (SCREEN_SIZE / MAP_SIZE))
                pygame.draw.line(screen, (0, 255, 255), (start_px, start_py), (end_px, end_py), 3)

        # Draw the current frontier target.
        if brain.target_frontier_world:
            tx, ty = brain.target_frontier_world
            pgx, pgy = world_to_grid(tx, ty)
            if 0 <= pgx < MAP_SIZE and 0 <= pgy < MAP_SIZE:
                ppx = int(pgx * (SCREEN_SIZE / MAP_SIZE))
                ppy = SCREEN_SIZE - int(pgy * (SCREEN_SIZE / MAP_SIZE))
                pygame.draw.circle(screen, (255, 255, 0), (ppx, ppy), 7)
                pygame.draw.circle(screen, (0, 0, 0), (ppx, ppy), 9, 2)

        # Draw the robot safety box.
        corners_local = [
            (brain.ENVELOPE_FRONT, brain.ENVELOPE_SIDE),
            (brain.ENVELOPE_FRONT, -brain.ENVELOPE_SIDE),
            (brain.ENVELOPE_REAR, -brain.ENVELOPE_SIDE),
            (brain.ENVELOPE_REAR, brain.ENVELOPE_SIDE)
        ]
        
        polygon_pixels = []
        for lx, ly in corners_local:
            wx = x + (lx * math.cos(theta) - ly * math.sin(theta))
            wy = y + (lx * math.sin(theta) + ly * math.cos(theta))
            
            pgx, pgy = world_to_grid(wx, wy)
            if 0 <= pgx < MAP_SIZE and 0 <= pgy < MAP_SIZE:
                ppx = int(pgx * (SCREEN_SIZE / MAP_SIZE))
                ppy = SCREEN_SIZE - int(pgy * (SCREEN_SIZE / MAP_SIZE))
                polygon_pixels.append((ppx, ppy))
        
        if len(polygon_pixels) == 4:
            pygame.draw.polygon(screen, (255, 0, 0), polygon_pixels, 2)
            
        rgx, rgy = world_to_grid(x, y)
        if 0 <= rgx < MAP_SIZE and 0 <= rgy < MAP_SIZE:
            rpx = int(rgx * (SCREEN_SIZE / MAP_SIZE))
            rpy = SCREEN_SIZE - int(rgy * (SCREEN_SIZE / MAP_SIZE))
            
            chamfer_radius_pixels = int((0.21 / RESOLUTION) * (SCREEN_SIZE / MAP_SIZE))
            pygame.draw.circle(screen, (255, 165, 0), (rpx, rpy), chamfer_radius_pixels, 2)

    pygame.display.flip()

# --- MANUAL CONTROL ---
def manual_control():
    # Simple WASD control for testing.
    speed = 8.0
    key = keyboard.getKey()

    if key == ord('W'):
        fl.setVelocity(speed)
        rl.setVelocity(speed)
        fr.setVelocity(speed)
        rr.setVelocity(speed)

    elif key == ord('S'):
        fl.setVelocity(-speed)
        rl.setVelocity(-speed)
        fr.setVelocity(-speed)
        rr.setVelocity(-speed)

    elif key == ord('A'):
        fl.setVelocity(-speed)
        rl.setVelocity(-speed)
        fr.setVelocity(speed)
        rr.setVelocity(speed)

    elif key == ord('D'):
        fl.setVelocity(speed)
        rl.setVelocity(speed)
        fr.setVelocity(-speed)
        rr.setVelocity(-speed)

    else:
        fl.setVelocity(0)
        rl.setVelocity(0)
        fr.setVelocity(0)
        rr.setVelocity(0)


# --- MAIN LOOP ---
# Main loop values.
step_count = 0
last_draw_time = -1e9
last_map_theta = theta

while robot.step(timestep) != -1:

    # Allow the pygame window to close.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # Update robot position.
    update_odometry()
    detect_tilted_contact_obstacle(x, y, theta)

    # Only update mapping when sensors are ready and the robot is not turning hard.
    heading_change = abs(normalize_angle(theta - last_map_theta))
    is_rotating = heading_change > ROTATION_GATING_THRESHOLD
    
    if step_count >= SENSOR_WARMUP_STEPS and not is_rotating:
        # Update all map layers before the navigator runs.
        update_map()
        detect_and_mark_pillars(x, y, theta)
        detect_green_floor(x, y, theta)
        if step_count % LOW_BUMP_INTERVAL == 0:
            detect_low_bumps(x, y, theta)
        detect_floating_obstacles(x, y, theta)
        if step_count % FLOAT_LINE_INTERVAL == 0:
            connect_floating_to_walls()
        clear_floating_padding_near_pillars()
    last_map_theta = theta

    # Run autonomous navigation.
    if step_count >= SENSOR_WARMUP_STEPS:
        current_time = robot.getTime()
        lidar_ranges = np.array(lidar.getRangeImage())
        lidar_fov = lidar.getFov()
        # Send the map and extra blocked areas to the navigator.
        brain.step_brain(
            current_time,
            grid,
            x,
            y,
            theta,
            lidar_ranges,
            lidar_fov,
            mission_pillars,
            green_floor=green_floor,
            extra_blocked=(floating_back_padding | floating_obstacle_memory),
        )

    # Draw the map at a fixed rate.
    now = robot.getTime()
    if now - last_draw_time >= PYGAME_MIN_FRAME_DT:
        draw_map_pygame()
        last_draw_time = now

    step_count += 1

pygame.quit()
