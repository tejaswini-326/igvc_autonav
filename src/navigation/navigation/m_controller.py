from math import radians
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import numpy as np
from nav_msgs.msg import OccupancyGrid
from math import radians
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from math import pi
import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Setting this to true will generate a new topic:
# intersection_llane_scan_2d_debug - An image showing all polar scans and detected distances
DEBUG = True

# Movement Related
LINEAR_SPEED                                    = 1                # m/s   (forward)
MIN_LINEAR_SPEED_WHILE_TURNING                  = LINEAR_SPEED * 0.2
LEFT_TURN_ANGULAR_SPEED                         = 0.5              # rad/s (+ve = CCW = left)

ANGLE_TOLERANCE                                 = radians(25)        # ± deg window around 90° – θ



MY_HZ = 60

# Exploratory Rotator thing

EXPL_ROT_ANGLE = np.deg2rad(45.0)   # explore ±45°
EXPL_YAW_TOL   = np.deg2rad(2.0)    # stop tolerance while rotating
EXPL_SETTLE_TICKS = 2               # ticks to settle before sampling

DISTANCE_THRESHOLD_TO_TRIGGER_CORNER_AVOIDANCE_MECHANISM = 35
AFTER_CENTERING_CORNER_AVOIDANCE_THRESHOLD = 70  # px or convert from meters if you prefer
YAW_ALIGN_TOL = np.deg2rad(3.0)

DISTANCE_TO_MOVE_BACK_ON_CORNER_DETECTED = 5   # meters
BACKUP_SPEED = 0.3                               # m/s (magnitude; command will be negative)
BACKUP_TIMEOUT_S = 2.5                           # safety cap


SCOOT_DISTANCE_M = 0.8        # how far to sidestep
SCOOT_SPEED = 0.35            # m/s
SCOOT_TIMEOUT_S = 3.0         # safety cap


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ──────────────────────────────────────────────────────────────────────────────


def normalise_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi

def _binary_from_costmap(costmap):
    res = costmap.info.resolution
    W   = costmap.info.width
    H   = costmap.info.height
    grid = np.array(costmap.data, dtype=np.int16).reshape(H, W)
    hit_mask = (grid >= 90)
    hit_mask |= (grid < 0)  # unknown as occupied
    binary = (hit_mask.astype(np.uint8) * 255)
    if np.any(binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.dilate(binary, None, iterations=1)
    CX = W // 2
    CY = H // 2
    return binary, CX, CY, W, H, res

def _ray_length_px(binary, CX, CY, theta, W, H):
    # march a single ray from (CX,CY) along theta, using the same image convention as your scans
    MAX_RANGE_PX = int(max(W, H))
    ux = math.cos(theta)
    uy = math.sin(theta)
    for s in range(0, MAX_RANGE_PX + 1):
        u = int(CX + ux * s)
        v = int(CY - uy * s)  # NOTE: same sign convention as radial_scans_from_costmap
        if not (0 <= u < W and 0 <= v < H):
            return s
        if binary[v, u] != 0:
            return s
    return MAX_RANGE_PX


def top_k_endpoint_lookahead_ok(costmap, yaw, k=10, skip_px=3, threshold_px=30):
    """
    Look at the K longest rays from the robot cell; for each, 'jump' skip_px past
    the first hit and keep ray-marching in the same direction. If any of those
    second-leg marches has free length >= threshold_px, return True.
    """
    binary, CX, CY, W, H, _ = _binary_from_costmap(costmap)

    # same angular sweep as radial_scans_from_costmap
    facing = -yaw
    angle_tolerance = np.deg2rad(45)
    nsteps = 2 * (2 * int(np.degrees(angle_tolerance)) + 1)
    angles = np.linspace(facing - angle_tolerance, facing + angle_tolerance, nsteps)

    MAX_RANGE_PX = int(max(W, H))
    dists = np.full(len(angles), MAX_RANGE_PX, dtype=np.int32)
    hits  = np.full(len(angles), False, dtype=bool)

    # Primary march from robot cell
    for i, theta in enumerate(angles):
        ux, uy = math.cos(theta), math.sin(theta)
        for s in range(MAX_RANGE_PX + 1):
            u = int(CX + ux * s)
            v = int(CY - uy * s)
            if not (0 <= u < W and 0 <= v < H):
                dists[i] = s
                break
            if binary[v, u] != 0:
                dists[i] = s
                hits[i] = True
                break

    # Pick top-K longest rays (that actually hit something)
    idx = np.argsort(-dists)  # descending
    picked = [i for i in idx if hits[i]][:k]

    # For each picked ray: jump skip_px beyond the first hit and keep going
    for i in picked:
        theta = angles[i]
        ux, uy = math.cos(theta), math.sin(theta)
        s0 = dists[i] + skip_px  # start a bit past the first obstacle
        free_len = 0

        for s in range(s0, MAX_RANGE_PX + 1):
            u = int(CX + ux * s)
            v = int(CY - uy * s)
            if not (0 <= u < W and 0 <= v < H):
                break
            if binary[v, u] != 0:
                break
            free_len += 1

        if free_len >= threshold_px:
            return True

    return False



class M_Controller(Node):
    def __init__(self):
        super().__init__("m_controller")
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)   
        self.create_subscription(Float64MultiArray, '/igvc/next_waypoint', self.next_waypoint_cb, 10)
        self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)
        self.create_subscription(Float64MultiArray, '/lane_directions_angles', self.lane_direction_angles_cb, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        
        if DEBUG:
            self.marker_publisher = self.create_publisher(Marker, "/debug/intersection/lane_marker", 10)
            self.lane_scan_2d_debug_publisher = self.create_publisher(Image, "/debug/intersection/lane_scan_2d_debug", 10)    

        self.costmap = None
        self.pts_xy = None
        self.qualification = False
        self.next_waypoint = None
        self.start_x_y = None
        self.best_theta = None
        self.linx_angz_to_publish = None
        self.lane_direction_theta1 = None
        self.lane_direction_theta2 = None
        self.absolute_theta1 = None
        self.absolute_theta2 = None

        self.explore_state = None           # None | "turn_left" | "sample_left" | "turn_right" | "sample_right" | "decide"
        self.explore_baseline_yaw = None
        self.explore_target_yaw = None
        self.explore_samples = {}           # {"left": dist_px, "right": dist_px}
        self._explore_settle_count = 0
        self.align_target_yaw = None        # used post-decision to align to chosen side

        # corner-avoidance micro-SM
        self.corner_state = None      # None | "align_center" | "check_center" | "turn_other" | "check_other" | "stop"
        self.corner_target_yaw = None
        self.corner_settle = 0
        self.corner_required_sign = 0 # sign of the initial correction to center
        self.corner_center_yaw = None      # yaw we aligned to (lane center)
        self.corner_other_side_sign = 0    # +1 = CCW (left), -1 = CW (right)

        # backup micro-state
        self.corner_backup_start = None   # (x, y)
        self.corner_backup_target = DISTANCE_TO_MOVE_BACK_ON_CORNER_DETECTED
        self.corner_backup_ticks = 0
        self.corner_backup_timeout_ticks = int(BACKUP_TIMEOUT_S * MY_HZ)

        # scoot micro-state
        self.scoot_target_yaw = None
        self.scoot_side_sign = 0      # +1 = left of center, -1 = right of center
        self.scoot_start = None       # (x, y)
        self.scoot_ticks = 0
        self.scoot_timeout_ticks = int(SCOOT_TIMEOUT_S * MY_HZ)

        # 45° try-both micro-state
        self.ca_45_a_yaw = None     # first 45° candidate (chosen by quick peek)
        self.ca_45_b_yaw = None     # second 45° candidate


        self.bridge = CvBridge()
        self.timer = self.create_timer(1/MY_HZ, self.publish_cmd)


    def costmap_cb(self, msg: OccupancyGrid):
        self.last_header = msg.header
        self.costmap = msg

    def lane_direction_angles_cb(self, msg: Float64MultiArray):
        if msg.data:
            self.lane_direction_theta1, self.lane_direction_theta2 = msg.data
            self.absolute_theta1 = normalise_angle(self.lane_direction_theta1 + self.yaw)
            self.absolute_theta2 = normalise_angle(self.lane_direction_theta2 + self.yaw)

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
        self.yaw = yaw



    def publish_cmd(self):
        if self.costmap is None:
            self.get_logger().info("self.costmap is None")
            return

        # ── Corner-avoidance micro state machine (runs before explore SM) ─────────────
        if self.corner_state is not None:
            if self.corner_state == "backup_init":
                # capture starting pose (extra safety if odom wasn’t ready)
                if self.corner_backup_start is None:
                    self.corner_backup_start = (getattr(self, "x", 0.0), getattr(self, "y", 0.0))
                self.corner_state = "backup_move"
                self.linx_angz_to_publish = (-BACKUP_SPEED, 0.0)

            elif self.corner_state == "backup_move":
                self.corner_backup_ticks += 1
                x = getattr(self, "x", 0.0); y = getattr(self, "y", 0.0)
                sx, sy = self.corner_backup_start
                dist = math.hypot(x - sx, y - sy)

                if (dist >= self.corner_backup_target) or (self.corner_backup_ticks >= self.corner_backup_timeout_ticks):
                    # stop and hand over to align-to-center
                    self.linx_angz_to_publish = (0.0, 0.0)
                    self.corner_settle = 0
                    self.corner_state = "align_center"
                else:
                    # keep reversing
                    self.linx_angz_to_publish = (-BACKUP_SPEED, 0.0)

            if self.corner_state == "align_center":
                yaw_err = normalise_angle(self.corner_target_yaw - self.yaw)
                if abs(yaw_err) > YAW_ALIGN_TOL:
                    ang_z = clamp(1.5 * yaw_err, -LEFT_TURN_ANGULAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
                    self.linx_angz_to_publish = (0.0, ang_z)
                else:
                    self.corner_settle += 1
                    self.linx_angz_to_publish = (0.0, 0.0)
                    if self.corner_settle >= EXPL_SETTLE_TICKS:
                        self.corner_settle = 0
                        self.corner_state = "check_center"
                        
            elif self.corner_state == "check_center":
                _, max_dist = self.radial_scans_from_costmap(self.costmap)
                if max_dist >= AFTER_CENTERING_CORNER_AVOIDANCE_THRESHOLD:
                    # Good path ahead; clear corner-SM and proceed normally (let normal logic move forward)
                    self.corner_state = None
                    self.linx_angz_to_publish = (LINEAR_SPEED, 0.0)
                else:
                    # === NEW: evaluate BOTH center ±45° and try the clearer one first ===
                    base = self.corner_center_yaw if self.corner_center_yaw is not None else self.yaw
                    th_plus  = normalise_angle(base + EXPL_ROT_ANGLE)
                    th_minus = normalise_angle(base - EXPL_ROT_ANGLE)

                    # Quick peeks (no turning yet)
                    binary, CX, CY, W, H, _ = _binary_from_costmap(self.costmap)
                    len_plus  = _ray_length_px(binary, CX, CY, th_plus,  W, H)
                    len_minus = _ray_length_px(binary, CX, CY, th_minus, W, H)

                    # Pick the better as A; other as B (tie-break → use +45° first)
                    if len_plus >= len_minus:
                        self.ca_45_a_yaw, self.ca_45_b_yaw = th_plus, th_minus
                    else:
                        self.ca_45_a_yaw, self.ca_45_b_yaw = th_minus, th_plus

                    # Turn to A first
                    self.corner_target_yaw = self.ca_45_a_yaw
                    self.corner_state = "turn_45_a"
                    self.corner_settle = 0


            elif self.corner_state == "turn_45_a":
                yaw_err = normalise_angle(self.corner_target_yaw - self.yaw)
                if abs(yaw_err) > YAW_ALIGN_TOL:
                    ang_z = clamp(1.5 * yaw_err, -LEFT_TURN_ANGULAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
                    self.linx_angz_to_publish = (0.0, ang_z)
                else:
                    self.corner_settle += 1
                    self.linx_angz_to_publish = (0.0, 0.0)
                    if self.corner_settle >= EXPL_SETTLE_TICKS:
                        self.corner_settle = 0
                        self.corner_state = "check_45_a"

            elif self.corner_state == "check_45_a":
                _, max_dist = self.radial_scans_from_costmap(self.costmap)
                if max_dist >= AFTER_CENTERING_CORNER_AVOIDANCE_THRESHOLD:
                    self.corner_state = None
                    self.linx_angz_to_publish = (LINEAR_SPEED, 0.0)
                else:
                    # Try the other 45°
                    self.corner_target_yaw = self.ca_45_b_yaw
                    self.corner_state = "turn_45_b"
                    self.corner_settle = 0

            elif self.corner_state == "turn_45_b":
                yaw_err = normalise_angle(self.corner_target_yaw - self.yaw)
                if abs(yaw_err) > YAW_ALIGN_TOL:
                    ang_z = clamp(1.5 * yaw_err, -LEFT_TURN_ANGULAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
                    self.linx_angz_to_publish = (0.0, ang_z)
                else:
                    self.corner_settle += 1
                    self.linx_angz_to_publish = (0.0, 0.0)
                    if self.corner_settle >= EXPL_SETTLE_TICKS:
                        self.corner_settle = 0
                        self.corner_state = "check_45_b"

            elif self.corner_state == "check_45_b":
                _, max_dist = self.radial_scans_from_costmap(self.costmap)
                if max_dist >= AFTER_CENTERING_CORNER_AVOIDANCE_THRESHOLD:
                    self.corner_state = None
                    self.linx_angz_to_publish = (LINEAR_SPEED, 0.0)
                else:
                    # Both ±45° failed → fall back to SCOOT (which already peeks both perpendiculars)
                    base = self.corner_center_yaw if self.corner_center_yaw is not None else self.yaw
                    theta_left  = normalise_angle(base + math.pi/2.0)
                    theta_right = normalise_angle(base - math.pi/2.0)

                    binary, CX, CY, W, H, _ = _binary_from_costmap(self.costmap)
                    left_len  = _ray_length_px(binary, CX, CY, theta_left,  W, H)
                    right_len = _ray_length_px(binary, CX, CY, theta_right, W, H)

                    # choose side with longer clear run; tie-break => LEFT
                    self.scoot_side_sign = 1 if left_len >= right_len else -1
                    self.scoot_target_yaw = normalise_angle(base + self.scoot_side_sign * (math.pi / 2.0))

                    # init forward scoot
                    self.scoot_start = (getattr(self, "x", 0.0), getattr(self, "y", 0.0))
                    self.scoot_ticks = 0

                    self.corner_state = "scoot_turn"
                    self.corner_settle = 0





            elif self.corner_state == "scoot_turn":
                yaw_err = normalise_angle(self.scoot_target_yaw - self.yaw)
                if abs(yaw_err) > YAW_ALIGN_TOL:
                    ang_z = clamp(1.5 * yaw_err, -LEFT_TURN_ANGULAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
                    self.linx_angz_to_publish = (0.0, ang_z)
                else:
                    self.corner_settle += 1
                    self.linx_angz_to_publish = (0.0, 0.0)
                    if self.corner_settle >= EXPL_SETTLE_TICKS:
                        self.corner_settle = 0
                        # start scoot forward
                        if self.scoot_start is None:
                            self.scoot_start = (getattr(self, "x", 0.0), getattr(self, "y", 0.0))
                        self.scoot_ticks = 0
                        self.corner_state = "scoot_forward"

            elif self.corner_state == "scoot_forward":
                # drive forward in the sideways heading
                self.scoot_ticks += 1
                x = getattr(self, "x", 0.0); y = getattr(self, "y", 0.0)
                sx, sy = self.scoot_start
                dist = math.hypot(x - sx, y - sy)

                if (dist >= SCOOT_DISTANCE_M) or (self.scoot_ticks >= self.scoot_timeout_ticks):
                    # stop and re-align to lane center heading
                    self.linx_angz_to_publish = (0.0, 0.0)
                    self.corner_settle = 0
                    # after scoot, face the original center heading
                    base = self.corner_center_yaw if self.corner_center_yaw is not None else self.yaw
                    self.corner_target_yaw = base
                    self.corner_state = "scoot_realign"
                else:
                    self.linx_angz_to_publish = (SCOOT_SPEED, 0.0)

            elif self.corner_state == "scoot_realign":
                yaw_err = normalise_angle(self.corner_target_yaw - self.yaw)
                if abs(yaw_err) > YAW_ALIGN_TOL:
                    ang_z = clamp(1.5 * yaw_err, -LEFT_TURN_ANGULAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
                    self.linx_angz_to_publish = (0.0, ang_z)
                else:
                    self.corner_settle += 1
                    self.linx_angz_to_publish = (0.0, 0.0)
                    if self.corner_settle >= EXPL_SETTLE_TICKS:
                        self.corner_settle = 0
                        # one more check from the new position; if still bad, we’ll stop
                        self.corner_state = "scoot_check"

            elif self.corner_state == "scoot_check":
                _, max_dist = self.radial_scans_from_costmap(self.costmap)
                if max_dist >= AFTER_CENTERING_CORNER_AVOIDANCE_THRESHOLD:
                    # success → hand control back to normal logic
                    self.corner_state = None
                    self.linx_angz_to_publish = (LINEAR_SPEED, 0.0)
                else:
                    self.get_logger().warn("Corner avoidance: center+other+side-scoot failed. Stopping.")
                    self.corner_state = "stop"



            elif self.corner_state == "stop":
                self.linx_angz_to_publish = (0.0, 0.0)

            # publish and return for this tick
            if self.linx_angz_to_publish:
                cmd = Twist()
                cmd.linear.x = float(self.linx_angz_to_publish[0])
                cmd.angular.z = float(self.linx_angz_to_publish[1])
                self.cmd_vel_publisher.publish(cmd)
            return
        # ── end corner-avoidance SM ───────────────────────────────────────────────────


        # --- Normal path: compute scans and act ---
        best_theta, max_dist = self.radial_scans_from_costmap(self.costmap)

        # --- Pre-check: can we see a good forward path just beyond the ends of the longest rays?
        precheck_ok = False
        if max_dist <= DISTANCE_THRESHOLD_TO_TRIGGER_CORNER_AVOIDANCE_MECHANISM:
            try:
                precheck_ok = top_k_endpoint_lookahead_ok(
                    self.costmap,
                    self.yaw,
                    k=10,
                    skip_px=3,
                    threshold_px=AFTER_CENTERING_CORNER_AVOIDANCE_THRESHOLD
                )
            except Exception as e:
                self.get_logger().warn(f"lookahead precheck failed: {e}")


        if precheck_ok:
            self.get_logger().info("Corner precheck: found viable continuation past ray endpoints; skipping corner avoidance.")


        # inside an obstacle
        if (best_theta, max_dist) == (-1, -1):
            self.get_logger().info("STOPPING BECAUSE IM INSIDE AN OBSTACLE")
            self.linx_angz_to_publish = (0.0, 0.0)


        elif (max_dist <= DISTANCE_THRESHOLD_TO_TRIGGER_CORNER_AVOIDANCE_MECHANISM and
            max_dist != -1 and self.lane_direction_theta1 is not None and
            True):

            self.get_logger().info("Corner Avoidance Mechanism Triggered")
            self.get_logger().info(f"Relative to the Bot: {self.lane_direction_theta1} | {self.lane_direction_theta2}")
            self.get_logger().info(f"Absolute Bot: {self.absolute_theta1} | {self.absolute_theta2}")

            # --- NEW: prefer waypoint heading (relative to bot) when available ---
            required_direction = None
            if self.next_waypoint is not None:
                if 'direction' in self.next_waypoint and self.next_waypoint['direction'] is not None:
                    required_direction = float(self.next_waypoint['direction'])

            # Fallback to the closer lane direction if waypoint heading isn't available
            if required_direction is None:
                if abs(self.lane_direction_theta1) < abs(self.lane_direction_theta2):
                    required_direction = self.lane_direction_theta1
                else:
                    required_direction = self.lane_direction_theta2

            # Sign of the centering correction (+CCW, -CW). If zero, stop (shouldn't happen normally).
            if required_direction == 0:
                raise RuntimeError("Required DIRECTION is 0")

            sign_correction = -1 if required_direction > 0 else 1
            self.corner_required_sign = sign_correction

            # 1) Rotate to center (align yaw to chosen direction)
            self.corner_target_yaw = normalise_angle(self.yaw + required_direction)
            self.corner_center_yaw = self.corner_target_yaw

            # pre-compute the opposite side relative to the centered heading
            self.corner_other_side_sign = -sign_correction

            # === Back up first ===
            self.corner_backup_start = (getattr(self, "x", 0.0), getattr(self, "y", 0.0))
            self.corner_backup_ticks = 0
            self.corner_state = "backup_init"
            self.corner_settle = 0

            # Command immediate reverse this tick
            self.linx_angz_to_publish = (-BACKUP_SPEED, 0.0)



        # Normal Mechanism
        else:
            # If a prior decision set an align target (from exploration), honor it until aligned
            if self.align_target_yaw is not None:
                heading_err = normalise_angle(self.align_target_yaw - self.yaw)
                if abs(heading_err) > np.deg2rad(3.0):
                    ang_z_required = clamp(1.5 * heading_err, -LEFT_TURN_ANGULAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
                    self.linx_angz_to_publish = (0.0, ang_z_required)
                else:
                    # aligned; clear and move forward
                    self.align_target_yaw = None
                    self.linx_angz_to_publish = (LINEAR_SPEED, 0.0)
            else:
                heading_err = normalise_angle(best_theta - self.yaw)
                ang_z_required = clamp(1.5 * heading_err, -LEFT_TURN_ANGULAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
                self.linx_angz_to_publish = (LINEAR_SPEED * (max(1-abs(ang_z_required/LEFT_TURN_ANGULAR_SPEED), MIN_LINEAR_SPEED_WHILE_TURNING)), ang_z_required)

                if abs(heading_err) < np.deg2rad(3.0):
                    self.best_theta = None
                    self.align_target_yaw = None
                    self.linx_angz_to_publish = (LINEAR_SPEED, 0.0)

        # Publish current command
        if self.linx_angz_to_publish:
            cmd = Twist()
            cmd.linear.x = float(self.linx_angz_to_publish[0])
            cmd.angular.z = float(self.linx_angz_to_publish[1])
            self.cmd_vel_publisher.publish(cmd)

    
    def next_waypoint_cb(self, msg:Float64MultiArray):
        distance, heading_error, idx = msg.data
        self.next_waypoint = {'distance':distance, 'direction':heading_error, 'waypoint_idx':idx}




    def radial_scans_from_costmap(self, costmap):
        """
        Ray-cast over an OccupancyGrid and pick a heading.
        If starting on high cost, search ONLY left/right (perpendicular to facing)
        within MIN_DISTANCE_FROM_ACTUAL_SPOT for a free pivot. If none, return (-1, -1).
        """

        # --- Tunables ---
        occ_thresh = 90
        treat_unknown_as_occupied = True
        robot_at_center = True
        robot_pose_map = None
        MIN_DISTANCE_FROM_ACTUAL_SPOT = 0.6  # meters, lateral search radius

        # --- 0) Extract grid/meta ---
        res = costmap.info.resolution
        W   = costmap.info.width
        H   = costmap.info.height

        grid = np.array(costmap.data, dtype=np.int16).reshape(H, W)

        hit_mask = (grid >= occ_thresh)
        if treat_unknown_as_occupied:
            hit_mask |= (grid < 0)
        binary = (hit_mask.astype(np.uint8) * 255)

        if np.any(binary):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            binary = cv2.dilate(binary, None, iterations=1)

        # --- 2) Robot cell (CX, CY) ---
        if robot_at_center:
            CX = W // 2
            CY = H // 2
        else:
            if robot_pose_map is None:
                raise ValueError("robot_pose_map=(x,y) required when robot_at_center=False")
            ox = costmap.info.origin.position.x
            oy = costmap.info.origin.position.y
            rx, ry = robot_pose_map
            cx = int((rx - ox) / res)
            cy = int((ry - oy) / res)
            CX = int(np.clip(cx, 0, W - 1))
            CY = int(np.clip(cy, 0, H - 1))

        # --- Facing and scan angles (keep your policy) ---
        facing = -self.yaw
        nominal_tol = np.deg2rad(45)

        # --- Start/pivot selection (lateral only if on high cost) ---
        start_in_high = (binary[CY, CX] != 0)
        pivot_CX, pivot_CY = CX, CY
        use_pivot = False
        R_px = int(max(0, round(MIN_DISTANCE_FROM_ACTUAL_SPOT / res)))

        if start_in_high:
            if R_px == 0:
                return -1, -1

            # Perpendicular (left/right) unit vector to facing
            # Facing dir = (cos f, sin f). Perp = (-sin f, cos f).
            nx = -np.sin(facing)
            ny =  np.cos(facing)

            found = False
            # search outward by distance s, checking right (+) then left (-) at each s
            for s in range(1, R_px + 1):
                # Right side
                u_r = int(round(CX + nx * s))
                v_r = int(round(CY - ny * s))  # minus because image v grows downward
                if 0 <= u_r < W and 0 <= v_r < H and binary[v_r, u_r] == 0:
                    pivot_CX, pivot_CY = u_r, v_r
                    found = True
                    break

                # Left side
                u_l = int(round(CX - nx * s))
                v_l = int(round(CY + ny * s))
                if 0 <= u_l < W and 0 <= v_l < H and binary[v_l, u_l] == 0:
                    pivot_CX, pivot_CY = u_l, v_l
                    found = True
                    break

            if not found:
                return -1, -1
            use_pivot = True

        # Use pivot (if any) as scan origin
        scan_CX, scan_CY = (pivot_CX, pivot_CY) if use_pivot else (CX, CY)

        angle_tolerance = nominal_tol
        nsteps = 2 * (2 * int(np.degrees(angle_tolerance)) + 1)
        angles = np.linspace(facing - angle_tolerance, facing + angle_tolerance, nsteps)
        dirs   = np.stack((np.cos(angles), np.sin(angles)), axis=1)

        MAX_RANGE_PX = int(max(W, H))
        dists     = np.full(len(angles), MAX_RANGE_PX, dtype=np.int32)
        never_hit = np.full(len(angles), True, dtype=bool)

        # --- Ray march from chosen center ---
        for k, (ux, uy) in enumerate(dirs):
            for s in range(0, MAX_RANGE_PX + 1):
                u = int(scan_CX + ux * s)
                v = int(scan_CY - uy * s)
                if not (0 <= u < W and 0 <= v < H):
                    break
                if binary[v, u] != 0:
                    dists[k] = s
                    never_hit[k] = False
                    break

        # --- Pick heading ---
        def pick_median_among_never_hit():
            nh_idx = np.flatnonzero(never_hit)
            best_idx = int(nh_idx[len(nh_idx) // 2])
            return best_idx, MAX_RANGE_PX

        def pick_longest_ray():
            best_idx = int(dists.argmax())
            return best_idx, int(dists[best_idx])

        if never_hit.any():
            best_idx, best_dist_px = pick_median_among_never_hit()
        else:
            best_idx, best_dist_px = pick_longest_ray()

        best_theta = float(angles[best_idx])

        # --- Debug viz ---
        msg_header   = self.last_header
        grid_img_pub = self.lane_scan_2d_debug_publisher
        cv2_bridge   = self.bridge

        viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Show robot cell (red), pivot (cyan), and the lateral search segment (yellow)
        cv2.circle(viz, (CX, CY), 3, (0, 0, 255), -1)
        if start_in_high:
            # draw left-right segment of length R_px along perpendicular
            p1 = (int(round(CX + (-np.sin(facing)) * R_px)), int(round(CY - ( np.cos(facing)) * R_px)))
            p2 = (int(round(CX - (-np.sin(facing)) * R_px)), int(round(CY + ( np.cos(facing)) * R_px)))
            cv2.line(viz, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        if use_pivot:
            cv2.circle(viz, (pivot_CX, pivot_CY), 3, (255, 255, 0), -1)

        def grid_uv(theta, dist_px):
            u = int(scan_CX + np.cos(theta) * dist_px)
            v = int(scan_CY - np.sin(theta) * dist_px)
            return u, v

        # Rays
        for k, (theta, dist_px) in enumerate(zip(angles, dists)):
            if never_hit[k]:
                dist_px = MAX_RANGE_PX
            u_end, v_end = grid_uv(theta, dist_px)
            cv2.line(viz, (scan_CX, scan_CY), (u_end, v_end), (128, 128, 255), 1, cv2.LINE_AA)

        # Best ray (green)
        u_best, v_best = grid_uv(best_theta, best_dist_px)
        cv2.arrowedLine(viz, (scan_CX, scan_CY), (u_best, v_best),
                        (0, 255, 0), 2, tipLength=0.08, line_type=cv2.LINE_AA)

        # Bot yaw (blue) from actual robot cell
        arrow_len_px = int(25 / res)
        u_yaw = int(CX + np.cos(self.yaw) * arrow_len_px)
        v_yaw = int(CY + np.sin(self.yaw) * arrow_len_px)
        cv2.arrowedLine(viz, (CX, CY), (u_yaw, v_yaw),
                        (255, 0, 0), 2, tipLength=0.08, line_type=cv2.LINE_AA)
        

        # ---- Lane-direction arrows (magenta & yellow), if available ----
        def draw_angle_arrow(img, origin_u, origin_v, theta, length_px, color, label):
            u2 = int(origin_u + np.cos(theta) * length_px)
            v2 = int(origin_v + np.sin(theta) * length_px)  # match your yaw arrow convention
            cv2.arrowedLine(img, (origin_u, origin_v), (u2, v2),
                            color, 2, tipLength=0.08, line_type=cv2.LINE_AA)
            # Put a small label slightly beyond the tip
            tu = int(origin_u + np.cos(theta) * (length_px + 10))
            tv = int(origin_v + np.sin(theta) * (length_px + 10))
            cv2.putText(img, label, (tu, tv), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        if self.absolute_theta1 is not None and self.absolute_theta2 is not None:
            lane_arrow_len_px = int(30 / res)  # a bit longer than yaw arrow
            # Use ACTUAL robot cell as origin so all three (yaw + lanes) share the same base
            draw_angle_arrow(
                viz, CX, CY, self.absolute_theta1,
                lane_arrow_len_px, (255, 0, 255),  # magenta
                f"θ1 {np.degrees(self.absolute_theta1):.0f}°"
            )
            draw_angle_arrow(
                viz, CX, CY, self.absolute_theta2,
                lane_arrow_len_px, (0, 255, 255),  # yellow
                f"θ2 {np.degrees(self.absolute_theta2):.0f}°"
            )

        # ---- Waypoint heading (relative), drawn as absolute from robot cell ----
        if self.next_waypoint is not None and ('direction' in self.next_waypoint):
            wp_rel = self.next_waypoint['direction']
            if wp_rel is not None:
                wp_abs = normalise_angle(self.yaw + float(wp_rel))
                lane_arrow_len_px = int(30 / res)
                draw_angle_arrow(
                    viz, CX, CY, wp_abs,
                    lane_arrow_len_px, (0, 165, 255),  # orange-ish for contrast
                    f"WP {np.degrees(wp_abs):.0f}°"
                )

        viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
        viz = cv2.flip(viz, 1)

        img_msg = cv2_bridge.cv2_to_imgmsg(viz, encoding="bgr8")
        img_msg.header = msg_header
        grid_img_pub.publish(img_msg)

        return -best_theta, int(max(dists))



def main(args=None):
    rclpy.init(args=args)
    node = M_Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
