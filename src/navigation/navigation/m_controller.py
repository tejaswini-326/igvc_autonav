from math import radians
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import numpy as np
from nav_msgs.msg import OccupancyGrid
from math import radians
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from message_filters import Subscriber, ApproximateTimeSynchronizer
from math import pi, degrees
import numpy as np
import cv2
from std_msgs.msg import Header


# ──────────────────────────────────────────────────────────────────────────────
# Setting this to true will generate a new topic:
# intersection_llane_scan_2d_debug - An image showing all polar scans and detected distances
DEBUG = True

# Movement Related
LINEAR_SPEED                                    = 0.5                # m/s   (forward)
LEFT_TURN_ANGULAR_SPEED                         = 0.2               # rad/s (+ve = CCW = left)

ANGLE_TOLERANCE                                 = radians(30)        # ± deg window around 90° – θ

MY_HZ = 60
# ──────────────────────────────────────────────────────────────────────────────


def normalise_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


class M_Controller(Node):
    def __init__(self):
        super().__init__("m_controller")
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)   
        self.create_subscription(Float64MultiArray, '/igvc/next_waypoint', self.next_waypoint_cb, 10)
        self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)
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

        self.bridge = CvBridge()
        self.timer = self.create_timer(1/MY_HZ, self.publish_cmd)


    def costmap_cb(self, msg: OccupancyGrid):
        self.last_header = msg.header
        self.costmap = msg


    def odom_cb(self, msg: Odometry): 
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
        self.yaw = yaw


    def publish_cmd(self):        
        if self.costmap is None:
            self.get_logger().info("self.costmap is None")
            return
        
        best_theta = self.radial_scans_from_costmap(self.costmap)
        heading_err = normalise_angle(best_theta - self.yaw)
        self.get_logger().info(f"Yaw {self.yaw}")
        self.get_logger().info(f"Best theta {best_theta}")
        self.get_logger().info(f"Heading Error {heading_err}")
        ang_z_required = max(-LEFT_TURN_ANGULAR_SPEED, min(LEFT_TURN_ANGULAR_SPEED, 1.5 * heading_err))
        self.linx_angz_to_publish = (LINEAR_SPEED, ang_z_required)
        if abs(heading_err) < np.deg2rad(3.0):
            self.best_theta = None
            self.align_target_yaw = None
            self.linx_angz_to_publish = (LINEAR_SPEED, 0)

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
        Ray-cast over an OccupancyGrid and pick a heading similar to your earlier pointcloud-based version.
        Returns: best_theta (radians) or (for 'horizontal line detect') the per-angle distances (pixels).

        Notes:
        - Coordinate convention kept consistent with your earlier grid math:
        image u increases to the right, v increases downward; robot at (CX, CY).
        - Rays 'hit' when they encounter a nonzero pixel in the binary mask
        (obstacle or unknown depending on flags).
        """

        occ_thresh=90 # cells >= this are considered obstacles. the costmap originally had 0-255 but only 0-100 is getting published
        treat_unknown_as_occupied=True
        robot_at_center=True         # True for nav2 local costmap (rolling window)
        robot_pose_map=None          # (x,y) in same frame as costmap if robot_at_center=False
        
        # --- 0) Extract grid/meta ---
        res   = costmap.info.resolution
        W     = costmap.info.width
        H     = costmap.info.height

        grid = np.array(costmap.data, dtype=np.int16).reshape(H, W)

        hit_mask = (grid >= occ_thresh)
        if treat_unknown_as_occupied:
            hit_mask |= (grid < 0)
        binary = (hit_mask.astype(np.uint8) * 255)

        # Optional thickening/smoothing (similar to your morphology)
        if np.any(binary):
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            binary  = cv2.dilate(binary, None, iterations=1)

        # --- 2) Robot cell (CX, CY) ---
        if robot_at_center:
            CX = W // 2
            CY = H // 2
        else:
            # Use world pose against grid origin to compute indices
            if robot_pose_map is None:
                raise ValueError("robot_pose_map=(x,y) required when robot_at_center=False")
            ox = costmap.info.origin.position.x
            oy = costmap.info.origin.position.y
            rx, ry = robot_pose_map
            cx = int((rx - ox) / res)
            cy = int((ry - oy) / res)
            CX = np.clip(cx, 0, W - 1)
            CY = np.clip(cy, 0, H - 1)

        # --- Decide angle span: wider if we start inside high-cost ---
        start_in_high = (binary[CY, CX] != 0)

        facing = -self.yaw
        # your nominal tolerance
        nominal_tol = radians(35)
        # narrower when starting in a blob (tune as needed)
        inside_blob_tol = radians(20)

        angle_tolerance = inside_blob_tol if start_in_high else nominal_tol
        nsteps = 2*(2 * int(degrees(angle_tolerance)) + 1)
        angles = np.linspace(facing - angle_tolerance, facing + angle_tolerance, nsteps)

        dirs = np.stack((np.cos(angles), np.sin(angles)), axis=1)

        MAX_RANGE_PX = int(max(W, H))
        dists     = np.full(len(angles), MAX_RANGE_PX, dtype=np.int32)
        never_hit = np.full(len(angles), True, dtype=bool)

        # --- Ray march with "initial forgiveness" if we start on high cost ---
        for k, (ux, uy) in enumerate(dirs):
            if not start_in_high:
                # Original behavior: stop at first high-cost cell
                for s in range(0, MAX_RANGE_PX + 1):
                    u = int(CX + ux * s)
                    v = int(CY - uy * s)
                    if not (0 <= u < W and 0 <= v < H):
                        break
                    if binary[v, u] != 0:            # first high → hit
                        dists[k] = s
                        never_hit[k] = False
                        break
            else:
                # Forgive initial high region:
                # phase 0: still in high; phase 1: in free; next high → hit
                phase = 0
                for s in range(0, MAX_RANGE_PX + 1):
                    u = int(CX + ux * s)
                    v = int(CY - uy * s)
                    if not (0 <= u < W and 0 <= v < H):
                        break
                    cell_high = (binary[v, u] != 0)
                    if phase == 0:
                        if not cell_high:
                            phase = 1                 # entered free region
                        # else keep forgiving while high near start
                    else:  # phase == 1
                        if cell_high:
                            dists[k] = s              # first high after free
                            never_hit[k] = False
                            break
                # If we never re-enter high, never_hit[k] stays True

        # --- 5) Choose heading (unchanged policy) ---
        def pick_median_among_never_hit():
            nh_idx = np.flatnonzero(never_hit)
            best_idx = int(nh_idx[len(nh_idx) // 2])
            best_dist_px = MAX_RANGE_PX
            return best_idx, best_dist_px

        def pick_longest_ray():
            best_idx     = int(dists.argmax())
            best_dist_px = int(dists[best_idx])
            return best_idx, best_dist_px

        if never_hit.any():
            best_idx, best_dist_px = pick_median_among_never_hit()
        else:
            best_idx, best_dist_px = pick_longest_ray()

        best_theta = float(angles[best_idx])

        # --- 6) Optional debug publishers (matches your earlier pattern) ---
        msg_header = self.last_header
        grid_img_pub = self.lane_scan_2d_debug_publisher
        cv2_bridge = self.bridge

        viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        def grid_uv(theta, dist_px):
            u = int(CX + np.cos(theta) * dist_px)
            v = int(CY - np.sin(theta) * dist_px)
            return u, v

        # all rays (faint)
        for k, (theta, dist_px) in enumerate(zip(angles, dists)):
            if dist_px < 0:
                continue
            if never_hit[k]:
                dist_px = MAX_RANGE_PX
            u_end, v_end = grid_uv(theta, dist_px)
            cv2.line(viz, (CX, CY), (u_end, v_end), (128,128,255), 1, cv2.LINE_AA)

        # best ray (green)
        u_best, v_best = grid_uv(best_theta, best_dist_px)
        cv2.arrowedLine(viz, (CX, CY), (u_best, v_best), (0,255,0), 2, tipLength=0.08, line_type=cv2.LINE_AA)
        # Robot current direction (blue)
        arrow_len_px = int(25 / res)   # 25 cm long arrow (tweak as you like)
        u_yaw = int(CX + np.cos(self.yaw) * arrow_len_px)
        v_yaw = int(CY + np.sin(self.yaw) * arrow_len_px)
        cv2.arrowedLine(
            viz, (CX, CY), (u_yaw, v_yaw),
            (255,0,0), 2, tipLength=0.08, line_type=cv2.LINE_AA
        )

        viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
        viz = cv2.flip(viz, 1)

        img_msg = cv2_bridge.cv2_to_imgmsg(viz, encoding="bgr8")
        img_msg.header = msg_header
        grid_img_pub.publish(img_msg)

        return -best_theta




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
