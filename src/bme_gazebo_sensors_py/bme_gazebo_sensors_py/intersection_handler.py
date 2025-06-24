from math import hypot, pi, radians
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from nav_msgs.msg import Odometry
from bme_gazebo_sensors_py.left_inter_completion_detector import LeftIntersectionDetector
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from math import radians, degrees
import struct
from std_msgs.msg import String
import cv2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point




# ──────────────────────────────────────────────────────────────────────────────

# Setting this to true will generate 3 new topics:
# intersection_lane_marker - Marker object showing direction chosen
# intersection_filtered_white - Pointcloud showing filtered white points in blue colour
# intersection_llane_scan_2d_debug - An image showing all polar scans and detected distances
DEBUG = True

# Lane Detection Related
MIN_NUMBER_OF_FILTERED_COLOURED_POINTS_REQUIRED = 60 

# Movement Related
LINEAR_SPEED                                    = 1.0                # m/s   (forward)
CMD_VEL_PUBLISHING_TIME_INTERVAL                = 0.1                # Time interval between 2 publishers in seconds

# Intersection Turning Related
ANGLE_TOLERANCE                                 = radians(30)        # ± deg window around 90° – θ
INITIAL_INTERSECTION_FORWARD_MOVEMENT           = 3                 # metres
LEFT_TURN_ANGULAR_SPEED                         = 0.1          # rad/s (+ve = CCW = left)
TURN_ANGLE                                      = radians(90.0)      # 90 was over-turning for me? I'm not sure why though
# ──────────────────────────────────────────────────────────────────────────────

def normalise_angle(angle: float) -> float:
    """Wrap `angle` to the interval [-π, π].""" 
    return (angle + pi) % (2 * pi) - pi


class PointcloudLeftTurnDriver(Node):
    def __init__(self):
        super().__init__("pointcloud_left_turn_driver")
        self.create_subscription(String, '/intersection', self.intersection_cb, 10)
        self.should_drive = False
        self.create_subscription(PointCloud2, "/camera/points", self.pointcloud_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 50)   
        self.cmd_vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        if DEBUG:
            self.marker_publisher = self.create_publisher(Marker, "intersection_lane_marker", 10)
            self.filtered_white_points_publilsher = self.create_publisher(PointCloud2, "intersection_filtered_white", 10)
            self.lane_scan_2d_debug_publisher = self.create_publisher(Image, "intersection_llane_scan_2d_debug", 10)    

        
        #subprocess.Popen(["python3", script_path])
        # Internal state
        self.prev_x = None
        self.prev_y = None
        self.distance_travelled = 0.0

        self.turning = False
        self.raw_turn_completed = False
        self.turn_start_yaw = None

        self.best_theta = None
        self.to_publish_because_of_polar_scans = None

        self.timer = self.create_timer(CMD_VEL_PUBLISHING_TIME_INTERVAL, self.publish_cmd)

        if DEBUG:
            self.get_logger().info(f"⏩ Driving {INITIAL_INTERSECTION_FORWARD_MOVEMENT:g} m, then left-turn 90 °, then straight again.")

    def intersection_cb(self, msg: String):
        if msg.data.lower() == "left":
            self.should_drive = True
            self.get_logger().info("🛑 Received 'left' from /intersection. Activating left-turn behavior.")
        else:
            self.should_drive = False
            self.get_logger().info(f"🛑 Received '{msg.data}' from /intersection. Ignoring.")

    def pointcloud_cb(self, msg: PointCloud2):
        # ----------------------------------------------------------------------
        # 1) Extracting required points from the pointcloud
        # ----------------------------------------------------------------------
        if not self.should_drive:
            return
        height = msg.height
        width = msg.width
        white_img = np.zeros((height, width, 3), dtype=np.uint8)
        pts_xy = []
        index = 0
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
            x, y, z, rgb = point
            row = index // width
            col = index % width
            index += 1

            # Skip invalid points
            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            except:
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            white_threshold = 90
            color_balance_threshold = 50
            
            # Check if pixel is white (high intensity + balanced RGB)
            avg_color = (r + g + b) / 3
            if (r > white_threshold and g > white_threshold and b > white_threshold and
                abs(r - avg_color) < color_balance_threshold and 
                abs(g - avg_color) < color_balance_threshold and 
                abs(b - avg_color) < color_balance_threshold):

                # Ground level filtering
                if z < 0:  # Adjusted range
                    white_img[row, col] = (255, 255, 255)
                    pts_xy.append([x, y])  # Store x,y,z coordinates

        if len(pts_xy) < MIN_NUMBER_OF_FILTERED_COLOURED_POINTS_REQUIRED:
            return

        pts_xy = np.asarray(pts_xy) # shape (N,2)

        if DEBUG:
            if len(pts_xy) > 0:
                BLUE_RGB_INT   = 0x0000FF
                BLUE_RGB_FLOAT = struct.unpack('f',
                                            struct.pack('I', BLUE_RGB_INT))[0]

                cloud_pts = [[x, y, 0.0, BLUE_RGB_FLOAT]
                            for (x, y) in pts_xy]

                fields = [PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
                        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
                        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
                        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]

                cloud_msg = pc2.create_cloud(msg.header, fields, cloud_pts)
                self.filtered_white_points_publilsher.publish(cloud_msg)

        # ----------------------------------------------------------------------
        # 2) Build a bird-eye binary image of white-floor points
        # ----------------------------------------------------------------------
        
        # We will create a slightly blurry bird's eye floor plan
        # As we shoot out our polar scans, if we hit a white pixel, we want to stop.
        # Because point cloud data has holes in it, we have intentionally set a poor resolution
        # So that every 'grid' in our representation contains at least some point of the pointcloud to create a continuous white stretch where the lanes are there
        # On top of this, we'll do cv2 morpology and dilation to ensure there are no holes
        # If you turn on DEBUG, you can see the grid made in the topic 'intersection_llane_scan_2d_debug'

        GRID_RES   = 0.1             # 0.1 metres per pixel
        GRID_SIZE  = 200             # Number of pixels
        CX = CY    = GRID_SIZE // 2  # robot centred

        binary = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

        for x, y in pts_xy:
            # shift robot to centre → map x (fwd), -y (left) to image coords
            u = int(CX +  x / GRID_RES)
            v = int(CY + -y / GRID_RES)
            if 0 <= u < GRID_SIZE and 0 <= v < GRID_SIZE:
                binary[v, u] = 255

        # Thicken the blobs to ensure no spaces are missed
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary  = cv2.dilate(binary, None, iterations=1) 

        # ----------------------------------------------------------------------
        # 3) Polar-clearance scan (“laser fan”) instead of Hough 
        #    Shoots virtual rays ahead and chooses the clearest heading
        # ----------------------------------------------------------------------

        # Bail out if we don’t yet know the desired lane heading
        if self.turn_start_yaw is None:
            if DEBUG:
                self.get_logger().info("Turn start yaw is None")
            return

        # Figure out the angle we 'want to go to', this is just 90 degrees left from the original starting point
        # and as we turn, we update it according to the odometer readings
        wanted =  pi/2 - normalise_angle(self.prev_yaw - self.turn_start_yaw)

        # Set up fan of directions
        ANGLES_DEG = np.arange(-45, 45, 1)                # 1-degree resolution
        angles     = np.deg2rad(ANGLES_DEG)
        dirs       = np.stack((np.cos(angles), np.sin(angles)), axis=1)

        MAX_RANGE_M  = 100000.0
        MAX_RANGE_PX = int(MAX_RANGE_M / GRID_RES)

        # initialise all rays as “clear to max range”
        dists     = np.full(len(angles), MAX_RANGE_PX, dtype=np.int32)
        hit_mask  = np.zeros(len(angles), dtype=bool)     # True if we did hit paint

        # March each ray
        for k, (ux, uy) in enumerate(dirs):
            for s in range(0, MAX_RANGE_PX + 1):
                u = int(CX + ux * s)
                v = int(CY - uy * s)                      # minus because image-y down
                if not (0 <= u < GRID_SIZE and 0 <= v < GRID_SIZE):
                    # ran off map ⇒ keep MAX_RANGE_PX, no hit
                    break
                if binary[v, u]:                          # hit white stripe
                    dists[k]   = s
                    hit_mask[k] = True
                    break

        # When choosing our direction to head to, we will only select rays that are within in the ANGLE_TOLERANCE of this wanted angle
        heading_err = np.abs([normalise_angle(a - wanted) for a in angles])
        on_lane_dir = heading_err < ANGLE_TOLERANCE
        dists[~on_lane_dir] = -1

        if self.raw_turn_completed and  (dists == -1).all():
            self.get_logger().info("No rays inside ANGLE_TOLERANCE band")
            return

        never_hit = (~hit_mask) & on_lane_dir

        if never_hit.any():
            # ---------------------------------------------------------
            # 1. take **all** never-hit indices, already sorted L→R
            # 2. pick the true middle one (median by position)
            #    – if there is an even count we bias slightly left
            # ---------------------------------------------------------
            nh_idx = np.flatnonzero(never_hit)          # integer positions
            best_idx = int(nh_idx[len(nh_idx) // 2])    # middle element
            best_dist_px = MAX_RANGE_PX                 # “infinite” range
        else:
            best_idx     = dists.argmax()               # longest finite ray
            best_dist_px = dists[best_idx]

        best_theta  = angles[best_idx]
        self.best_theta = best_theta

        # ----------------------------------------------------------------------
        # 4) Publish all candidate rays in blue + highlight the chosen best one in green
        # ----------------------------------------------------------------------

        if DEBUG:
            clear_mkr            = Marker()
            clear_mkr.header     = msg.header
            clear_mkr.ns         = "lane_all"
            clear_mkr.id         = 0
            clear_mkr.action     = Marker.DELETEALL
            self.marker_publisher.publish(clear_mkr)


            def ray_end(theta, dist_px):
                """helper: pixel distance → (x,y) metres in robot frame"""
                return (float(dist_px * GRID_RES * np.cos(theta)),
                        float(dist_px * GRID_RES * np.sin(theta)))

            # 1.  Publish every on-lane ray (except best) as thin blue
            for k, (theta, dist_px) in enumerate(zip(angles, dists)):
                # ----- publish *all* rays, even those with dist_px == -1 -------------
                colour_blue  = (0.0, 0.0, 1.0, 0.4)      # default: translucent blue
                if k == best_idx:
                    continue                             # best ray handled later
                if dist_px < 0:                          # invalid / out of band
                    colour_blue = (1.0, 0.0, 0.0, 0.3)   # show invalids in faint red
                    dist_px     = MAX_RANGE_PX           # draw full-length for clarity

                mkr              = Marker()
                mkr.header       = msg.header
                mkr.ns           = "lane_all"
                mkr.id           = k + 1
                mkr.type         = Marker.LINE_STRIP
                mkr.action       = Marker.ADD
                mkr.scale.x      = 0.04
                mkr.color.r, mkr.color.g, mkr.color.b, mkr.color.a = colour_blue
                x_end, y_end     = ray_end(theta, dist_px)
                mkr.points       = [Point(x=0.0, y=0.0, z=0.0),
                                    Point(x=x_end, y=y_end, z=0.0)]
                mkr.lifetime     = rclpy.duration.Duration(seconds=0.2).to_msg()
                self.marker_publisher.publish(mkr)

            # 2.  Publish the *best* ray in thick green
            best_mkr            = Marker()
            best_mkr.header     = msg.header
            best_mkr.ns         = "lane_best"
            best_mkr.id         = 0
            best_mkr.type       = Marker.LINE_STRIP
            best_mkr.action     = Marker.ADD
            best_mkr.scale.x    = 0.05
            best_mkr.color.r    = 0.0
            best_mkr.color.g    = 1.0                    # green highlight
            best_mkr.color.b    = 0.0
            best_mkr.color.a    = 1.0
            x_best, y_best      = ray_end(best_theta, best_dist_px)
            best_mkr.points     = [Point(x=0.0, y=0.0, z=0.0),
                                Point(x=x_best, y=y_best, z=0.0)]
            best_mkr.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            self.marker_publisher.publish(best_mkr)


        # ----------------------------------------------------------------------
        # 5) Visual-debug image on the 2-D grid
        # ----------------------------------------------------------------------

        if DEBUG:
            # A. build a white-lane bitmap → 3-ch image
            viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            # helper (same pixel maths you used earlier)
            def grid_uv(theta, dist_px):
                u = int(CX + np.cos(theta) * dist_px)
                v = int(CY - np.sin(theta) * dist_px)   # image-y is down
                return u, v

            # 1) draw every on-lane ray very faintly
            for k, (theta, dist_px) in enumerate(zip(angles, dists)):
                if dist_px < 0:          # skip invalid / out-of-band rays
                    continue
                u, v = grid_uv(theta, dist_px)
                                   # light red-blue
                
                

            colour = (128, 128, 255)
            thickness = 1
            # March each ray
            for k, (ux, uy) in enumerate(dirs):
                for s in range(0, MAX_RANGE_PX + 1):
                    u = int(CX + ux * s)
                    v = int(CY - uy * s)                      # minus because image-y down
                    cv2.line(viz, (CX, CY), (u, v), colour, thickness, cv2.LINE_AA)
                    if not (0 <= u < GRID_SIZE and 0 <= v < GRID_SIZE):
                        # ran off map ⇒ keep MAX_RANGE_PX, no hit
                        break
                    if binary[v, u]:                          # hit white stripe
                        dists[k]   = s
                        hit_mask[k] = True
                        break

            # 2) draw the *best* ray nice and bold
            u_best, v_best = grid_uv(best_theta, best_dist_px)
            cv2.arrowedLine(
                viz, (CX, CY), (u_best, v_best),
                (0, 255, 0),              # bright green
                2, tipLength=0.08, line_type=cv2.LINE_AA
            )

            viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # B. publish to a ROS image topic
            if not hasattr(self, "bridge"):
                from cv_bridge import CvBridge
                self.bridge = CvBridge()


            img_msg = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            img_msg.header = msg.header            # time-sync with point cloud
            self.lane_scan_2d_debug_publisher.publish(img_msg)




    def odom_cb(self, msg: Odometry):
        if not self.should_drive:
            return
        # ─────────────────────────────────────────────────────────────────────
        # 0.  Current pose & yaw  (make yaw available everywhere below)
        # ─────────────────────────────────────────────────────────────────────
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))   # <── moved here
        self.prev_yaw = yaw                                       # keep latest copy

        if self.turn_start_yaw is not None:
            print(yaw, self.turn_start_yaw, normalise_angle(yaw - self.turn_start_yaw))

        # -------------------------------------------------------------------
        # 1.  Straight-line distance check
        # -------------------------------------------------------------------
        if self.prev_x is not None:
            self.distance_travelled += hypot(x - self.prev_x, y - self.prev_y)
        self.prev_x, self.prev_y = x, y

        if (not self.turning) and (not self.raw_turn_completed) \
                and self.distance_travelled >= INITIAL_INTERSECTION_FORWARD_MOVEMENT:
            self.turning = True
            self.turn_start_yaw = yaw
            self.get_logger().info("🚦 Reached distance - starting intersection left turn…")

        # -------------------------------------------------------------------
        # 2.  90 ° left-turn controller
        # -------------------------------------------------------------------
        if self.turning:
            if self.turn_start_yaw is None:
                self.turn_start_yaw = yaw
                return

            delta = normalise_angle(yaw - self.turn_start_yaw)
            if delta >= TURN_ANGLE:
                self.turning         = False
                self.raw_turn_completed  = True
                self.get_logger().info("✅ Raw Turn complete")

        # -------------------------------------------------------------------
        # 3.  Snap to lane heading (best_theta)  — uses *yaw* defined above
        # -------------------------------------------------------------------
        ALIGN_TOL     = np.deg2rad(3.0)
        KP_ALIGN      = 1.5

        if self.raw_turn_completed and (self.best_theta is not None):

            if DEBUG:
                self.get_logger().info(f"🧭  Aligning to {self.best_theta} °")

            twist           = Twist()
            twist.linear.x  = 0.0
            twist.angular.z = max(-LEFT_TURN_ANGULAR_SPEED, min(LEFT_TURN_ANGULAR_SPEED, KP_ALIGN * self.best_theta))
            self.to_publish_because_of_polar_scans = twist

            if abs(self.best_theta) < ALIGN_TOL:
                self.best_theta = None
                self.align_target_yaw = None
                twist.linear.x        = LINEAR_SPEED
                twist.angular.z       = 0.0
                self.to_publish_because_of_polar_scans = twist


    def publish_cmd(self):
        if not self.should_drive:
            return
        if self.to_publish_because_of_polar_scans:
            self.cmd_vel_publisher.publish(self.to_publish_because_of_polar_scans)
        else:
            cmd = Twist()
            cmd.linear.x = LINEAR_SPEED
            cmd.angular.z = LEFT_TURN_ANGULAR_SPEED if self.turning else 0.0
            self.cmd_vel_publisher.publish(cmd)

    # ───────────────────────────────────────────────────────────────────── #
    # Graceful shutdown                                                    #
    # ───────────────────────────────────────────────────────────────────── #
    def stop(self):
        self.cmd_vel_publisher.publish(Twist())  # send zero cmd


def main(args=None):
    rclpy.init(args=args)
    node = PointcloudLeftTurnDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
