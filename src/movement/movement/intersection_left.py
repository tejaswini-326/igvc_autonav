from math import radians
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import numpy as np
from math import radians
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from message_filters import Subscriber, ApproximateTimeSynchronizer
from movement.intersection_funcs import radial_scans, normalise_angle, get_merged_xy_points, zero_copy_xy_pointcloud_reader_view




# ──────────────────────────────────────────────────────────────────────────────
# Setting this to true will generate 3 new topics:
# intersection_lane_marker - Marker object showing direction chosen
# intersection_filtered_white - Pointcloud showing filtered white points in blue colour
# intersection_llane_scan_2d_debug - An image showing all polar scans and detected distances
DEBUG = True

# In the future add some other fallback for not enough points being detected
MIN_NUMBER_OF_FILTERED_COLOURED_POINTS_REQUIRED = 60 

# Movement Related
LINEAR_SPEED                                    = 1.5                # m/s   (forward)
LEFT_TURN_ANGULAR_SPEED                         = 0.2               # rad/s (+ve = CCW = left)

# Intersection Turning Related
ANGLE_TOLERANCE                                 = radians(30)        # ± deg window around 90° – θ
INITIAL_INTERSECTION_FORWARD_MOVEMENT_SQUARED   = (3) ** 2           # metres
TURN_ANGLE                                      = radians(80.0)      # 90 will overturn if the bot started out a bit turning to the left

# Completion Threshold - After this distance this node will handover control to main lane follower
TARGET_LEFT_DISPLACEMENT = 9
# ──────────────────────────────────────────────────────────────────────────────



class IntersectionLeftTurnDriver(Node):
    def __init__(self):
        super().__init__("intersection_left_turn_driver")
        self.create_subscription(String, '/intersection', self.intersection_cb, 10) 
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)   
        self.create_subscription(Float64MultiArray, '/igvc/next_waypoint', self.next_waypoint_cb, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.intersection_pub = self.create_publisher(String, '/intersection', 10)

        sub_white  = Subscriber(self, PointCloud2, '/igvc/white_points')
        sub_yellow = Subscriber(self, PointCloud2, '/igvc/yellow_points')
        sync = ApproximateTimeSynchronizer([sub_white, sub_yellow], queue_size=10, slop=0.1)
        sync.registerCallback(self.sync_cb)
        
        if DEBUG:
            self.marker_publisher = self.create_publisher(Marker, "/debug/intersection/lane_marker", 10)
            self.intersection_filtered_points_publisher = self.create_publisher(PointCloud2, "/debug/intersection/filtered_points", 10)
            self.lane_scan_2d_debug_publisher = self.create_publisher(Image, "/debug/intersection/lane_scan_2d_debug", 10)    

        # This variable will always be one of these four:
        # '0. waiting for /intersection'
        # '1. straight'
        # '2. raw turn'
        # '3. radial scan'
        self.stage = '0. waiting for /intersection'

        self.pts_xy = None
        self.qualification = False
        self.next_waypoint = None
        self.start_x_y = None
        self.turn_start_yaw = None
        self.best_theta = None
        self.linx_angz_to_publish = None

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.05, self.publish_cmd)


    def intersection_cb(self, msg: String):
        if msg.data.lower() == "left":
            self.stage = '1. straight'
            self.get_logger().info("🟢 Received 'left' from /intersection.")
        elif msg.data.lower() == "qualification_left":
            self.stage = '1. straight'
            self.qualification = True
            self.get_logger().info("🟢 Received 'qualification_left' from /intersection.")
        else:
            self.stage = '0. waiting for /intersection'
            self.get_logger().info(f"🛑 Ignoring '{msg.data}' from /intersection.")


    def sync_cb(self, white_msg: PointCloud2, yellow_msg: PointCloud2):
        self.pts_xy = get_merged_xy_points(white_msg, yellow_msg)
        self.last_white_msg_header = white_msg.header


    def odom_cb(self, msg: Odometry): 
        if self.stage == '0. waiting for /intersection':
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
        self.yaw = yaw

        if self.turn_start_yaw is None:
            self.turn_start_yaw = yaw

        if self.stage == '1. straight':
            self.linx_angz_to_publish = (LINEAR_SPEED, 0)
            if self.start_x_y is None:
                self.start_x_y = (x, y)
            if (x - self.start_x_y[0])**2 + (y - self.start_x_y[1])**2 >= INITIAL_INTERSECTION_FORWARD_MOVEMENT_SQUARED:
                self.stage = '2. raw turn'
                self.get_logger().info("✅ Stage 1. straight completed.") 

        elif self.stage == '2. raw turn':
            self.linx_angz_to_publish = (LINEAR_SPEED, LEFT_TURN_ANGULAR_SPEED)
            delta = normalise_angle(yaw - self.turn_start_yaw)
            if delta >= TURN_ANGLE:
                self.stage = '3. radial scan'
                self.get_logger().info("✅ Stage 2. raw turn completed.")

        elif self.stage == '3. radial scan' and (self.best_theta is not None):
            ang_z_required = max(-LEFT_TURN_ANGULAR_SPEED, min(LEFT_TURN_ANGULAR_SPEED, 1.5 * self.best_theta))
            self.linx_angz_to_publish = (0, ang_z_required)
            if abs(self.best_theta) < np.deg2rad(3.0):
                self.best_theta = None
                self.align_target_yaw = None
                self.linx_angz_to_publish = (LINEAR_SPEED, 0)

        # Compute rightward displacement
        dx = x - self.start_x_y[0]
        dy = y - self.start_x_y[1]
        left_displacement = dx * np.sin(-self.turn_start_yaw) + dy * np.cos(-self.turn_start_yaw)

        if DEBUG:
            self.get_logger().info(f"📏 Leftward displacement: {left_displacement:.2f} m")

        if left_displacement >= TARGET_LEFT_DISPLACEMENT:
            msg = String()
            if self.qualification: msg.data = "follow_barrel_and_stop"
            else: msg.data = "none"
            self.get_logger().info(f"✅ Intersection Left Turn Complete. Publishing '{msg.data}' into /intersection")
            self.intersection_pub.publish(msg)
            self.stage = '0. waiting for /intersection'
            self.start_x_y = None
            self.turn_start_yaw = None
            self.best_theta = None
            self.linx_angz_to_publish = None



    def publish_cmd(self):
        if self.stage == '0. waiting for /intersection':
            return

        if self.stage == '3. radial scan' and self.pts_xy is not None:
            if len(self.pts_xy) < MIN_NUMBER_OF_FILTERED_COLOURED_POINTS_REQUIRED:
                return
            if self.turn_start_yaw is None:
                if DEBUG:
                    self.get_logger().info("Turn start yaw is None")
                return
            if not DEBUG:
                self.best_theta = radial_scans(self.pts_xy, 'left', self.yaw, self.turn_start_yaw, ANGLE_TOLERANCE, None)
            else:
                debug_stuff = (self.last_white_msg_header, self.marker_publisher, self.lane_scan_2d_debug_publisher, self.intersection_filtered_points_publisher, self.bridge, self)
                self.best_theta = radial_scans(self.pts_xy, 'left', self.yaw, self.turn_start_yaw, ANGLE_TOLERANCE, debug_stuff)

        if self.linx_angz_to_publish:
            cmd = Twist()
            cmd.linear.x = float(self.linx_angz_to_publish[0])
            cmd.angular.z = float(self.linx_angz_to_publish[1])
            self.cmd_vel_publisher.publish(cmd)

    
    def next_waypoint_cb(self, msg:Float64MultiArray):
        distance, heading_error, idx = msg.data
        self.next_waypoint = {'distance':distance, 'direction':heading_error, 'waypoint_idx':idx}

        

def main(args=None):
    rclpy.init(args=args)
    node = IntersectionLeftTurnDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
