from math import hypot, pi, radians
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import numpy as np
from math import radians, degrees
from std_msgs.msg import String, Float64MultiArray
import cv2
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker

from movement.intersection_funcs import get_xy_of_all_white_and_yellow_points_from_pointcloud_msg, radial_scans, normalise_angle




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
LINEAR_SPEED_WHEN_RADIAL_SCAN_TURNING           = 0.5
LEFT_TURN_ANGULAR_SPEED                         = 0.22               # rad/s (+ve = CCW = left)

# Intersection Turning Related
ANGLE_TOLERANCE                                 = radians(20)        # ± deg window around 90° – θ
INITIAL_INTERSECTION_FORWARD_MOVEMENT_SQUARED   = (10)**2             # metres 

# Completion Threshold - After this distance this node will handover control to main lane follower
TARGET_FORWARD_DISPLACEMENT = 15
# ──────────────────────────────────────────────────────────────────────────────



class IntersectionStraightDriver(Node):
	def __init__(self):
		super().__init__("intersection_straight_driver")

		self.create_subscription(String, '/intersection', self.intersection_cb, 10) 
		self.create_subscription(PointCloud2, "/camera/points", self.pointcloud_cb, 10)
		self.create_subscription(Odometry, "/odom", self.odom_cb, 10)   
		self.create_subscription(Float64MultiArray, '/igvc/next_waypoint', self.next_waypoint_cb, 10)

		self.cmd_vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
		self.intersection_pub = self.create_publisher(String, '/intersection', 10)
		
		if DEBUG:
			self.marker_publisher = self.create_publisher(Marker, "/debug/intersection/lane_marker", 10)
			self.intersection_filtered_points_publisher = self.create_publisher(PointCloud2, "/debug/intersection/filtered_points", 10)
			self.lane_scan_2d_debug_publisher = self.create_publisher(Image, "/debug/intersection/lane_scan_2d_debug", 10)    

		
		self.next_waypoint = None

		self.start_x_y = None

		# This variable will always be one of these four:
		# '0. waiting for /intersection'
		# '1. straight'
		# '3. radial scan'
		self.stage = '0. waiting for /intersection'

		self.qualification = False
		self.turn_start_yaw = None
		self.best_theta = None
		self.linx_angz_to_publish = None

		self.bridge = CvBridge()
		self.timer = self.create_timer(0.05, self.publish_cmd)


	def intersection_cb(self, msg: String):
		if msg.data.lower() == "straight":
			self.stage = '1. straight'
			self.get_logger().info("🟢 Received 'straight' from /intersection.")
		elif msg.data.lower() == "qualification_straight":
			self.stage = '1. straight'
			self.qualification = True
			self.get_logger().info("🟢 Received 'qualification_straight' from /intersection.")
		else:
			self.stage = '0. waiting for /intersection'
			self.get_logger().info(f"🛑 Ignoring '{msg.data}' from /intersection.")

	def pointcloud_cb(self, msg: PointCloud2):
		if self.stage != '3. radial scan':
			return

		pts_xy = get_xy_of_all_white_and_yellow_points_from_pointcloud_msg(msg)

		if len(pts_xy) < MIN_NUMBER_OF_FILTERED_COLOURED_POINTS_REQUIRED:
			return

		if not DEBUG:
			self.best_theta = radial_scans(pts_xy, 'straight', self.yaw, self.turn_start_yaw, ANGLE_TOLERANCE, None)
		else:
			debug_stuff = (msg.header, self.marker_publisher, self.lane_scan_2d_debug_publisher, self.intersection_filtered_points_publisher, self.bridge, self)
			self.best_theta = radial_scans(pts_xy, 'straight', self.yaw, self.turn_start_yaw, ANGLE_TOLERANCE, debug_stuff)



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
				self.stage = '3. radial scan'
				self.get_logger().info("✅ Stage 1. straight completed.") 

		elif self.stage == '3. radial scan' and (self.best_theta is not None):
			ang_z_required = max(-LEFT_TURN_ANGULAR_SPEED, min(LEFT_TURN_ANGULAR_SPEED, 1.5 * self.best_theta))
			self.linx_angz_to_publish = (LINEAR_SPEED_WHEN_RADIAL_SCAN_TURNING, ang_z_required)
			if abs(self.best_theta) < np.deg2rad(3.0):
				self.best_theta = None
				self.align_target_yaw = None
				self.linx_angz_to_publish = (LINEAR_SPEED, 0)

		# Compute Straight Displacement
		dx = x - self.start_x_y[0]
		dy = y - self.start_x_y[1]
		forward_disp = dx * np.cos(self.turn_start_yaw) + dy * np.sin(self.turn_start_yaw)

		if DEBUG:
			self.get_logger().info(f"📏 Forward displacement: {forward_disp:.2f} m")

		if forward_disp >= TARGET_FORWARD_DISPLACEMENT:
			self.get_logger().info("✅ Intersection Straight Drive Complete. Publishing 'none' into /intersection")
			msg = String()
			if self.qualification: msg.data = "follow_barrel_and_stop"
			else: msg.data = "none"
			self.intersection_pub.publish(msg)
			self.stage = '0. waiting for /intersection'
			self.start_x_y = None
			self.turn_start_yaw = None
			self.best_theta = None
			self.linx_angz_to_publish = None

	def publish_cmd(self):
		if self.stage == '0. waiting for /intersection':
			return

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
	node = IntersectionStraightDriver()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info("Shutting down…")
	finally:
		
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
