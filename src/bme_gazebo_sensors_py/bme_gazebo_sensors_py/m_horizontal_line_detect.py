from math import hypot, pi, radians
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from math import radians, degrees
import struct
from std_msgs.msg import String, Float64MultiArray
import cv2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from collections import deque

from bme_gazebo_sensors_py.intersection_funcs import get_xy_of_all_white_and_yellow_points_from_pointcloud_msg, radial_scans, GRID_RES, normalise_angle

import matplotlib
matplotlib.use("Agg")                     # headless rendering in ROS 2 nodes
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image




# ──────────────────────────────────────────────────────────────────────────────
# Setting this to true will generate 3 new topics:
# intersection_lane_marker - Marker object showing direction chosen
# intersection_filtered_white - Pointcloud showing filtered white points in blue colour
# intersection_lane_scan_2d_debug - An image showing all polar scans and detected distances
DEBUG = True

# Lane Detection Related
MIN_NUMBER_OF_FILTERED_COLOURED_POINTS_REQUIRED = 0 

# Movement Related
LINEAR_SPEED                                    = 5.0                # m/s   (forward)
CMD_VEL_PUBLISHING_TIME_INTERVAL                = 0.1                # Time interval between 2 publishers in seconds
LEFT_TURN_ANGULAR_SPEED                         = 0.1                # rad/s (+ve = CCW = left)

THRESHOLD_DISTANCE_TO_BEGIN_DECELERATING_FOR_HORIZONTAL_WHITE_LINE = 3

# ──────────────────────────────────────────────────────────────────────────────



class M_HorizontalLineDetect(Node):
	def __init__(self):
		super().__init__("m_horizontal_line_detect")
		self.create_subscription(String, '/intersection', self.intersection_cb, 10) 
		self.create_subscription(PointCloud2, "/camera/points", self.pointcloud_cb, 10)
		self.create_subscription(Odometry, "/odom", self.odom_cb, 10)   
		self.create_subscription(Float64MultiArray, '/igvc/next_waypoint', self.next_waypoint_cb, 10)

		self.cmd_vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
		self.intersection_pub = self.create_publisher(String, '/intersection', 10)
		
		if DEBUG: 
			self.plot_pub = self.create_publisher(Image, "/debug/intersection/white_line_stop_dists_plot", 10)
			self._bridge = CvBridge()

		self.active = True
		self.actively_searching = False
		self.horizontal_white_line_within_threshold = False
		self.has_fully_stopped_and_horizontal_line_detected = False
		self.timer = self.create_timer(CMD_VEL_PUBLISHING_TIME_INTERVAL, self.publish_cmd)


	def intersection_cb(self, msg: String):
		if msg.data.lower() == "none":
			self.active = True
			self.actively_searching = True
			self.get_logger().info(f"🟢 Received 'none' from /intersection.")
		elif msg.data.lower() == 'stopping_for_horizontal_white_line':
			pass
		else:
			self.active = False
			self.actively_searching = False
			self.horizontal_white_line_within_threshold = False
			self.has_fully_stopped_and_horizontal_line_detected = False
			self.get_logger().info(f"🛑 Ignoring '{msg.data}' from /intersection.")


	def pointcloud_cb(self, msg: PointCloud2):
		if not self.actively_searching:
			return

		pts_xy = get_xy_of_all_white_and_yellow_points_from_pointcloud_msg(msg)

		if len(pts_xy) < MIN_NUMBER_OF_FILTERED_COLOURED_POINTS_REQUIRED:
			return
		
		dists = radial_scans(pts_xy, 'horizontal line detect', None, None, None, None)

		if DEBUG:
			# 1) Draw the plot into an in-memory buffer
			fig, ax = plt.subplots(figsize=(6, 3))
			ax.plot(list(reversed(dists)), lw=1.5)
			ax.set_title("Ray-wise distances")
			ax.set_xlabel("Ray index")
			ax.set_ylabel("Distance (px)")
			ax.grid(True, linestyle="--", alpha=0.4)

			buf = BytesIO()
			fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
			plt.close(fig)
			buf.seek(0)

			# 2) Decode PNG → OpenCV BGR image
			png_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
			cv_img    = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)   # shape (H,W,3)

			# 3) Convert to ROS 2 Image and publish
			img_msg = self._bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
			img_msg.header.stamp = self.get_clock().now().to_msg()
			self.plot_pub.publish(img_msg)

		if np.all(dists < THRESHOLD_DISTANCE_TO_BEGIN_DECELERATING_FOR_HORIZONTAL_WHITE_LINE / GRID_RES):
			self.horizontal_white_line_within_threshold = True



	def odom_cb(self, msg: Odometry):
		if not self.active:
			return
		
		t = msg.twist.twist
		if (self.horizontal_white_line_within_threshold and t.linear.x == 0 and t.linear.y == 0 and t.angular.z == 0):
			self.has_fully_stopped_and_horizontal_line_detected = True
		else:
			self.has_fully_stopped_and_horizontal_line_detected = False


	def publish_cmd(self):
		if not self.active:
			return
		
		if self.actively_searching and self.horizontal_white_line_within_threshold:
			self.actively_searching = False
			intersection_msg = String()
			intersection_msg.data = 'stopping_for_horizontal_white_line'
			self.get_logger().info(f"📢 Publishing On Intersection Topic: Msg Data: {intersection_msg.data}")
			self.intersection_pub.publish(intersection_msg)

		if self.horizontal_white_line_within_threshold:
			self.get_logger().info("Detected a horizontal white line and publishing zero velocity")
			cmd = Twist()
			cmd.linear.x = 0.0
			cmd.angular.z = 0.0
			self.cmd_vel_publisher.publish(cmd)

		if self.has_fully_stopped_and_horizontal_line_detected:
			self.get_logger().info(f"Has fully stopped Detected")
			intersection_msg = String()
			degees_to_next_waypoint_at_intersection = normalise_angle(degrees(self.next_waypoint['direction']))
			if degees_to_next_waypoint_at_intersection < -30:
				intersection_msg.data = 'left'
			elif degees_to_next_waypoint_at_intersection > 30:
				intersection_msg.data = 'right'
			else:
				intersection_msg.data = 'straight'

			self.get_logger().info(f"📢 Publishing On Intersection Topic: Msg Data: {intersection_msg.data}")
			self.intersection_pub.publish(intersection_msg)

			self.active = False
			self.horizontal_white_line_within_threshold = False
			self.has_fully_stopped_and_horizontal_line_detected = False
	
	def next_waypoint_cb(self, msg:Float64MultiArray):
		distance, heading_error, idx = msg.data
		self.next_waypoint = {'distance':distance, 'direction':heading_error, 'waypoint_idx':idx}

		

def main(args=None):
	rclpy.init(args=args)
	node = M_HorizontalLineDetect()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info("Shutting down…")
	finally:
		
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
