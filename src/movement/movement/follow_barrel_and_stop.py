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
from object_detection.msg import ObjectData
from geometry_msgs.msg import PointStamped
import tf2_ros, tf2_geometry_msgs
from rclpy.duration import Duration


LINEAR_SPEED_WHEN_RADIAL_SCAN_TURNING           = 0.5
LEFT_TURN_ANGULAR_SPEED                         = 0.22   

# Needs to be changed in the real bot
DISTANCE_SQUARED_BEFORE_STOPPING_AT_BARREL = 6



class FollowBarrelAndStop(Node):
	def __init__(self):
		super().__init__("follow_barrel_and_stop")
		self.create_subscription(String, '/intersection', self.intersection_cb, 10) 
		self.create_subscription(Odometry, "/odom", self.odom_cb, 10)   
		self.create_subscription(ObjectData, 'object_data', self.yolo_object_data_cb, 10)

		self.cmd_vel_publisher = self.create_publisher(Twist, "cmd_vel", 10)
		self.intersection_pub = self.create_publisher(String, '/intersection', 10)
		self.object_position_label = None

		self.active = False
		self.stopping  = False
		self.linx_angz_to_publish = None

		self.timer = self.create_timer(0.05, self.publish_cmd)

		self.tf_buffer   = tf2_ros.Buffer(cache_time=Duration(seconds=10))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

	def transform_to_odom(self, x, y, z, frame_id="camera_link"):
		try:
			point = PointStamped()
			point.header.frame_id = frame_id
			point.header.stamp    = self.get_clock().now().to_msg()
			point.point.x, point.point.y, point.point.z = x, y, z

			# Wait up to 1 s for TF to become available
			tf_ok = self.tf_buffer.can_transform(
				"base_link", frame_id, rclpy.time.Time(),
				timeout=Duration(seconds=1.0)
			)
			if not tf_ok:
				self.get_logger().debug(f"TF {frame_id}base_link not ready")
				return None

			transform = self.tf_buffer.lookup_transform(
				"base_link", frame_id, rclpy.time.Time())
			base_link_point = tf2_geometry_msgs.do_transform_point(point, transform)
			return base_link_point.point.x, base_link_point.point.y, base_link_point.point.z

		except Exception as e:
			self.get_logger().warn(f"Transform failure: {e}")
			return None

	def intersection_cb(self, msg: String):
		if msg.data.lower() == "follow_barrel_and_stop":
			self.active = True
			self.get_logger().info("🟢 Received 'follow_barrel_and_stop' from /intersection.")
		else:
			self.active = False
			self.get_logger().info(f"🛑 Ignoring '{msg.data}' from /intersection.")


	def yolo_object_data_cb(self, msg:ObjectData):
		if not self.active:
			return
		self.object_position_label = (msg.position, msg.label)

	def odom_cb(self, msg: Odometry):
		if not self.active:
			return

		if self.object_position_label:
			raw_pos, label = self.object_position_label
			base_link_pos = self.transform_to_odom(raw_pos.x, raw_pos.y, raw_pos.z, frame_id="camera_link")

			if base_link_pos is None:
				return
			
			a, b, c = base_link_pos
			self.get_logger().info(f"{a**2 + b**2} and {DISTANCE_SQUARED_BEFORE_STOPPING_AT_BARREL}")
			if a**2 + b**2 < DISTANCE_SQUARED_BEFORE_STOPPING_AT_BARREL:
				self.stopping = True
			else:
				self.linx_angz_to_publish = LINEAR_SPEED_WHEN_RADIAL_SCAN_TURNING, LEFT_TURN_ANGULAR_SPEED*b


		if self.stopping and msg.twist.twist.linear.x == 0 and msg.twist.twist.linear.y == 0 and msg.twist.twist.angular.z == 0:
			self.active = False
			self.stopping = False
			intersection_msg = String()
			intersection_msg.data = 'stopped_at_barrel'
			self.get_logger().info(f"📢 Publishing On Intersection Topic: Msg Data: {intersection_msg.data}")
			self.intersection_pub.publish(intersection_msg)



	def publish_cmd(self):
		if not self.active:
			return
		
		if self.stopping:
			cmd = Twist()
			cmd.linear.x = 0.0
			cmd.angular.z = 0.0
			self.cmd_vel_publisher.publish(cmd)
		elif self.linx_angz_to_publish:
			cmd = Twist()
			cmd.linear.x = float(self.linx_angz_to_publish[0])
			cmd.angular.z = float(self.linx_angz_to_publish[1])
			self.cmd_vel_publisher.publish(cmd)

		

def main(args=None):
	rclpy.init(args=args)
	node = FollowBarrelAndStop()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info("Shutting down…")
	finally:
		
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
