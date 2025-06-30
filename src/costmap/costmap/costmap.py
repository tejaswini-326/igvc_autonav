import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from scipy.ndimage import gaussian_filter
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped


class CostmapNode(Node):
	def __init__(self):
		super().__init__('costmap_node')

		# Parameters
		self.resolution = 0.05  # meters per cell
		self.width = 500
		self.height = 500
		self.origin_x = 0
		self.origin_y = 0
		self.frame_id = 'odom'

		# Costmap publisher
		self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap', 10)

		# Subscriptions
		self.object_pc_sub = self.create_subscription(PointCloud2, '/object_pc', self.object_pc_callback, 10)
		self.wlane_pc_sub = self.create_subscription(PointCloud2, '/white_lane_points', self.white_lane_pc_callback, 10)
		self.ylane_pc_sub = self.create_subscription(PointCloud2, '/yellow_lane_points', self.yellow_lane_pc_callback, 10)

		# TF listener
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# Internal maps
		self.white_map = None
		self.yellow_map = None
		self.object_map = None

	def object_pc_callback(self, msg):
		self.generate_costmap(msg, tag="object")

	def white_lane_pc_callback(self, msg):
		self.generate_costmap(msg, tag="white")

	def yellow_lane_pc_callback(self, msg):
		self.generate_costmap(msg, tag="yellow")

	def generate_costmap(self, msg, tag="lane"):
		try:
			tf = self.tf_buffer.lookup_transform('odom', 'camera_link', rclpy.time.Time())
			robot_x = tf.transform.translation.x
			robot_y = tf.transform.translation.y

			self.origin_x = robot_x - (self.width * self.resolution) / 2
			self.origin_y = robot_y - (self.height * self.resolution) / 2

		except Exception as e:
			self.get_logger().warn(f"TF error: {e}")
			return

		costmap = np.zeros((self.height, self.width), dtype=np.uint8)
		points = point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)

		for i in points:
			result = self.transform_to_odom(i[0], i[1], i[2])
			if result is None:
				continue
			x, y, z = result

			if tag in ["lane", "white", "yellow"] and z > 0.1:
				continue

			mx = int((x - self.origin_x) / self.resolution)
			my = int((y - self.origin_y) / self.resolution)

			if 0 <= mx < self.width and 0 <= my < self.height:
				if tag == "object":
					costmap[my, mx] = 100
				elif tag == "white":
					costmap[my, mx] = 250
				elif tag == "yellow":
					costmap[my, mx] = 200

		# Gaussian inflation
		binary = costmap.astype(np.float32)
		gradient = gaussian_filter(binary, sigma=3)

		if gradient.max() > 0:
			if tag == "object":
				scaled = ((gradient / (gradient.max() ** 2)) * 100).astype(np.uint8)
			else:
				scaled = ((gradient / gradient.max()) * 100).astype(np.uint8)
		else:
			scaled = np.zeros_like(costmap)

		costmap = np.maximum(costmap, scaled)

		# Store to appropriate internal map
		if tag == "white":
			self.white_map = costmap
		elif tag == "yellow":
			self.yellow_map = costmap
		elif tag == "object":
			self.object_map = costmap

		# Publish only when all layers are available
				# Fill missing maps with zeros
		white = self.white_map if self.white_map is not None else np.zeros((self.height, self.width), dtype=np.uint8)
		yellow = self.yellow_map if self.yellow_map is not None else np.zeros((self.height, self.width), dtype=np.uint8)
		objects = self.object_map if self.object_map is not None else np.zeros((self.height, self.width), dtype=np.uint8)

		# Merge and publish
		combined = np.maximum.reduce([white, yellow, objects])
		self.publish_costmap(combined, msg.header.stamp)

		#clears 
		'''if tag == "white":
			self.white_map = None
		elif tag == "yellow":
			self.yellow_map = None
		elif tag == "object":
			self.object_map = None'''


	def publish_costmap(self, costmap, stamp):
		msg = OccupancyGrid()
		msg.header = Header()
		msg.header.stamp = stamp
		msg.header.frame_id = self.frame_id

		msg.info.resolution = self.resolution
		msg.info.width = self.width
		msg.info.height = self.height
		msg.info.origin.position.x = self.origin_x
		msg.info.origin.position.y = self.origin_y
		msg.info.origin.orientation.w = 1.0

		msg.data = [int(v / 255 * 100) for v in costmap.flatten()]
		self.costmap_pub.publish(msg)

	def transform_to_odom(self, x, y, z, frame_id='camera_link'):
		try:
			point = PointStamped()
			point.header.frame_id = frame_id
			point.header.stamp = self.get_clock().now().to_msg()
			point.point.x = x
			point.point.y = y
			point.point.z = z

			if self.tf_buffer.can_transform('odom', frame_id, rclpy.time.Time()):
				transform = self.tf_buffer.lookup_transform(
					'odom',
					frame_id,
					rclpy.time.Time(),
				)
			else:
				self.get_logger().warn(f"Transform from {frame_id} to odom not available")
				return None

			transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
			return (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)

		except Exception as e:
			self.get_logger().warn(f"Transform failed: {e}")
			return None
	


def main(args=None):
	rclpy.init(args=args)
	node = CostmapNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
