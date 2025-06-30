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
import tf_transformations

class CostmapNode(Node):
	def __init__(self):
		super().__init__('costmap_node')

		# Parameters
		self.resolution = 0.05  # meters per cell
		self.width = 500
		self.height = 500
		self.origin_x = 0
		self.origin_y = 0
		self.frame_id = 'odom'  # match your PC frame

		# Costmap publisher
		self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap', 10)

		# Subscribe to your point cloud topic
		self.object_pc_sub = self.create_subscription(PointCloud2, '/object_pc', self.object_pc_callback, 10)
		self.lane_pc_sub = self.create_subscription(PointCloud2, '/lane_cluster', self.lane_pc_callback, 10)
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.lane_map = None
		self.object_map = None


	def object_pc_callback(self, msg):
		self.generate_costmap(msg, "object")

	def lane_pc_callback(self, msg):
		self.generate_costmap(msg, "lane")

	def generate_costmap(self, msg, tag = "lane"):
		try:
			tf = self.tf_buffer.lookup_transform('camera_link', 'odom', rclpy.time.Time())
			robot_x = tf.transform.translation.x
			robot_y = tf.transform.translation.y

			# center map around robot
			self.origin_x = robot_x - (self.width * self.resolution) / 2
			self.origin_y = robot_y - (self.height * self.resolution) / 2

		except Exception as e:
			self.get_logger().warn(f"TF error: {e}")
			return
		costmap = np.zeros((self.height, self.width), dtype=np.uint8)

		points = point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)

		for i in points:
			# These are already [x, y, z] as per your rotated format
			result = self.transform_to_odom(i[0], i[1], i[2])
			if result is None:
				continue
			
			x, y, z = result
			if tag == "lane" and result[2] > .1:
				continue
			mx = int((x - self.origin_x) / self.resolution)
			my = int((y - self.origin_y) / self.resolution)

			if 0 <= mx < self.width and 0 <= my < self.height:
				if tag == "object":
					costmap[my, mx] = 125
				else:
					costmap[my, mx] = 250

		# Gradient inflation using Gaussian filter
		'''binary = (costmap == 125).astype(np.float32)
		gradient = gaussian_filter(binary, sigma=3)
		if tag =="object":
			scaled = ((gradient / gradient.max()**2) * 100).astype(np.uint8)
		else: scaled = ((gradient / gradient.max()) * 100).astype(np.uint8)
		costmap = np.maximum(costmap, scaled)# Gradient inflation'''
		binary = costmap.astype(np.float32)
		gradient = gaussian_filter(binary, sigma=3)

		# Avoid division by 0
		if gradient.max() > 0:
			if tag == "object":
				scaled = ((gradient / (gradient.max() ** 2)) * 100).astype(np.uint8)
			else:  # lane
				scaled = ((gradient / gradient.max()) * 100).astype(np.uint8)
		else:
			scaled = np.zeros_like(costmap)

		# Combine with existing costmap
		costmap = np.maximum(costmap, scaled)


		if tag == "lane":
			self.lane_map = costmap
		elif tag == "object":
			self.object_map = costmap

		if self.lane_map is not None and self.object_map is not None:
			combined = np.maximum(self.lane_map, self.object_map)
			self.publish_costmap(combined, msg.header.stamp)
			# Optional: reset after publishing if you want to avoid stale maps
			self.lane_map = None
			self.object_map = None


	

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

			# Use timeout to ensure TF is ready
			if self.tf_buffer.can_transform('odom', frame_id, rclpy.time.Time()):
				transform = self.tf_buffer.lookup_transform(
					'odom',
					frame_id,
					rclpy.time.Time(),  # latest available
					#timeout=rclpy.duration.Duration(seconds=1.0)
				)
			else:    
				self.get_logger().warn(f"Transform from {frame_id} to odom not yet available")
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
