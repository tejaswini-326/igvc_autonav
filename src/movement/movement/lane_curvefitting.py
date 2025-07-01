import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
from sklearn.cluster import DBSCAN
import math
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String, ColorRGBA
from nav_msgs.msg import Odometry

MIN_CLUSTERING_DISTANCE = 0.971
MIN_CLUSTERING_POINTS = 20
WHITE_THRESHOLD = 100
COLOR_BALANCE_THRESHOLD = 25
# Yellow detection thresholds
YELLOW_R_MIN = 90
YELLOW_G_MIN = 90
YELLOW_B_MAX = 150
YELLOW_BALANCE_THRESHOLD = 0
ANGLE_FACTOR = 1.1

class LaneFollowerNode(Node):
	def __init__(self):
		super().__init__('lane_follower_node')
		self.subscription = self.create_subscription(
			PointCloud2,
			'/camera/points',
			self.pointcloud_callback,
			10
		)
		self.marker_pub = self.create_publisher(Marker, '/lane_marker', 10)
		self.markers_pub = self.create_publisher(MarkerArray, '/lane_visualization', 10)
		self.white_pub = self.create_publisher(PointCloud2, "/white_lane_points", 10)
		self.yellow_pub = self.create_publisher(PointCloud2, "/yellow_lane_points", 10)    

	# FUNCTION TO PUBLISH STUFF FOR LANE VISUALISATION IN RVIZ
	def publish_lane_visualization(self, msg, target_point, cluster_curves, white_ground_points, yellow_ground_points):
		marker_array = MarkerArray()
		
		# Clear previous markers
		clear_marker = Marker()
		clear_marker.header.frame_id = msg.header.frame_id
		clear_marker.header.stamp = self.get_clock().now().to_msg()
		clear_marker.action = Marker.DELETEALL
		marker_array.markers.append(clear_marker)
		
		# Marker for white ground points (detected lane pixels)
		if white_ground_points:
			points_marker = Marker()
			points_marker.header.frame_id = msg.header.frame_id
			points_marker.header.stamp = self.get_clock().now().to_msg()
			points_marker.ns = "white_points"
			points_marker.id = 0
			points_marker.type = Marker.POINTS
			points_marker.action = Marker.ADD
			points_marker.scale.x = 0.02
			points_marker.scale.y = 0.02
			points_marker.color.r = 1.0
			points_marker.color.g = 1.0
			points_marker.color.b = 1.0
			points_marker.color.a = 0.6
			
			for point in white_ground_points:
				pt = Point()
				pt.x = float(point[0])
				pt.y = float(point[1])
				pt.z = float(point[2])
				points_marker.points.append(pt)
			
			marker_array.markers.append(points_marker)
		
		# Marker for yellow ground points (detected lane pixels)
		if yellow_ground_points:
			yellow_points_marker = Marker()
			yellow_points_marker.header.frame_id = msg.header.frame_id
			yellow_points_marker.header.stamp = self.get_clock().now().to_msg()
			yellow_points_marker.ns = "yellow_points"
			yellow_points_marker.id = 1
			yellow_points_marker.type = Marker.POINTS
			yellow_points_marker.action = Marker.ADD
			yellow_points_marker.scale.x = 0.02
			yellow_points_marker.scale.y = 0.02
			yellow_points_marker.color.r = 1.0
			yellow_points_marker.color.g = 1.0
			yellow_points_marker.color.b = 0.0
			yellow_points_marker.color.a = 0.8
			
			for point in yellow_ground_points:
				pt = Point()
				pt.x = float(point[0])
				pt.y = float(point[1])
				pt.z = float(point[2])
				yellow_points_marker.points.append(pt)
			
			marker_array.markers.append(yellow_points_marker)
		
		# Markers for fitted curves
		for i, (label, coeffs, color_type) in enumerate(cluster_curves):
			curve_marker = Marker()
			curve_marker.header.frame_id = msg.header.frame_id
			curve_marker.header.stamp = self.get_clock().now().to_msg()
			curve_marker.ns = "lane_curves"
			curve_marker.id = i + 2
			curve_marker.type = Marker.LINE_STRIP
			curve_marker.action = Marker.ADD
			curve_marker.scale.x = 0.05
			
			# Different colors for different curves and types
			if color_type == 'white':
				if i == 0:
					curve_marker.color.r = 1.0
					curve_marker.color.g = 0.0
					curve_marker.color.b = 0.0
				elif i == 1:
					curve_marker.color.r = 0.0
					curve_marker.color.g = 1.0
					curve_marker.color.b = 0.0
				else:
					curve_marker.color.r = 0.0
					curve_marker.color.g = 0.0
					curve_marker.color.b = 1.0
			else:  # yellow
				if i == 0:
					curve_marker.color.r = 1.0
					curve_marker.color.g = 1.0
					curve_marker.color.b = 0.0
				elif i == 1:
					curve_marker.color.r = 1.0
					curve_marker.color.g = 0.5
					curve_marker.color.b = 0.0
				else:
					curve_marker.color.r = 0.8
					curve_marker.color.g = 0.8
					curve_marker.color.b = 0.0
			
			curve_marker.color.a = 1.0
			
			# Generate curve points
			x_line = np.linspace(0.0, 4.0, 50)
			a, b, c = coeffs
			
			for x_val in x_line:
				y_val = a * x_val**2 + b * x_val + c  
				pt = Point()
				pt.x = float(x_val)
				pt.y = float(y_val)
				pt.z = -1.35  # Ground level
				curve_marker.points.append(pt)
			
			marker_array.markers.append(curve_marker)
		
		# Marker for target point
		if target_point is not None:
			target_marker = Marker()
			target_marker.header.frame_id = msg.header.frame_id
			target_marker.header.stamp = self.get_clock().now().to_msg()
			target_marker.ns = "target_point"
			target_marker.id = 100
			target_marker.type = Marker.SPHERE
			target_marker.action = Marker.ADD
			target_marker.scale.x = 0.2
			target_marker.scale.y = 0.2
			target_marker.scale.z = 0.2
			target_marker.color.r = 1.0
			target_marker.color.g = 1.0
			target_marker.color.b = 0.0
			target_marker.color.a = 1.0
			
			target_marker.pose.position.x = float(target_point[0])
			target_marker.pose.position.y = float(target_point[1])
			target_marker.pose.position.z = -1.3
			target_marker.pose.orientation.w = 1.0
			
			marker_array.markers.append(target_marker)
		
		# Publish marker array
		self.markers_pub.publish(marker_array)

	def pointcloud_callback(self, msg):
		height = msg.height
		width = msg.width

		white_ground_points = []
		yellow_ground_points = []

		index = 0
		for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
			x, y, z, rgb = point
			row = index // width
			col = index % width

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

			# White detection
			white_threshold = WHITE_THRESHOLD
			color_balance_threshold = COLOR_BALANCE_THRESHOLD
			avg_color = (r + g + b) / 3
			if (r > white_threshold and g > white_threshold and b > white_threshold and
				abs(r - avg_color) < color_balance_threshold and 
				abs(g - avg_color) < color_balance_threshold and 
				abs(b - avg_color) < color_balance_threshold):

				# Ground level filtering
				if -2.0 < z < -1.3 and 0.0 < x < 10.0:
					white_ground_points.append([x, y, z])
			
			# Yellow detection (high red and green, low blue)
			if (r > YELLOW_R_MIN and g > YELLOW_G_MIN and b < YELLOW_B_MAX and not (r > white_threshold and g > white_threshold and b > white_threshold and
				abs(r - avg_color) < color_balance_threshold and 
				abs(g - avg_color) < color_balance_threshold and 
				abs(b - avg_color) < color_balance_threshold)):
				# Ground level filtering
				if -2.0 < z < -1.3 and 0.0 < x < 10.0:
					yellow_ground_points.append([x, y, z])
			
			index += 1

		# Combine white and yellow points for clustering
		all_lane_points = white_ground_points + yellow_ground_points
		
		# Only do 3D clustering for lane following if we have enough points
		if len(all_lane_points) < 10:
			self.get_logger().warn(f"Not enough lane points for detection (white: {len(white_ground_points)}, yellow: {len(yellow_ground_points)})")
			# Still publish empty visualization
			self.publish_lane_visualization(msg, None, [], white_ground_points, yellow_ground_points)
			return

		points_np = np.array(all_lane_points)
		self.get_logger().warn(f"no of yellow ground points : {len(yellow_ground_points)}")
		points_np_y = np.array(yellow_ground_points)
		
		self.get_logger().warn(f" shape : {points_np_y.shape}")

		# Use only x,y coordinates for clustering (ignore z)
		points_xy = points_np[:, :2]  # Extract x,y coordinates
		clustering = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy)
		labels = clustering.labels_
		# WHITE
		clustered_white_points = points_np[labels != -1]
		self.get_logger().warn(f" white points being published : {len(clustered_white_points)}")
		if len(clustered_white_points) > 0:
			white_msg = pc2.create_cloud_xyz32(msg.header, clustered_white_points.tolist())
			self.white_pub.publish(white_msg)

		points_xy_y = points_np_y[:, :2]
		clustering_y = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy_y)
		labels_y = clustering_y.labels_
		clustered_yellow_points = points_np_y[labels_y != -1]
		self.get_logger().warn(f"yellow points being published : {len(clustered_yellow_points)}")
		if len(clustered_yellow_points) > 0:
			yellow_msg = pc2.create_cloud_xyz32(msg.header, clustered_yellow_points.tolist())
			self.yellow_pub.publish(yellow_msg)

		clustering = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy)
		labels = clustering.labels_
		
		# Count clusters and noise points
		unique_labels = set(labels)

		n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
		n_noise = list(labels).count(-1)
		
		# Process lane clusters
		cluster_curves = []

		for label in unique_labels:
			if label == -1:
				continue

			cluster_points = points_xy[labels == label]
			cluster_size = len(cluster_points)
			if cluster_size < 10:
				continue

			x_vals = cluster_points[:, 0]
			y_vals = cluster_points[:, 1]
			coeffs = np.polyfit(x_vals, y_vals, deg=2)
			
			# Determine if this cluster is primarily white or yellow based on original points
			cluster_indices = np.where(labels == label)[0]
			white_count = sum(1 for i in cluster_indices if i < len(white_ground_points))
			yellow_count = cluster_size - white_count
			color_type = 'white' if white_count >= yellow_count else 'yellow'
			cluster_curves.append((label, coeffs, color_type))

		self.publish_lane_visualization(msg, None, cluster_curves, white_ground_points, yellow_ground_points)

		

def main(args=None):
	rclpy.init(args=args)
	node = LaneFollowerNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()