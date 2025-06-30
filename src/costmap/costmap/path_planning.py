import rclpy
from rclpy.node import Node
import numpy as np
import heapq

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
from std_msgs.msg import Header

class PathPlanner(Node):
	def __init__(self):
		super().__init__("path_planner_node")
		self.costmap_sub = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)
		self.line_marker_sub = self.create_subscription(MarkerArray, '/lane_visualization', self.marker_cb, 10)
		self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.costmap = None
		self.robot_pose = None
		self.goal_point = None
		
	def costmap_cb(self, msg):
		self.get_logger().info("received costmap")
		self.costmap = msg
		self.origin_x = msg.info.origin.position.x
		self.origin_y = msg.info.origin.position.y
		self.resolution = msg.info.resolution
		self.width = msg.info.width
		self.height = msg.info.height


	def odom_cb(self, msg):
		self.robot_pose = msg.pose.pose

	def marker_cb(self, msg):
		self.goal_point = self.estimate_goal_from_markers(msg)

	def estimate_goal_from_markers(self, marker_array):
		lane_markers = [m for m in marker_array.markers if m.ns == "lane_curves" and m.type == Marker.LINE_STRIP]

		if len(lane_markers) < 2:
			self.get_logger().warn("Not enough lane markers to estimate goal")
			return None

		right_marker = max(lane_markers, key=lambda m: m.points[0].y)
		points = right_marker.points

		if len(points) < 5:
			self.get_logger().warn("Too few points in right lane marker")
			return None

		# Use last few points to compute average goal
		N = 5
		tail = points[-N:]
		avg_x = sum(p.x for p in tail) / N
		avg_y = sum(p.y for p in tail) / N

		# Slight inward offset from the line
		goal_cam_x = avg_x
		goal_cam_y = avg_y - 0.3
		goal_cam_z = 0.0

		# Transform to odom
		result = self.transform_to_odom(goal_cam_x, goal_cam_y, goal_cam_z, frame_id=right_marker.header.frame_id)
		if result is None:
			self.get_logger().warn("Failed to transform goal to odom frame")
			return None

		x_odom, y_odom, z_odom = result

		goal_pose = PoseStamped()
		goal_pose.header.frame_id = 'odom'
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.pose.position.x = x_odom
		goal_pose.pose.position.y = y_odom
		goal_pose.pose.position.z = z_odom
		goal_pose.pose.orientation.w = 1.0

		self.get_logger().info(f"Goal (odom): ({x_odom:.2f}, {y_odom:.2f})")
		return goal_pose

	
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
		
	def world_to_map(self, x, y):
		mx = int((x - self.origin_x) / self.resolution)
		my = int((y - self.origin_y) / self.resolution)
		if 0 <= mx < self.width and 0 <= my < self.height:
			return (mx, my)
		return None

	def map_to_world(self, mx, my):
		x = mx * self.resolution + self.origin_x + self.resolution / 2
		y = my * self.resolution + self.origin_y + self.resolution / 2
		return (x, y)
	
def main(args=None):
	rclpy.init(args=args)
	node = PathPlanner()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
