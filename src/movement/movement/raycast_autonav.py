#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math
from collections import deque
from builtin_interfaces.msg import Time

class RaycastNavigator(Node):
    def __init__(self):
        super().__init__('raycast')

        # -------------------
        # State
        # -------------------
        self.white_points = []  # (x, y) in odom frame
        self.obstacle_points = []  # (x, y) relative to robot
        self.robot_pose = None

        # Smoothing
        self.angle_history = deque(maxlen=5)
        self.target_angle = 0.0
        self.angle_smoothing_factor = 0.3
        self.min_angle_change = 0.01

        # Parameters
        self.raycast_range = 5.0       # meters
        self.raycast_step_deg = 1.0
        self.raycast_angle_min_deg = -90
        self.raycast_angle_max_deg = 90
        self.linear_speed = 0.8        # meters/sec
        self.max_angular_vel = 0.5

        # Dire situation
        self.danger_cost = 5.0
        self.front_angle_range = (-30, 30)  # degrees

        # -------------------
        # Subscribers
        # -------------------
        self.create_subscription(PointCloud2, '/white_lane_points', self.white_callback, 10)
        self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)

        # -------------------
        # Publishers
        # -------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.ray_marker_pub = self.create_publisher(MarkerArray, '/raycast_costs', 10)

        # Timer
        self.timer = self.create_timer(0.2, self.navigate)

    # -------------------
    # Callbacks
    # -------------------
    def white_callback(self, msg: PointCloud2):
        self.white_points = np.array([[p[0], p[1]] for p in pc2.read_points(msg, skip_nans=True)])

    def costmap_cb(self, msg: OccupancyGrid):
        res = msg.info.resolution
        origin = msg.info.origin.position
        w = msg.info.width
        h = msg.info.height

        self.obstacle_points.clear()

        for gy in range(h):
            for gx in range(w):
                idx = gy * w + gx
                if msg.data[idx] > 50:  # occupied cell
                    wx = gx * res + origin.x + res / 2.0
                    wy = gy * res + origin.y + res / 2.0
                    # relative to robot at origin
                    self.obstacle_points.append((wx, wy))

    # -------------------
    # Navigation
    # -------------------
    def navigate(self):
        if len(self.white_points) == 0 and len(self.obstacle_points) == 0:
            self.get_logger().warn("No data yet, skipping navigation")
            return

        best_cost = float('inf')
        best_angle = None
        ray_costs = []  # store all rays for dire situation check
        marker_array = MarkerArray()
        marker_id = 0

        for deg in np.arange(self.raycast_angle_min_deg, self.raycast_angle_max_deg + self.raycast_step_deg, self.raycast_step_deg):
            theta = math.radians(deg)
            ray_vec = np.array([math.cos(theta), math.sin(theta)]) * self.raycast_range
            cost = self.compute_ray_cost(ray_vec, theta)
            ray_costs.append((deg, cost))

            if cost < best_cost:
                best_cost = cost
                best_angle = theta

            # Visualization
            ray_marker = Marker()
            ray_marker.header.frame_id = "base_link"
            ray_marker.header.stamp = Time(sec=0, nanosec=0)
            ray_marker.ns = "raycast_costs"
            ray_marker.id = marker_id
            ray_marker.type = Marker.ARROW
            ray_marker.action = Marker.ADD
            ray_marker.scale.x = 0.05
            ray_marker.scale.y = 0.02
            ray_marker.scale.z = 0.02
            ray_marker.color.r = float(min(1.0, cost / 5.0))
            ray_marker.color.g = float(min(1.0, 1.0 - cost / 5.0))
            ray_marker.color.b = 0.0
            ray_marker.color.a = 1.0
            viz_scale = 0.5  # 50% of actual length
            viz_vec = ray_vec * viz_scale
            ray_marker.points = [Point(), Point(x=float(viz_vec[0]), y=float(viz_vec[1]), z=0.0)]
            marker_array.markers.append(ray_marker)
            marker_id += 1

        self.ray_marker_pub.publish(marker_array)

        # -------------------
        # Dire situation check
        # -------------------
        front_costs = [cost for deg, cost in ray_costs if self.front_angle_range[0] <= deg <= self.front_angle_range[1]]
        if front_costs and min(front_costs) > self.danger_cost:
            self.handle_dire_situation(ray_costs)
            return

        # -------------------
        # Normal operation
        # -------------------
        if best_angle is not None:
            self.angle_history.append(best_angle)
            smoothed_angle = sum(self.angle_history) / len(self.angle_history)
            self.update_target_angle(smoothed_angle)
            self.move_in_direction()
            self.get_logger().info(f"Best angle: {math.degrees(best_angle):.1f}°, Cost: {best_cost:.2f}")

    # -------------------
    # Dire situation handler
    # -------------------
    def handle_dire_situation(self, ray_costs):
        # Stop linear motion
        twist = Twist()
        twist.linear.x = 0.0

        # Find the safest ray (lowest cost overall)
        safest_deg, _ = min(ray_costs, key=lambda rc: rc[1])
        safest_angle = math.radians(safest_deg)

        # Rotate sharply toward it
        angle_diff = safest_angle - self.target_angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        twist.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, angle_diff))
        self.cmd_pub.publish(twist)
        self.get_logger().warn(f"Dire situation! Rotating toward {safest_deg:.1f}°")

    # -------------------
    # Helper functions
    # -------------------
    def compute_ray_cost(self, ray_vec, theta):
        ray_unit = ray_vec / np.linalg.norm(ray_vec)
        ray_length = np.linalg.norm(ray_vec)

        def min_distance_to(points):
            if len(points) == 0:
                return float('inf')
            points_np = np.array(points)
            proj_lens = np.dot(points_np, ray_unit)
            valid_mask = (proj_lens >= 0) & (proj_lens <= ray_length)
            if not np.any(valid_mask):
                return float('inf')
            proj_points = np.outer(proj_lens[valid_mask], ray_unit)
            distances = np.linalg.norm(points_np[valid_mask] - proj_points, axis=1)
            return np.min(distances)

        d_white = min_distance_to(self.white_points)
        d_obstacle = min_distance_to(self.obstacle_points)

        lane_penalty = 1.0 / (d_white + 1e-2)
        obstacle_penalty = 1.2 / max(d_obstacle, 0.5)

        total_cost = 0.6 * lane_penalty + 1.2 * obstacle_penalty
        return total_cost

    def update_target_angle(self, new_angle):
        angle_diff = new_angle - self.target_angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        angle_diff = np.clip(angle_diff, -math.radians(10), math.radians(10))
        if abs(angle_diff) > self.min_angle_change:
            self.target_angle += angle_diff * self.angle_smoothing_factor
            self.target_angle = math.atan2(math.sin(self.target_angle), math.cos(self.target_angle))

    def move_in_direction(self):
        twist = Twist()
        twist.linear.x = self.linear_speed
        angular_gain = 0.8
        twist.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, self.target_angle * angular_gain))
        self.cmd_pub.publish(twist)

# -------------------
# Main
# -------------------
def main(args=None):
    rclpy.init(args=args)
    node = RaycastNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
