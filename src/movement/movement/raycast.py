import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from collections import deque


class RaycastNavigator(Node):
    def __init__(self):
        super().__init__('raycast_navigator')
        self.white_curve_points = []
        self.yellow_curve_points = []
        self.obstacle_points = []

        self.white_sub = self.create_subscription(
            MarkerArray,
            '/lane_fitted_white',
            self.white_callback,
            10
        )
        self.yellow_sub = self.create_subscription(
            MarkerArray,
            '/lane_fitted_yellow',
            self.yellow_callback,
            10
        )
        self.obstacle_sub = self.create_subscription(
            PointCloud2,
            'igvc/midz_points',
            self.obstacle_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cost_marker_pub = self.create_publisher(MarkerArray, '/raycast_costs', 10)

        self.object_detected = False
        self.obstacle_distance_threshold = 3.0  # meters

        # For smoothing
        self.angle_history = deque(maxlen=5)
        self.target_angle = 0.0
        self.angle_smoothing_factor = 0.3
        self.min_angle_change = 0.1

        # Navigation frequency
        self.navigation_interval = 0.3  # seconds
        self.declare_parameter('raycast_range', 3.0)
        self.declare_parameter('raycast_step_deg', 0.5)
        self.declare_parameter('raycast_angle_min_deg', -90)
        self.declare_parameter('raycast_angle_max_deg', 90)

        # Timer for periodic navigation
        self.timer = self.create_timer(self.navigation_interval, self.navigate)

    def white_callback(self, msg):
        self.white_curve_points = []
        for marker in msg.markers:
            if marker.type == Marker.LINE_STRIP:
                for p in marker.points:
                    self.white_curve_points.append((p.x, p.y))

    def yellow_callback(self, msg):
        self.yellow_curve_points = []
        if not self.object_detected:
            for marker in msg.markers:
                if marker.type == Marker.LINE_STRIP:
                    for p in marker.points:
                        self.yellow_curve_points.append((p.x, p.y))

    def obstacle_callback(self, msg):
        self.obstacle_points.clear()
        self.object_detected = False

        for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            x, y, z = point
            distance = math.sqrt(x**2 + y**2 + z**2)
            if distance < self.obstacle_distance_threshold:
                self.obstacle_points.append((x, y))
                self.object_detected = True

    def navigate(self):
        raycast_range = self.get_parameter('raycast_range').value
        step_deg = self.get_parameter('raycast_step_deg').value
        min_angle = self.get_parameter('raycast_angle_min_deg').value
        max_angle = self.get_parameter('raycast_angle_max_deg').value

        # If no lane data is available, continue in previous direction
        if not self.white_curve_points and not self.yellow_curve_points:
            self.get_logger().warn("No lane markers detected. Continuing in previous direction.")
            self.update_target_angle(self.target_angle)  # keep angle unchanged
            self.move_in_direction()
            return

        best_cost = float('inf')
        best_angle = None
        marker_array = MarkerArray()
        marker_id = 0

        for deg in np.arange(min_angle, max_angle + step_deg, step_deg):
            theta = math.radians(deg)
            ray_vec = np.array([math.cos(theta), math.sin(theta)]) * raycast_range
            cost = self.compute_ray_cost(ray_vec, theta)

            if cost < best_cost:
                best_cost = cost
                best_angle = theta

            # Visualization
            ray_marker = Marker()
            ray_marker.header.frame_id = "base_link"
            ray_marker.header.stamp = self.get_clock().now().to_msg()
            ray_marker.ns = "raycast_costs"
            ray_marker.id = marker_id
            ray_marker.type = Marker.ARROW
            ray_marker.action = Marker.ADD
            ray_marker.scale.x = 0.1
            ray_marker.scale.y = 0.05
            ray_marker.scale.z = 0.05

            ray_marker.color.r = float(min(1.0, cost / 5.0))
            ray_marker.color.g = float(min(1.0, 1.0 - cost / 5.0))
            ray_marker.color.b = 0.0
            ray_marker.color.a = 1.0

            ray_marker.points = [Point(), Point(x=float(ray_vec[0]), y=float(ray_vec[1]), z=0.0)]
            marker_array.markers.append(ray_marker)
            marker_id += 1

        self.cost_marker_pub.publish(marker_array)

        if best_angle is not None:
            if not self.object_detected:
                self.angle_history.append(best_angle)
                smoothed_angle = sum(self.angle_history) / len(self.angle_history)
            else:
                smoothed_angle = best_angle
            self.update_target_angle(smoothed_angle)
            self.move_in_direction()
        else:
            self.get_logger().warn("No valid ray direction found")

    def update_target_angle(self, new_angle):
        angle_diff = new_angle - self.target_angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        max_delta = math.radians(10)  # max change per step
        angle_diff = np.clip(angle_diff, -max_delta, max_delta)

        if abs(angle_diff) > self.min_angle_change:
            self.target_angle += angle_diff * self.angle_smoothing_factor
            self.target_angle = math.atan2(math.sin(self.target_angle), math.cos(self.target_angle))

    def compute_ray_cost(self, ray_vec, theta):
        ray_unit = ray_vec / np.linalg.norm(ray_vec)
        ray_length = np.linalg.norm(ray_vec)

        def min_distance_to(points):
            if not points:
                return float('inf')
            points_np = np.array(points)
            proj_lens = np.dot(points_np, ray_unit)
            valid_mask = (proj_lens >= 0) & (proj_lens <= ray_length)
            if not np.any(valid_mask):
                return float('inf')
            proj_points = np.outer(proj_lens[valid_mask], ray_unit)
            distances = np.linalg.norm(points_np[valid_mask] - proj_points, axis=1)
            return np.min(distances)

        d_white = min_distance_to(self.white_curve_points)
        d_yellow = min_distance_to(self.yellow_curve_points)
        d_obstacle = min_distance_to(self.obstacle_points)

        lane_center_penalty = 1.0 / (min(d_white, d_yellow) + 1e-2)
        obstacle_penalty = 1.0 / (d_obstacle + 1e-2) if d_obstacle != float('inf') else 0.0

        forward_bias = 1.0

        total_cost = 0.6 * lane_center_penalty + 1.2 * obstacle_penalty
        return total_cost * forward_bias

    def move_in_direction(self):
        twist = Twist()
        twist.linear.x = 0.5
        max_angular_vel = 0.5
        angular_gain = 0.8
        twist.angular.z = max(-max_angular_vel, min(max_angular_vel, self.target_angle * angular_gain))
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Target angle: {math.degrees(self.target_angle):.1f}°, Angular vel: {twist.angular.z:.3f}")
        #pass


def main(args=None):
    rclpy.init(args=args)
    node = RaycastNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
