import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class RaycastNavigator(Node):
    def __init__(self):
        super().__init__('raycast_navigator')
        self.white_curve_points = []
        self.yellow_curve_points = []
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
            '/igvc/z_filtered',
            self.obstacle_callback,
            10
        )
        self.obstacle_points = []
        self.obstacle_distance_threshold = 0.5  # meters

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cost_marker_pub = self.create_publisher(MarkerArray, '/raycast_costs', 10)

        self.curve_points = []
        self.object_detected=True
        self.declare_parameter('raycast_range', 4.0)
        self.declare_parameter('raycast_step_deg', 3)
        self.declare_parameter('raycast_angle_min_deg', -90)
        self.declare_parameter('raycast_angle_max_deg', 90)

    
    def white_callback(self, msg):
        self.white_curve_points = []
        for marker in msg.markers:
            if marker.type == Marker.LINE_STRIP:
                for p in marker.points:
                    self.white_curve_points.append((p.x, p.y))
        self.navigate()

    def yellow_callback(self, msg):
        self.yellow_curve_points = []
        if not self.object_detected:
            for marker in msg.markers:
                if marker.type == Marker.LINE_STRIP:
                    for p in marker.points:
                        self.yellow_curve_points.append((p.x, p.y))
        self.navigate()

    def obstacle_callback(self, msg):
        self.obstacle_points.clear()
        self.object_detected = False  # reset

        for point in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            x, y, z = point
            distance = math.sqrt(x**2 + y**2 + z**2)
            if distance < self.obstacle_distance_threshold:
                self.obstacle_points.append((x, y))
                self.object_detected = True

        self.navigate() 

    def navigate(self):

        self.curve_points = self.white_curve_points + self.yellow_curve_points
        # Add obstacle points
        if self.obstacle_points:
            self.curve_points += self.obstacle_points
        if not self.curve_points:
            self.get_logger().warn("No curve points received")
            return

        raycast_range = self.get_parameter('raycast_range').value
        step_deg = self.get_parameter('raycast_step_deg').value
        min_angle = self.get_parameter('raycast_angle_min_deg').value
        max_angle = self.get_parameter('raycast_angle_max_deg').value

        best_cost = float('inf')
        best_angle = None

        marker_array = MarkerArray()
        marker_id = 0

        for deg in range(-90, 91, step_deg):  # from -90 to +90 degrees
            theta = math.radians(deg)
            ray_vec = np.array([math.cos(theta), math.sin(theta)]) * raycast_range

            cost = self.compute_ray_cost(ray_vec)

            # Log for debugging
            self.get_logger().info(f"Angle {deg}° -> Cost: {cost:.3f}")

            # Create a marker arrow
            ray_marker = Marker()
            ray_marker.header.frame_id = "base_link"
            ray_marker.header.stamp = self.get_clock().now().to_msg()
            ray_marker.ns = "raycast_costs"
            ray_marker.id = marker_id
            ray_marker.type = Marker.ARROW
            ray_marker.action = Marker.ADD
            ray_marker.scale.x = 0.1  # shaft length scaling
            ray_marker.scale.y = 0.05  # shaft diameter
            ray_marker.scale.z = 0.05  # head diameter

            ray_marker.color.r = float(min(1.0, cost / 5.0))
            ray_marker.color.g = float(min(1.0, 1.0 - cost / 5.0))
            ray_marker.color.b = 0.0
            ray_marker.color.a = 1.0

            start = Point()
            end = Point()
            end.x = float(ray_vec[0])
            end.y = float(ray_vec[1])
            end.z = 0.0
            ray_marker.points = [start, end]

            marker_array.markers.append(ray_marker)

            if cost < best_cost:
                best_cost = cost
                best_angle = theta

            marker_id += 1

        self.cost_marker_pub.publish(marker_array)

        if best_angle is not None:
            self.move_in_direction(best_angle)
        else:
            self.get_logger().warn("No valid ray direction found")

    def compute_ray_cost(self, ray_vec):
        if not self.curve_points:
            return float('inf')

        ray_unit = ray_vec / np.linalg.norm(ray_vec)
        ray_length = np.linalg.norm(ray_vec)
        curve_array = np.array(self.curve_points)  # shape: (N, 2)

        # Vector from ray start (0,0) to all points
        vec_to_pts = curve_array  # Since ray_start = (0, 0)
        
        # Projection lengths of all points onto the ray
        proj_lens = np.dot(vec_to_pts, ray_unit)  # shape: (N,)

        # Mask for points that project within the ray segment
        valid_mask = (proj_lens >= 0) & (proj_lens <= ray_length)
        
        if not np.any(valid_mask):
            return 0.0  # No obstacle in the way

        # Compute projection points on the ray
        proj_points = np.outer(proj_lens[valid_mask], ray_unit)  # shape: (M, 2)

        # Compute distances from actual points to projected points
        valid_pts = curve_array[valid_mask]
        distances = np.linalg.norm(valid_pts - proj_points, axis=1)

        min_distance_to_lane = np.min(distances)

        obstacle_penalty = 1.0 / (min_distance_to_lane + 1e-2)
        return obstacle_penalty

    def move_in_direction(self, angle_rad):
        # twist = Twist()
        # twist.linear.x = 0.1
        # twist.angular.z = -math.sin(angle_rad) * 0.8
        # self.cmd_pub.publish(twist)
        pass


def main(args=None):
    rclpy.init(args=args)
    node = RaycastNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
