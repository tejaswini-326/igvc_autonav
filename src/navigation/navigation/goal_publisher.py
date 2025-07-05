import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from rclpy.duration import Duration

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.lane = 'right'  # or 'left'
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_point', 10)
        self.marker_sub = self.create_subscription(MarkerArray, '/lane_visualization', self.marker_callback, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info("GoalPublisher node initialized")

    def marker_callback(self, msg):
        lane_markers = [m for m in msg.markers if m.ns == "lane_curves" and m.type == Marker.LINE_STRIP]

        if len(lane_markers) < 3:
            self.get_logger().warn(f"Expected at least 3 lane markers (left, mid, right), got {len(lane_markers)}")
            return

        # Sort by y to guess left < mid < right
        lane_markers.sort(key=lambda m: m.points[0].y)
        left_marker, mid_marker, right_marker = lane_markers[-1], lane_markers[1], lane_markers[0]

        try:
            if self.lane == 'right':
                p1 = self.average_last_n_points(right_marker.points, 5)
            else:
                p1 = self.average_last_n_points(left_marker.points, 5)

            p2 = self.average_last_n_points(mid_marker.points, 5)

            # Midpoint between p1 and p2
            goal_x = (p1[0] + p2[0]) / 2.0
            goal_y = (p1[1] + p2[1]) / 2.0
            goal_z = 0.0

            transformed = self.transform_to_baselink(goal_x, goal_y, goal_z)
            if transformed:
                self.publish_goal(transformed)
            else:
                self.get_logger().warn("Could not transform goal point to base_link")

        except Exception as e:
            self.get_logger().error(f"Failed to estimate goal: {e}")

    def average_last_n_points(self, points, n):
        n = min(n, len(points))
        avg_x = sum(p.x for p in points[-n:]) / n
        avg_y = sum(p.y for p in points[-n:]) / n
        return (avg_x, avg_y)

    def transform_to_baselink(self, x, y, z, frame_id = 'camera_link'):
        try:
            stamped_point = PointStamped()
            stamped_point.header.stamp = self.get_clock().now().to_msg()
            stamped_point.header.frame_id = frame_id
            stamped_point.point.x = x
            stamped_point.point.y = y
            stamped_point.point.z = z

            transform = self.tf_buffer.lookup_transform(
                'odom',
                frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )

            transformed_point = tf2_geometry_msgs.do_transform_point(stamped_point, transform)
            return transformed_point.point

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    def publish_goal(self, point):
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'odom'
        goal_pose.pose.position = point
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0  # No rotation

        self.goal_pub.publish(goal_pose)
        self.get_logger().info(f"Published goal at ({point.x:.2f}, {point.y:.2f}) in odom")

def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
