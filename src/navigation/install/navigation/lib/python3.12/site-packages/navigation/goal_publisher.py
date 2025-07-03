import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
import os 

class Goal_Publisher(Node):
    def __init__(self):
        super().__init__("path_planner_node")
        self.line_marker_sub = self.create_subscription(MarkerArray, '/lane_visualization', self.estimate_goal_from_markers, 10)
        self.goal_pub = self.create_publisher(Point, '/goal_point', 10)

        self.param_file_path = os.path.expanduser('~/.config/config_igvc_ui/config.yaml')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal = Point()
        self.lane = 'right'
        
    def estimate_goal_from_markers(self, marker_array):
        lane_markers = [m for m in marker_array.markers if m.ns == "lane_curves" and m.type == Marker.LINE_STRIP]

        if len(lane_markers) < 2:
            self.get_logger().warn("Not enough lane markers to estimate goal")
            return None

        sorted_markers = sorted(lane_markers, key=lambda m: m.points[0].y)

        if(self.lane == 'right'):
            right_marker = sorted_markers[0]
            right_points = right_marker.points
            mid_marker = sorted_markers[1]
            mid_points = mid_marker.points
            # Use last few points to compute average goal
            N = min(5, len(right_points), len(mid_points))

            # Average the last N points
            avg_rx = sum(p.x for p in right_points[-N:]) / N
            avg_ry = sum(p.y for p in right_points[-N:]) / N
        else:
            left_marker = sorted_markers[2]
            left_points = left_marker.points
            mid_marker = sorted_markers[1]
            mid_points = mid_marker.points
            N = min(5, len(left_points), len(mid_points))
            avg_lx = sum(p.x for p in left_points[-N:]) / N
            avg_ly = sum(p.y for p in left_points[-N:]) / N

        if N < 2:
            self.get_logger().warn("Too few points in lane markers")
            return None

        avg_mx = sum(p.x for p in mid_points[-N:]) / N
        avg_my = sum(p.y for p in mid_points[-N:]) / N

        # Midpoint
        if(self.lane == 'right'):
            mid_x = (avg_rx + avg_mx) / 2.0
            mid_y = (avg_ry + avg_my) / 2.0
            mid_z = 0.0
            # Transform to base_link
            result = self.transform_to_baselink(mid_x, mid_y, mid_z, frame_id=right_marker.header.frame_id)
        else:
            mid_x = (avg_lx + avg_mx) / 2.0
            mid_y = (avg_ly + avg_my) / 2.0
            mid_z = 0.0
            result = self.transform_to_baselink(mid_x, mid_y, mid_z, frame_id=left_marker.header.frame_id)

        if result is not None:
            self.goal.x, self.goal.y, self.goal.z = result
            self.goal_pub.publish(self.goal)
        else:
            self.get_logger().warn("Failed to transform goal point to base_link")


    def transform_to_baselink(self, x, y, z, frame_id='camera_link'):
        try:
            point = PointStamped()
            point.header.frame_id = frame_id
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x = x
            point.point.y = y
            point.point.z = z

            if self.tf_buffer.can_transform('base_link', frame_id, rclpy.time.Time()):
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    frame_id,
                    rclpy.time.Time(),
                )
            else:
                self.get_logger().warn(f"Transform from {frame_id} to base_link not available")
                return None

            transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
            return (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)

        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = Goal_Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()