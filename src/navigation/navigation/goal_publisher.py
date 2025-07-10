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
        self.target_lane = 'right'  # or 'left'
        self.current_lane = 'left'
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_point', 10)
        self.marker_sub = self.create_subscription(MarkerArray, '/lane_visualization', self.marker_callback, 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, '/lane_debug_points', 10)
        self.horizontal_line_pos = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info("GoalPublisher node initialized")
        self.last_rp = None
        self.last_lp = None
        self.last_mp = None

    def marker_callback(self, msg):
        # self.get_logger().info(self.current_lane)
        lane_markers = [m for m in msg.markers if m.ns == "lane_curves" and m.type == Marker.LINE_STRIP]

        try:
            averaged_points = [(marker, self.average_last_n_points(marker.points, 5)) for marker in lane_markers]
            averaged_points = [tup for tup in averaged_points if tup[1] != (0.0, 0.0)]  # Remove invalid ones
            
            averaged_points.sort(key=lambda tup: tup[1][1])  # sort by avg_y
            lane_markers = [tup[0] for tup in averaged_points]

            if len(lane_markers) == 3:
                left_marker, mid_marker, right_marker = lane_markers[-1], lane_markers[1], lane_markers[0]

                rp = self.average_last_n_points(right_marker.points, 5)
                lp = self.average_last_n_points(left_marker.points, 5)
                mp = self.average_last_n_points(mid_marker.points, 5)
                self.last_rp = rp
                self.last_lp = lp
                self.last_mp = mp

                if self.target_lane == 'right':
                    self.goal_x = (rp[0] + mp[0]) / 2.0
                    self.goal_y = (rp[1] + mp[1]) / 2.0
                    self.current_lane = 'right'
                else:
                    self.goal_x = (lp[0] + mp[0]) / 2.0
                    self.goal_y = (lp[1] + mp[1]) / 2.0
                    self.current_lane = 'left'

                self.goal_z = 0.0

            elif len(lane_markers) == 2:
                if self.current_lane == 'right':
                    mid_marker, right_marker = lane_markers[1], lane_markers[0]
                    rp = self.average_last_n_points(right_marker.points, 5)
                    mp = self.average_last_n_points(mid_marker.points, 5)
                    dx = rp[0] - mp[0]  # +ve
                    dy = rp[1] - mp[1]  # -ve
                    lp = (mp[0] - dx, mp[1] - dy)  # try changing mp[0] + dx to mp[0] - dx
                else:
                    mid_marker, left_marker = lane_markers[0], lane_markers[1]
                    lp = self.average_last_n_points(left_marker.points, 5)
                    mp = self.average_last_n_points(mid_marker.points, 5)
                    dx = lp[0] - mp[0]
                    dy = lp[1] - mp[1]
                    rp = (mp[0] - dx, mp[1])
                self.last_rp = rp
                self.last_lp = lp
                self.last_mp = mp

                if self.target_lane == 'right':
                    self.goal_x = (rp[0] + mp[0]) / 2.0
                    self.goal_y = (rp[1] + mp[1]) / 2.0
                    self.current_lane = 'right'
                else:
                    self.goal_x = (lp[0] + mp[0]) / 2.0
                    self.goal_y = (lp[1] + mp[1]) / 2.0
                    self.current_lane = 'left'

                self.goal_z = 0.0

            elif len(lane_markers) == 1:
                if self.current_lane == 'right':
                    right_marker = lane_markers[0]
                    rp = self.average_last_n_points(right_marker.points, 5)
                    rdx = rp[0] - self.last_mp[0]
                    rdy = rp[1] - self.last_mp[1]
                    mp = (rp[0] - rdx, rp[1] - rdy)

                    ldx = self.last_mp[0] - self.last_lp[0]
                    ldy = self.last_lp[1] - self.last_mp[1]
                    mp = (rp[0] - rdx, rp[1] - rdy)
                else:
                    mid_marker, left_marker = lane_markers[0], lane_markers[1]
                    lp = self.average_last_n_points(left_marker.points, 5)
                    mp = self.average_last_n_points(mid_marker.points, 5)
                    dx = lp[0] - mp[0]
                    dy = lp[1] - mp[1]
                    rp = (mp[0] - dx, mp[1])
                self.last_rp = rp
                self.last_lp = lp
                self.last_mp = mp

                if self.target_lane == 'right':
                    self.goal_x = (rp[0] + mp[0]) / 2.0
                    self.goal_y = (rp[1] + mp[1]) / 2.0
                    self.current_lane = 'right'
                else:
                    self.goal_x = (lp[0] + mp[0]) / 2.0
                    self.goal_y = (lp[1] + mp[1]) / 2.0
                    self.current_lane = 'left'

                self.goal_z = 0.0         

            else:
                self.get_logger().warn("Not enough lane markers for goal computation.")
                return

            transformed = self.transform_to_odom(self.goal_x, self.goal_y, self.goal_z)
            if transformed:
                self.publish_goal(transformed)
                self.publish_debug_markers()
            else:
                self.get_logger().warn("Could not transform goal point to base_link")

        except Exception as e:
            self.get_logger().error(f"Failed to estimate goal: {e}")

    def average_last_n_points(self, points, n, max_distance=6.5): #added a dsitance threshold for goal calc
        # Only include points within max_distance from origin
        filtered = [p for p in points if (p.x**2 + p.y**2)**0.5 <= max_distance]

        if not filtered:
            self.get_logger().warn("No points within distance threshold.")
            # return (0.0, 0.0)

        n = min(n, len(filtered))
        avg_x = sum(p.x for p in filtered[-n:]) / n
        avg_y = sum(p.y for p in filtered[-n:]) / n
        return (avg_x, avg_y)

    def transform_to_odom(self, x, y, z, frame_id='camera_link'):
        try:
            stamped_point = PointStamped()
            stamped_point.header.stamp = self.get_clock().now().to_msg()
            stamped_point.header.frame_id = frame_id
            stamped_point.point.x = x
            stamped_point.point.y = y
            stamped_point.point.z = z

            transform = self.tf_buffer.lookup_transform('odom', frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.5))
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
        goal_pose.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_pose)
        # self.get_logger().info(f"Published goal at ({point.x:.2f}, {point.y:.2f}) in odom")

    def publish_debug_markers(self):
        marker_array = MarkerArray()
        marker_id = 0
        timestamp = self.get_clock().now().to_msg()

        def make_marker(point, color, label):
            nonlocal marker_id
            m = Marker()
            m.header.stamp = timestamp
            m.header.frame_id = "camera_link"
            m.ns = label
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = point[0]
            m.pose.position.y = point[1]
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color.r = color[0]
            m.color.g = color[1]
            m.color.b = color[2]
            m.color.a = 1.0
            return m

        if self.last_rp:
            marker_array.markers.append(make_marker(self.last_rp, (1.0, 0.0, 0.0), "right_point"))
        if self.last_lp:
            marker_array.markers.append(make_marker(self.last_lp, (0.0, 1.0, 0.0), "left_point"))
        if self.last_mp:
            marker_array.markers.append(make_marker(self.last_mp, (0.0, 0.0, 1.0), "mid_point"))

        self.debug_marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()