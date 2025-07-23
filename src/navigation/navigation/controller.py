'''Pure pursuit controller for published path'''
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import Twist, Point, PointStamped
import tf_transformations
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Imu
from collections import deque
from time import time
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Path

VERBOSE_UNECESSARY_THINGS = False


class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        self.path = []

        self.lookahead_distance = 1.2
        self.linear_speed = 0.5
        self.goal_tolerance = 0.5
        self.control_rate = 10  # Hz

        self.yaw_buffer = deque(maxlen=10)
        self.prev_angular_z = 0.0
        self.angular_damping_factor = .95
        self.current_lookahead = None
        self.max_angular_speed = 0.75

        self.last_log_time = 0.0
        self.log_interval = 0.5

        self.active = True
        self.pose = None
        self.imu_yaw = None
        self.scanning = False

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Path, '/sm_planned_path', self.path_callback, 10)
        self.create_subscription(String, '/intersection', self.intersection_cb, 10) 

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.marker_timer = self.create_timer(0.05, self.publish_marker_timer)

        self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        #self.goal_reached = False


        self.get_logger().info("Controller initialized.")

    def intersection_cb(self, msg: String):
        if msg.data.lower()  == "none":
            self.active = True
            self.get_logger().info(f"received '{msg.data}' from /intersection.")
        else:
            self.active = False
            self.get_logger().info(f"ignoring '{msg.data}' from /intersection.")

    def log_info_throttled(self, msg):
        if not self.active:
            return
        now = time()
        if now - self.last_log_time > self.log_interval:
            if VERBOSE_UNECESSARY_THINGS: self.get_logger().info(msg)
            self.last_log_time = now

    def scan(self):
        self.scanning = True


    def odom_callback(self, msg):
        if not self.active:
            return
        self.pose = msg.pose.pose

    def imu_callback(self, msg):
        if not self.active:
            return

        try:
            q = msg.orientation
            quat = [q.x, q.y, q.z, q.w]
            _, _, yaw = tf_transformations.euler_from_quaternion(quat)

            self.imu_yaw = yaw
            if VERBOSE_UNECESSARY_THINGS: self.get_logger().info("IMU yaw = {:.2f}".format(yaw))
        except Exception as e:
            if VERBOSE_UNECESSARY_THINGS: self.get_logger().warn(f"[IMU callback error] {e}")

        self.yaw_buffer.append(yaw)

        if len(self.yaw_buffer) > 2:
            yaw_array = np.unwrap(np.array(self.yaw_buffer))

            median = np.median(yaw_array)
            deviation = np.abs(yaw_array - median)
            std_dev = np.std(yaw_array)

            if std_dev > 0.05:
                filtered_yaws = yaw_array[deviation < 1.5 * std_dev]
            else:
                filtered_yaws = yaw_array 

            if len(filtered_yaws) > 0:
                self.imu_yaw = float(np.mean(filtered_yaws))
            else:
                self.imu_yaw = float(median)
        else:
            self.imu_yaw = yaw
        #self.get_logger().info("IMU callback received.")

    def path_callback(self, msg: Path):
        if msg and msg.poses:
            self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
            if self.scanning:
                self.get_logger().info("Path received. Stopping scan.")
                self.scanning = False
                if hasattr(self, 'scan_timer'):
                    self.scan_timer.cancel()
        else:
            if not self.scanning:
                self.get_logger().info("No path received. Starting scan.")
                self.scan()
 
        if VERBOSE_UNECESSARY_THINGS: self.get_logger().info(f"Received path with {len(self.path)} valid points.")

    def adaptive_lookahead(self, base_distance=1.2):
        if not self.active:
            return
        yaw_variability = np.std(self.yaw_buffer) if len(self.yaw_buffer) > 2 else 0.0
        adaptive_factor = np.clip(1.0 + 2.5 * yaw_variability, 1.0, 1.5)
        return base_distance * adaptive_factor
        #self.get_logger().info("adapted whatever")

    def find_lookahead_point(self):
        if not self.active:
            return
        if self.pose is None or not self.path:
            return None

        rx, ry = self.pose.position.x, self.pose.position.y
        if self.imu_yaw is None:
            self.get_logger().warn("IMU yaw not yet available.")
            return None

        closest_idx = int(np.argmin([math.hypot(px - rx, py - ry) for px, py in self.path]))
        max_lookahead_range = 3.0

        for px, py in self.path[closest_idx:]:
            dx, dy = px - rx, py - ry
            dist = math.hypot(dx, dy)

            if dist < 1e-3 or dist > max_lookahead_range:
                continue

            forward_vec = np.array([math.cos(self.imu_yaw), math.sin(self.imu_yaw)])
            to_point_vec = np.array([dx, dy])

            if np.dot(forward_vec, to_point_vec / (np.linalg.norm(to_point_vec) + 1e-6)) < 0.3:
                continue

            try:
                idx = self.path.index((px, py))
                if idx + 1 >= len(self.path):
                    continue
                path_dx = self.path[idx + 1][0] - px
                path_dy = self.path[idx + 1][1] - py
                path_dir = np.array([path_dx, path_dy])
                path_dir /= np.linalg.norm(path_dir) + 1e-6

                alignment = np.dot(forward_vec, path_dir)
                if alignment < 0.0:
                    continue
            except ValueError:
                continue

            if dist >= self.lookahead_distance:
                self.log_info_throttled(
                    f"Lookahead chosen: ({px:.2f}, {py:.2f}), dist={dist:.2f}, align={alignment:.2f}"
                )
                return (px, py)

        self.log_info_throttled("No valid forward lookahead point found! Using last point.")
        return self.path[-1]

    def transform_point_to_base_link(self, point):
        try:
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = "odom"
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = 0.0

            transform = self.tf_buffer.lookup_transform(
                target_frame='base_link',
                source_frame='odom',
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed.point.x, transformed.point.y

        except Exception as e:
            self.log_info_throttled(f"[TF ERROR] {str(e)}")
            return 0.0, 0.0

    def control_loop(self):
        if not self.active:
            return
        if self.pose is None or not self.path:
            return

        gx, gy = self.path[-1]
        dx = gx - self.pose.position.x
        dy = gy - self.pose.position.y
        if math.hypot(dx, dy) < self.goal_tolerance:
            #self.goal_reached = True
            self.log_info_throttled("Reached goal.")
            self.cmd_pub.publish(Twist())
            return

        self.lookahead_distance = min(self.adaptive_lookahead(), 1.2)
        lookahead = self.find_lookahead_point()
        self.current_lookahead = lookahead
        if lookahead is None:
            return

        xr, yr = self.transform_point_to_base_link(lookahead)
        lookahead_dist = math.hypot(xr, yr)
        angle = math.atan2(yr, xr)
        if abs(math.degrees(angle)) < .5:
            angle = 0.0

        if lookahead_dist < 0.1:
            curvature = 0.0
        else:
            curvature = np.clip(2 * yr / (lookahead_dist ** 2 + 1e-3), -2.0, 2.0)

        self.log_info_throttled(
            f"Lookahead (robot frame): xr={xr:.2f}, yr={yr:.2f}, angle={math.degrees(angle):.2f}°"
        )

        twist = Twist()
        if xr < 0:
            self.log_info_throttled("Lookahead behind. Turning.")
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.z = float(np.sign(angle) * self.max_angular_speed)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            self.cmd_pub.publish(twist)
            return

        if lookahead_dist < 0.2:
            if abs(angle) > math.radians(10):
                self.log_info_throttled("Close to lookahead but misaligned. Rotating in place.")
                twist.linear.x = 0.0
                twist.linear.y = 0.0
                twist.linear.z = 0.0
                twist.angular.z = float(np.clip(2.0 * angle, -self.max_angular_speed, self.max_angular_speed))
                twist.angular.x = 0.0
                twist.angular.y = 0.0
            else:
                self.log_info_throttled("Lookahead very close and aligned. Continuing cautiously.")
                twist.linear.x = float(self.linear_speed) #  * 0.3
                twist.linear.y = 0.0
                twist.linear.z = 0.0
                twist.angular.z = 0.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
            self.cmd_pub.publish(twist)
            return

        if abs(angle) > math.radians(25):
            twist.linear.x = float(self.linear_speed) #  * 0.1
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.z = float(np.clip(2.0 * angle, -self.max_angular_speed, self.max_angular_speed))
            twist.angular.x = 0.0
            twist.angular.y = 0.0

        elif abs(angle) > math.radians(5):
            twist.linear.x = float(self.linear_speed) #  * 0.4
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.z = float(np.clip(1.5 * angle, -self.max_angular_speed, self.max_angular_speed))
            twist.angular.x = 0.0
            twist.angular.y = 0.0
        else:
            twist.linear.x = float(self.linear_speed)
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.z = float(np.clip(1.2 * angle, -self.max_angular_speed, self.max_angular_speed))
            twist.angular.x = 0.0
            twist.angular.y = 0.0

        twist.angular.z = (self.angular_damping_factor * self.prev_angular_z +(1 - self.angular_damping_factor) * twist.angular.z)
        self.prev_angular_z = twist.angular.z
        twist.angular.z = float(np.clip(twist.angular.z, -1.0, 1.0))
        self.cmd_pub.publish(twist)
        if VERBOSE_UNECESSARY_THINGS: self.get_logger().info(f"Publishing Twist: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}")
      


    def publish_marker_timer(self):
        if not self.active:
            return
        self.publish_markers(self.path)

    def publish_markers(self, raw_path):
        if not self.active:
            return
        
        if raw_path is None:
            return

        def make_marker(points, mid, color, ns):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = mid
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker.lifetime = Duration(sec=0)
            marker.points = [Point(x=x, y=y, z=0.0) for x, y in points]
            return marker

        def make_lookahead_marker(point, mid, ns):
            if point is None:
                return None
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = mid
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.0
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            marker.color.r = 0.0
            marker.color.g = 0.5
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.lifetime = Duration(sec=0)
            return marker

        markers = [make_marker(raw_path, 0, (1.0, 0.0, 0.0, 1.0), "raw_path")]
        lookahead_marker = make_lookahead_marker(self.current_lookahead, 2, "lookahead_point")
        if lookahead_marker:
            markers.append(lookahead_marker)

        self.marker_pub.publish(MarkerArray(markers=markers))

    def transform_to_odom(self, x, y, z, frame_id='base_link'):
        try:
            point = PointStamped()
            point.header.frame_id = frame_id
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x = x
            point.point.y = y
            point.point.z = z

            #checks for latest transforms
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
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()