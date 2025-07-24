import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf_transformations import euler_from_quaternion
from tf2_ros import TransformListener, Buffer
from rclpy.duration import Duration
import math


class PathRayVisualizer(Node):
    def __init__(self):
        super().__init__('path_ray_visualizer')

        self.path_sub = self.create_subscription(
            Path, '/sm_planned_path', self.path_callback, 10)

        self.ray_pub = self.create_publisher(MarkerArray, '/ray_visualization', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.target_heading = 0.0
        self.timer = self.create_timer(0.1, self.publish_rays)

    def path_callback(self, msg):
        if not msg.poses:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', msg.header.frame_id,
                rclpy.time.Time(), timeout=Duration(seconds=0.5)
            )
            robot_pos = transform.transform.translation
            robot_x, robot_y = robot_pos.x, robot_pos.y

            closest_pose = min(
                msg.poses,
                key=lambda p: (p.pose.position.x - robot_x) ** 2 +
                              (p.pose.position.y - robot_y) ** 2
            )

            q = closest_pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.target_heading = yaw

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

    def publish_rays(self):
        marker_array = MarkerArray()

        # Rays in base_link frame
        marker_array.markers.extend(
            self.generate_rays_in_frame("base_link", origin=(0.0, 0.0), yaw=0.0, start_id=0)
        )

        # Rays in odom frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.5)
            )
            origin = (
                transform.transform.translation.x,
                transform.transform.translation.y,
            )
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            marker_array.markers.extend(
                self.generate_rays_in_frame("odom", origin=origin, yaw=yaw, start_id=1000)
            )

        except Exception as e:
            self.get_logger().warn(f"TF transform to odom failed: {e}")
            return

        self.ray_pub.publish(marker_array)

    def generate_rays_in_frame(self, frame_id, origin, yaw, start_id):
        num_rays = 60
        angle_range = math.pi*(110/180)
        start_angle = -angle_range / 2
        length = 4.0
        rays = []

        for i in range(num_rays):
            angle = start_angle + i * (angle_range / (num_rays - 1))
            global_angle = angle + yaw
            angle_diff = abs(angle - self.target_heading)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            norm = angle_diff / (angle_range / 2)

            r = min(1.0, norm)
            g = max(0.0, 1.0 - norm)
            b = 0.0

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"ray_fan_{frame_id}"
            marker.id = start_id + i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.04  # shaft diameter
            marker.scale.y = 0.08  # head diameter
            marker.scale.z = 0.1   # head length

            start = Point(x=origin[0], y=origin[1], z=0.0)
            end = Point(
                x=origin[0] + length * math.cos(global_angle),
                y=origin[1] + length * math.sin(global_angle),
                z=0.0
            )

            marker.points = [start, end]
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0

            marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()

            rays.append(marker)

        return rays


def main(args=None):
    rclpy.init(args=args)
    node = PathRayVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
