# yellow_line_follower.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import math
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class YellowLineFollower(Node):
    def __init__(self):
        super().__init__('yellow_line_follower')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.pointcloud_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/lane_marker', 10)
        self.last_cmd = Twist()
        self.last_cmd.linear.x = 1.0
        self.last_cmd.angular.z = 0.0
        self.which_lane = 'right'  # can be 'left' or 'right'

    def publish(self, cmd):
        self.cmd_pub.publish(cmd)

    def calculate_velocity(self, target, msg, img, centers):
        cmd = Twist()
        angle = math.atan2(target[1], target[0])

        cmd.linear.x = 5.0
        if abs(angle) > 0.05:
            cmd.angular.z = angle
            if abs(angle) > 0.4:
                cmd.linear.x = 0.2
        else:
            cmd.angular.z = 0.0

        return cmd

    def pointcloud_callback(self, msg):
        height = msg.height
        width = msg.width
        yellow_img = np.zeros((height, width, 3), dtype=np.uint8)
        yellow_points = []

        index = 0
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
            x, y, z, rgb = point
            row = index // width
            col = index % width

            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            except:
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            # Detect yellow
            if r > 150 and g > 150 and b < 100:
                if -1.4 < z < -1.3 and 0.0 < x < 3.0:
                    yellow_img[row, col] = (0, 255, 255)
                    yellow_points.append([x, y, z])
            index += 1

        if len(yellow_points) < 10:
            self.get_logger().warn("Not enough yellow points detected")
            self.publish(self.last_cmd)
            return

        points_np = np.array(yellow_points)
        points_xy = points_np[:, :2]

        clustering = DBSCAN(eps=0.75, min_samples=20).fit(points_xy)
        labels = clustering.labels_
        unique_labels = set(labels)

        centers = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster = points_xy[labels == label]
            if len(cluster) < 10:
                continue
            center = np.mean(cluster, axis=0)
            centers.append((label, center))

        cmd = Twist()
        if len(centers) == 2:
            centers.sort(key=lambda c: c[1][1])
            left = centers[1][1]
            right = centers[0][1]
            target = (left + right) / 2
            cmd = self.calculate_velocity(target, msg, yellow_img, centers)
            self.publish(cmd)
            self.last_cmd = cmd
            return

        elif len(centers) >= 3:
            centers.sort(key=lambda c: c[1][1])
            right = centers[0][1]
            middle = centers[1][1]
            left = centers[2][1]
            target = ((middle + right) / 2) if self.which_lane == 'right' else ((middle + left) / 2)
            cmd = self.calculate_velocity(target, msg, yellow_img, centers)
            self.publish(cmd)
            self.last_cmd = cmd
            return

        elif len(centers) == 1:
            label, center = centers[0]
            cluster_points = points_xy[labels == label]

            if len(cluster_points) > 7:
                x = cluster_points[:, 0]
                y = cluster_points[:, 1]
                coeffs = np.polyfit(x, y, 2)
                a, b, c = coeffs
                x_center = np.mean(x)
                slope = 2 * a * x_center + b
                angle = math.atan(slope)

                cmd.linear.x = 0.2
                cmd.angular.z = angle

                marker = Marker()
                marker.header.frame_id = msg.header.frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "fitted_yellow"
                marker.id = 0
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.05
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                x_vals = np.linspace(np.min(x), np.max(x), num=30)
                for x_i in x_vals:
                    y_i = a * x_i**2 + b * x_i + c
                    z_i = np.mean(points_np[:, 2])
                    marker.points.append(Point(x=x_i, y=y_i, z=z_i))

                self.marker_pub.publish(marker)
            else:
                cmd.linear.x = 0.2
                cmd.angular.z = 0.0

            self.publish(cmd)
            self.last_cmd = cmd
        else:
            self.get_logger().warn("No valid clusters found")
            self.publish(self.last_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = YellowLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
