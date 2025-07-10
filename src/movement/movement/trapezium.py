#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import math

class WhiteCornerDebugger(Node):
    def __init__(self):
        super().__init__('white_corner_debugger')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/igvc/white_points',
            self.pc_callback,
            10
        )
        self.marker_pub = self.create_publisher(MarkerArray, '/white_corner_markers', 10)
        self.marker_id = 0
        self.get_logger().info("White corner debugger node started.")

    def pc_callback(self, msg):
        points = np.array([
            [p[0], p[1]] for p in pc2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        ])
        if len(points) < 20:
            return

        clustering = DBSCAN(eps=0.3, min_samples=10).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels)

        marker_array = MarkerArray()
        self.marker_id = 0

        for label in unique_labels:
            if label == -1:
                continue
            cluster = points[labels == label]
            if len(cluster) < 30:
                continue

            m1, inliers1, line1_pts = self.fit_ransac(cluster)
            if m1 is None:
                continue
            outliers = cluster[~inliers1]
            if len(outliers) < 10:
                continue
            m2, _, line2_pts = self.fit_ransac(outliers)
            if m2 is None:
                continue

            # Always publish both lines
            marker_array.markers.append(self.create_line_marker(line1_pts, color=(1.0, 0.0, 0.0)))  # Red
            marker_array.markers.append(self.create_line_marker(line2_pts, color=(0.0, 1.0, 0.0)))  # Green

            # Only log if angle is ~90°
            angle_deg = abs(math.degrees(math.atan((m2 - m1) / (1 + m1 * m2))))
            if 80 <= angle_deg <= 100:
                self.get_logger().info(f"90° corner detected! Angle = {angle_deg:.2f}°")

        self.marker_pub.publish(marker_array)

    def fit_ransac(self, points):
        if len(points) < 10:
            return None, None, None
        try:
            X = points[:, 0].reshape(-1, 1)
            y = points[:, 1]
            model = RANSACRegressor()
            model.fit(X, y)
            m = model.estimator_.coef_[0]
            inliers = model.inlier_mask_

            # Create endpoints for line visualization
            x_vals = np.linspace(X[inliers].min(), X[inliers].max(), 2)
            y_vals = model.predict(x_vals.reshape(-1, 1))
            line_pts = list(zip(x_vals, y_vals))

            return m, inliers, line_pts
        except Exception as e:
            self.get_logger().warn(f"RANSAC failed: {e}")
            return None, None, None

    def create_line_marker(self, line_pts, color=(1.0, 0.0, 0.0)):
        marker = Marker()
        marker.header.frame_id = "odom"  # Change if needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "debug_lines"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        for x, y in line_pts:
            pt = self.create_point(x, y)
            marker.points.append(pt)

        return marker

    def create_point(self, x, y, z=0.0):
        from geometry_msgs.msg import Point
        pt = Point()
        pt.x = float(x)
        pt.y = float(y)
        pt.z = float(z)
        return pt

def main(args=None):
    rclpy.init(args=args)
    node = WhiteCornerDebugger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
