#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class ContourPublisher(Node):
    def __init__(self):
        super().__init__("contour_publisher")

        # Subscriber to costmap
        self.sub = self.create_subscription(
            OccupancyGrid, "/costmap", self.callback, 10
        )

        # Publisher for visualization
        self.pub = self.create_publisher(Marker, "/contours_marker", 10)

    def callback(self, msg: OccupancyGrid):
        # Extract map info
        # Clear old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = msg.header.frame_id
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(delete_marker)

        width, height = msg.info.width, msg.info.height
        resolution = msg.info.resolution
        origin_x, origin_y = msg.info.origin.position.x, msg.info.origin.position.y

        # Convert occupancy data into numpy image
        img = np.array(msg.data, dtype=np.int16).reshape((height, width))
        img[img < 0] = 0   # Replace unknowns

        # Extract only corridor band (20–105)
        mask1 = cv2.inRange(img, 0, 15)       # almost free / empty (white-ish)
        mask2 = cv2.inRange(img, 75, 255)     # higher free values
        corridor_mask = cv2.bitwise_or(mask1, mask2)

        contours, hierarchy = cv2.findContours(
            corridor_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Loop over each contour and publish separately
        for idx, cnt in enumerate(contours):
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "contours"
            marker.id = idx
            marker.type = Marker.LINE_STRIP   # connected line
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            for point in cnt:
                x, y = point[0]
                p = Point()
                p.x = origin_x + x * resolution
                p.y = origin_y + y * resolution
                p.z = 0.05
                marker.points.append(p)

            self.pub.publish(marker)
            self.get_logger().info(f"Published contour {idx} with {len(cnt)} points")


def main(args=None):
    rclpy.init(args=args)
    node = ContourPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
