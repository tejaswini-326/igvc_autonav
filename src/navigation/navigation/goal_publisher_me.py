#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from itertools import combinations
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
        self.heading_sub = self.create_subscription("/heading_angle", self.heading_cb, 10)

        # Publisher for visualization
        self.pub = self.create_publisher(Marker, "/contours_marker", 10)
        self.bot_x = 300
        self.bot_y = 300
        self.cone_angle = 30
        self.cone_radius = 5
        self.heading = 0

    def heading_cb(self, msg):
        self.heading = msg

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
        mask1 = cv2.inRange(img, 0, 20)       # almost free / empty (white-ish)
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

    def points_in_cone(self, contours):
        cone_angle /= 2  # half-angle for comparison
        inside_points = {}

        for i, contour in enumerate(contours):
            pts_in_cone = []
            for point in contour:
                x, y = point[0]
                dx = x - self.box_x
                dy = y - self.bot_y
                dist = np.hypot(dx, dy)
                if dist > self.cone_radius:
                    continue
                angle_to_point = np.arctan2(dy, dx)
                angle_diff = np.arctan2(np.sin(angle_to_point - self.heading),
                                        np.cos(angle_to_point - self.heading))
                if abs(angle_diff) <= cone_angle:
                    pts_in_cone.append((x, y))
            if pts_in_cone:
                inside_points[i] = pts_in_cone
        return inside_points



    def find_goal_from_contours(points_dict):
        contour_ids = list(points_dict.keys())

        # 1. Single contour
        if len(contour_ids) == 1:
            points = np.array(points_dict[contour_ids[0]])
            goal = np.mean(points, axis=0)
            return goal

        # 2. Multiple contours
        closest_pair = None
        min_dist = float('inf')

        for id1, id2 in combinations(contour_ids, 2):
            points1 = np.array(points_dict[id1])
            points2 = np.array(points_dict[id2])
            # Compute all pairwise distances efficiently
            diff = points1[:, None, :] - points2[None, :, :]  # shape (N1, N2, 2)
            dists = np.linalg.norm(diff, axis=2)
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            dist = dists[idx]
            if dist < min_dist:
                min_dist = dist
                closest_pair = (points1[idx[0]], points2[idx[1]])

        # Midpoint of closest points
        goal = (closest_pair[0] + closest_pair[1]) / 2
        return goal


def main(args=None):
    rclpy.init(args=args)
    node = ContourPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
