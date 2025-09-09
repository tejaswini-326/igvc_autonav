#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from itertools import combinations
import cv2
import math
import tf_transformations

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Float32


class ContourPublisher(Node):
    def __init__(self):
        super().__init__("contour_publisher")

        # Subscriber to costmap
        self.sub = self.create_subscription(
            OccupancyGrid, "/costmap", self.callback, 10
        )
        # Heading (expecting Float32 on /heading_angle) — safe extraction handled
        self.heading_sub = self.create_subscription(Float32, "/heading_angle", self.heading_cb, 10)

        # Fallback: odom subscription for heading (if /heading_angle missing/stale)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # Publisher for visualization
        self.marker_pub = self.create_publisher(Marker, "/contours_marker", 10)
        self.goal_pub = self.create_publisher(PointStamped, "/new_goal", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        # robot/map params (these are in image/grid coordinates)
        self.bot_x = 130
        self.bot_y = 150
        self.cone_angle = 35      # degrees
        self.cone_radius = 60     # units: pixels (choose consistent with your map)

        # heading sources
        self.heading = 0.0            # last heading from /heading_angle (radians)
        self.heading_time = None      # rclpy Time when heading was received
        self.odom_heading = 0.0       # last heading extracted from /odom
        self.odom_time = None

    def heading_cb(self, msg: Float32):
        # Support Float32 or raw float-like input; store time
        try:
            self.heading = float(msg.data)
            self.heading_time = self.get_clock().now()
        except Exception:
            # ignore malformed messages
            pass

    def odom_cb(self, msg: Odometry):
        try:
            q = msg.pose.pose.orientation
            _, _, yaw = tf_transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
            self.odom_heading = float(yaw)
            self.odom_time = self.get_clock().now()
        except Exception:
            pass


    def callback(self, msg: OccupancyGrid):
        # Clear old markers
        delete_marker = Marker()
        delete_marker.ns = "contours"
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = msg.header.frame_id
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(delete_marker)

        width, height = msg.info.width, msg.info.height
        resolution = msg.info.resolution
        origin_x, origin_y = msg.info.origin.position.x, msg.info.origin.position.y

        # Convert occupancy data into numpy image
        img = np.array(msg.data, dtype=np.int16).reshape((height, width))
        img[img < 0] = 0   # Replace unknowns

        # Convert to uint8 for OpenCV functions
        img_u8 = img.astype(np.uint8)

        # Example: build corridor_mask (adjust thresholds to your map)
        mask1 = cv2.inRange(img_u8, 0, 20)       # almost free / empty
        mask2 = cv2.inRange(img_u8, 75, 255)     # higher values
        corridor_mask = cv2.bitwise_or(mask1, mask2)

        # find contours
        contours_result = cv2.findContours(corridor_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV returns either (contours, hierarchy) or (image, contours, hierarchy) depending on version
        if len(contours_result) == 3:
            _, contours, hierarchy = contours_result
        else:
            contours, hierarchy = contours_result

        # Publish raw contours for visualization (unchanged)
        for idx, cnt in enumerate(contours):
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "contours"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            for point in cnt:
                x, y = point[0]
                p = Point()
                # pixel -> world (same convention used elsewhere)
                p.x = origin_x + float(x) * resolution
                p.y = origin_y + float(y) * resolution
                p.z = 0.05
                marker.points.append(p)

            self.marker_pub.publish(marker)

        # --- compute robot pixel center in the incoming costmap window ----
        # Your costmap node centers the robot, so use center pixel.
        bot_px = width // 2
        bot_py = height // 2
        # keep self.bot_x/self.bot_y updated for backward compatibility
        self.bot_x = bot_px
        self.bot_y = bot_py

        # robot world coords (pixel -> world)
        bot_world_x = origin_x + float(bot_px) * resolution
        bot_world_y = origin_y + float(bot_py) * resolution

        # Pick heading source: prefer /heading_angle if fresh, else fall back to /odom
        now = self.get_clock().now()
        chosen_src = "none"
        chosen_heading = None
        freshness_s = 0.5  # seconds

        if self.heading_time is not None:
            age = (now - self.heading_time).nanoseconds * 1e-9
            if age <= freshness_s:
                chosen_heading = float(self.heading)
                chosen_src = "/heading_angle"

        if chosen_heading is None and self.odom_time is not None:
            ageo = (now - self.odom_time).nanoseconds * 1e-9
            if ageo <= freshness_s:
                chosen_heading = float(self.odom_heading)
                chosen_src = "/odom"

        if chosen_heading is None:
            # last-resort: keep previous heading value (could be initial 0.0)
            chosen_heading = float(self.heading if self.heading_time is not None else self.odom_heading)
            chosen_src = "last_known"

        # Debug log (throttle as needed)
        self.get_logger().debug(f"Heading chosen from {chosen_src}: {chosen_heading:.3f} rad")

        # --- find points inside the cone using WORLD coordinates (robust) ---
        inside_points = self.points_in_cone_world(
            contours, origin_x, origin_y, resolution, bot_world_x, bot_world_y, chosen_heading
        )

        if not inside_points:
            self.get_logger().info("No contour points inside cone.")
            return

        result = self.find_goal_from_contours(inside_points)
        if result is None:
            # This includes the single-contour case (we treat as obstacle and ignore)
            self.get_logger().info("Only one contour visible in cone (obstacle). Ignoring for now.")
            return

        # result is (goal_pixel, closest_pair_pixel, min_dist)
        goal_pixel, closest_pair, min_dist = result

        # Convert pixel goal -> world coords using map origin + resolution
        goal_world_x = origin_x + float(goal_pixel[0]) * resolution
        goal_world_y = origin_y + float(goal_pixel[1]) * resolution

        # Publish goal marker (sphere)
        goal_marker = Marker()
        goal_marker.header.frame_id = msg.header.frame_id
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "gap_goal"
        goal_marker.id = 999
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.scale.x = 0.12
        goal_marker.scale.y = 0.12
        goal_marker.scale.z = 0.12
        goal_marker.color.r = 0.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 1.0
        goal_marker.color.a = 1.0
        goal_marker.pose.position.x = goal_world_x
        goal_marker.pose.position.y = goal_world_y
        goal_marker.pose.position.z = 0.1
        goal_marker.pose.orientation.w = 1.0
        self.marker_pub.publish(goal_marker)

        # Original point in costmap frame (PointStamped)
        gap_point = PointStamped()
        gap_point.header.frame_id = msg.header.frame_id  # "odom"
        gap_point.header.stamp = self.get_clock().now().to_msg()
        gap_point.point.x = goal_world_x
        gap_point.point.y = goal_world_y
        gap_point.point.z = 0.0
        self.goal_pub.publish(gap_point)

        # OPTIONAL: publish the two closest points as small markers for debugging
        (pA, pB) = closest_pair
        for i, p_pixel in enumerate((pA, pB)):
            p_world_x = origin_x + float(p_pixel[0]) * resolution
            p_world_y = origin_y + float(p_pixel[1]) * resolution
            pm = Marker()
            pm.header.frame_id = msg.header.frame_id
            pm.header.stamp = self.get_clock().now().to_msg()
            pm.ns = "closest_pair"
            pm.id = 1000 + i
            pm.type = Marker.SPHERE
            pm.action = Marker.ADD
            pm.scale.x = pm.scale.y = pm.scale.z = 0.06
            pm.color.r = 1.0
            pm.color.g = 1.0
            pm.color.b = 0.0
            pm.color.a = 1.0
            pm.pose.position.x = p_world_x
            pm.pose.position.y = p_world_y
            pm.pose.position.z = 0.08
            pm.pose.orientation.w = 1.0
            self.marker_pub.publish(pm)

        # --- Cone visualization marker (world coords, no Y flip) ---
        cone_marker = Marker()
        cone_marker.header.frame_id = msg.header.frame_id
        cone_marker.header.stamp = self.get_clock().now().to_msg()
        cone_marker.ns = "cone"
        cone_marker.id = 1
        cone_marker.type = Marker.LINE_STRIP
        cone_marker.action = Marker.ADD
        cone_marker.scale.x = 0.2   # thicker for visibility
        cone_marker.color.r = 1.0
        cone_marker.color.g = 0.0
        cone_marker.color.b = 0.0
        cone_marker.color.a = 1.0

        cone_radius = self.cone_radius * resolution
        half_angle = np.deg2rad(self.cone_angle / 2.0)

        # apex of the cone
        cone_marker.points.append(Point(x=bot_world_x, y=bot_world_y, z=0.05))

        robot_heading = float(chosen_heading)  # radians in odom/world

        # draw cone edges — no Y flip (consistent with world coords)
        for angle_offset in np.linspace(-half_angle, half_angle, 15):
            angle = robot_heading + angle_offset
            x = bot_world_x + cone_radius * np.cos(angle)
            y = bot_world_y + cone_radius * np.sin(angle)
            cone_marker.points.append(Point(x=x, y=y, z=0.05))

        # close back to apex
        cone_marker.points.append(Point(x=bot_world_x, y=bot_world_y, z=0.05))

        self.marker_pub.publish(cone_marker)

        self.get_logger().info(f"Published gap goal at pixel {goal_pixel}, map ({goal_world_x:.2f}, {goal_world_y:.2f}), gap {min_dist:.2f}")

    def points_in_cone(self, contours):
        """
        Return dict: contour_index -> list of (x,y) points (pixel/map indices)
        Points are **in the same coordinate space** as the contour points (image pixels).
        """
        half_angle = np.deg2rad(self.cone_angle / 2.0)
        inside_points = {}

        for i, contour in enumerate(contours):
            pts_in_cone = []
            for point in contour:
                x, y = point[0]        # pixel coords (int)
                dx = float(x) - float(self.bot_x)
                dy = float(y) - float(self.bot_y)
                dist = np.hypot(dx, dy)
                if dist > self.cone_radius:
                    continue

                angle_to_point = np.arctan2(dy, dx)  # radians
                # smallest signed angle difference
                angle_diff = np.arctan2(np.sin(angle_to_point - self.heading),
                                        np.cos(angle_to_point - self.heading))
                if abs(angle_diff) <= half_angle:
                    pts_in_cone.append((float(x), float(y)))

            if pts_in_cone:
                inside_points[i] = pts_in_cone

        return inside_points

    def points_in_cone_world(self, contours, origin_x, origin_y, resolution, bot_world_x, bot_world_y, chosen_heading):
        """
        Build dict: contour_index -> list of (x_pixel,y_pixel) that lie within the cone.
        Tests are done in WORLD coordinates so heading (odom radians) and cone geometry align.
        """
        half_angle = np.deg2rad(self.cone_angle / 2.0)
        inside_points = {}

        for i, contour in enumerate(contours):
            pts_in_cone = []
            for point in contour:
                px, py = point[0]    # pixel coords from OpenCV (x right, y down)

                # convert pixel -> world (same transform used for contour markers & goal)
                wx = origin_x + float(px) * resolution
                wy = origin_y + float(py) * resolution

                dx = wx - bot_world_x
                dy = wy - bot_world_y
                dist = np.hypot(dx, dy)

                # cone radius is stored in pixels; convert to meters for world test
                if dist > (self.cone_radius * resolution):
                    continue

                # angle from robot to the point (world frame)
                angle_to_point = math.atan2(dy, dx)  # radians

                # smallest signed angle difference between robot heading and point angle
                angle_diff = math.atan2(
                    math.sin(angle_to_point - float(chosen_heading)),
                    math.cos(angle_to_point - float(chosen_heading))
                )

                if abs(angle_diff) <= half_angle:
                    pts_in_cone.append((float(px), float(py)))

            if pts_in_cone:
                inside_points[i] = pts_in_cone

        return inside_points


    def find_goal_from_contours(self, points_dict):
        """
        points_dict: {contour_id: [(x,y), ...], ...}
        Returns:
            None           -> if only one contour (we ignore single-contour as obstacle)
            (goal, pair, min_dist) -> goal is (x,y) midpoint in pixel coords; pair is (pA, pB)
        Logic:
          - If only one contour present -> return None (treat as obstacle).
          - If >= 2 contours -> find the pair of points (from two different contours) with minimum distance,
            and return the midpoint of that closest pair.
        """
        contour_ids = list(points_dict.keys())

        if len(contour_ids) == 0:
            return None

        if len(contour_ids) == 1:
            # All points from a single contour -> we're looking directly at an obstacle, ignore.
            return None

        closest_pair = None
        min_dist = float('inf')

        # Check all pairs of distinct contours
        for id1, id2 in combinations(contour_ids, 2):
            pts1 = np.array(points_dict[id1])  # shape (N1, 2)
            pts2 = np.array(points_dict[id2])  # shape (N2, 2)
            if pts1.size == 0 or pts2.size == 0:
                continue

            # Efficient pairwise distance calculation
            diff = pts1[:, None, :] - pts2[None, :, :]  # (N1, N2, 2)
            dists = np.linalg.norm(diff, axis=2)       # (N1, N2)
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            dist = dists[idx]
            if dist < min_dist:
                min_dist = float(dist)
                closest_pair = (pts1[idx[0]], pts2[idx[1]])

        if closest_pair is None:
            return None

        # midpoint of the closest pair
        goal = (closest_pair[0] + closest_pair[1]) / 2.0
        return (goal, closest_pair, min_dist)


def main(args=None):
    rclpy.init(args=args)
    node = ContourPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
