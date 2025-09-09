#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from itertools import combinations
import cv2
import math
import tf_transformations

from tf2_ros import Buffer, TransformListener

from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32


class ContourPublisher(Node):
    def __init__(self):
        super().__init__("contour_publisher")

        # Parameters (can be changed with ros2 param set)
        self.declare_parameter("publish_contours", True)
        self.declare_parameter("publish_contours_every_n", 5)
        self.declare_parameter("contour_downsample_max_points", 400)
        self.declare_parameter("ignore_outermost_contour", True)
        self.declare_parameter("cone_angle_deg", 35.0)
        self.declare_parameter("cone_radius_pixels", 60)
        self.declare_parameter("publish_cone", True)
        self.declare_parameter("publish_goal_when_missing", True)
        self.declare_parameter("tilt_angle_deg", 20.0)  # how much to tilt from opposite direction
        self.declare_parameter("tilt_based_on_motion", True)  # pick tilt side based on odom velocity
        self.declare_parameter("tilt_speed_threshold", 0.05)  # m/s threshold to consider motion meaningful

        # read params into attributes
        self.publish_contours = self.get_parameter("publish_contours").get_parameter_value().bool_value
        self.publish_contours_every_n = int(self.get_parameter("publish_contours_every_n").get_parameter_value().integer_value)
        self.contour_downsample_max_points = int(self.get_parameter("contour_downsample_max_points").get_parameter_value().integer_value)
        self.ignore_outermost_contour = self.get_parameter("ignore_outermost_contour").get_parameter_value().bool_value
        self.cone_angle = float(self.get_parameter("cone_angle_deg").get_parameter_value().double_value)
        self.cone_radius = int(self.get_parameter("cone_radius_pixels").get_parameter_value().integer_value)
        self.publish_cone = self.get_parameter("publish_cone").get_parameter_value().bool_value
        self.publish_goal_when_missing = self.get_parameter("publish_goal_when_missing").get_parameter_value().bool_value
        self.tilt_angle_deg = float(self.get_parameter("tilt_angle_deg").get_parameter_value().double_value)
        self.tilt_based_on_motion = self.get_parameter("tilt_based_on_motion").get_parameter_value().bool_value
        self.tilt_speed_threshold = float(self.get_parameter("tilt_speed_threshold").get_parameter_value().double_value)

        # Subscribe to costmap and heading sources
        self.sub = self.create_subscription(OccupancyGrid, "/costmap", self.callback, 10)
        self.heading_sub = self.create_subscription(Float32, "/heading_angle", self.heading_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # Publishers
        # NOTE: publish PoseStamped on /new_goal now
        self.marker_pub = self.create_publisher(Marker, "/contours_marker", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/new_goal", 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # basic robot/map params
        self.bot_x = 130
        self.bot_y = 150

        # heading sources
        self.heading = 0.0
        self.heading_time = None
        self.odom_heading = 0.0
        self.odom_time = None

        # store linear velocity from odom for motion-based tilt selection
        self.odom_vx = 0.0
        self.odom_vy = 0.0

        # internal counters / timers
        self.cb_count = 0

        # persist last goal so we can republish if new goal not found
        self.last_goal_pixel = None
        self.last_goal_world = None
        self.last_goal_time = None

    def heading_cb(self, msg: Float32):
        try:
            self.heading = float(msg.data)
            self.heading_time = self.get_clock().now()
        except Exception:
            pass

    def odom_cb(self, msg: Odometry):
        try:
            q = msg.pose.pose.orientation
            _, _, yaw = tf_transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
            self.odom_heading = float(yaw)
            self.odom_time = self.get_clock().now()
            # store linear velocity (robot body frame expressed in odom frame)
            self.odom_vx = float(msg.twist.twist.linear.x)
            self.odom_vy = float(msg.twist.twist.linear.y)
        except Exception:
            pass

    def callback(self, msg: OccupancyGrid):
        # Increment callback counter (used for throttling visualizations)
        self.cb_count += 1

        width, height = msg.info.width, msg.info.height
        resolution = msg.info.resolution
        origin_x, origin_y = msg.info.origin.position.x, msg.info.origin.position.y

        # Convert occupancy data into numpy image
        img = np.array(msg.data, dtype=np.int16).reshape((height, width))
        img[img < 0] = 0
        img_u8 = img.astype(np.uint8)

        # Build corridor mask (tweak thresholds if needed)
        mask1 = cv2.inRange(img_u8, 0, 20)
        mask2 = cv2.inRange(img_u8, 75, 255)
        corridor_mask = cv2.bitwise_or(mask1, mask2)

        # find contours (OpenCV version differences handled)
        contours_result = cv2.findContours(corridor_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_result) == 3:
            _, contours, hierarchy = contours_result
        else:
            contours, hierarchy = contours_result

        if not contours:
            self.get_logger().debug("No contours found in costmap.")
            # no contours: still republish last goal if allowed
            if self.last_goal_world is not None and self.publish_goal_when_missing:
                self._publish_goal_marker_and_pose(msg.header.frame_id, self.last_goal_world, used_heading=None)
            return

        # Optionally ignore the outermost (largest area) contour which often represents a border/capsule
        if self.ignore_outermost_contour and len(contours) > 1:
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = int(np.argmax(areas))
            contours_filtered = [c for i, c in enumerate(contours) if i != max_idx]
            contours = contours_filtered

        # Publish simplified contours (throttled) if enabled
        if self.publish_contours and (self.cb_count % max(1, self.publish_contours_every_n) == 0):
            # Delete prior contour namespace markers once per visualization update
            del_m = Marker()
            del_m.ns = "contours"
            del_m.action = Marker.DELETEALL
            del_m.header.frame_id = msg.header.frame_id
            del_m.header.stamp = self.get_clock().now().to_msg()
            self.marker_pub.publish(del_m)

            for idx, cnt in enumerate(contours):
                # Reduce vertex count with approxPolyDP (epsilon proportional to perimeter)
                peri = cv2.arcLength(cnt, True)
                epsilon = max(1.0, 0.003 * peri)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Further downsample if too many points
                pts = approx.reshape(-1, 2)
                max_pts = max(4, self.contour_downsample_max_points)
                if pts.shape[0] > max_pts:
                    stride = int(np.ceil(pts.shape[0] / float(max_pts)))
                    pts = pts[::stride]

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

                for x, y in pts:
                    p = Point()
                    p.x = origin_x + float(x) * resolution
                    p.y = origin_y + float(y) * resolution
                    p.z = 0.05
                    marker.points.append(p)

                self.marker_pub.publish(marker)

        # compute robot pixel center (assumes costmap centered on robot)
        bot_px = width // 2
        bot_py = height // 2
        self.bot_x = bot_px
        self.bot_y = bot_py

        bot_world_x = origin_x + float(bot_px) * resolution
        bot_world_y = origin_y + float(bot_py) * resolution

        # Choose heading source with freshness
        now = self.get_clock().now()
        chosen_src = "none"
        chosen_heading = None
        freshness_s = 0.5

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
            chosen_heading = float(self.heading if self.heading_time is not None else self.odom_heading)
            chosen_src = "last_known"

        self.get_logger().debug(f"Heading chosen from {chosen_src}: {chosen_heading:.3f} rad")

        # --- find points inside the cone using WORLD coordinates (vectorized per contour) ---
        inside_points = self.points_in_cone_world_vectorized(
            contours, origin_x, origin_y, resolution, bot_world_x, bot_world_y, chosen_heading
        )

        used_heading_for_vis = float(chosen_heading)  # what we'll visualize (may change if we tilt)

        # if only one contour visible in cone (we're looking at a blob) -> attempt tilted search behind robot
        if len(inside_points) == 1:
            self.get_logger().info("Single contour in forward cone: attempting tilted search behind robot.")
            # compute headings to try: directly opposite, opposite + tilt, opposite - tilt
            tilt_rad = math.radians(self.tilt_angle_deg)
            forward = chosen_heading
            opposite = forward + math.pi

            # normalize helper
            def _norm(a):
                return math.atan2(math.sin(a), math.cos(a))

            left = _norm(forward + tilt_rad)
            right = _norm(forward - tilt_rad)
            opp = _norm(opposite)

            # Decide order based on motion
            if self.odom_time is not None and (now - self.odom_time).nanoseconds * 1e-9 < 0.5:
                vx, vy = self.odom_vx, self.odom_vy
                speed = math.hypot(vx, vy)
                if speed > self.tilt_speed_threshold:
                    movement_angle = math.atan2(vy, vx)
                    angle_diff = math.atan2(math.sin(movement_angle - forward),
                                            math.cos(movement_angle - forward))
                    if angle_diff > 0:
                        # moving left of heading → look right first
                        candidates = [right, left, opp]
                    else:
                        # moving right of heading → look left first
                        candidates = [left, right, opp]
                else:
                    # not moving → default order
                    candidates = [left, right, opp]
            else:
                # no odom info → default order
                candidates = [left, right, opp]


            found = None
            for th in candidates:
                try_points = self.points_in_cone_world_vectorized(
                    contours, origin_x, origin_y, resolution, bot_world_x, bot_world_y, th
                )
                # want at least two separate contours visible when searching backward
                if len(try_points) >= 2:
                    found = (try_points, th)
                    break

            if found is not None:
                inside_points, used_heading_for_vis = found[0], found[1]
                self.get_logger().info(f"Tilted search successful using heading {used_heading_for_vis:.2f} rad")
            else:
                self.get_logger().info("Tilted search failed to find >=2 contours; re-publishing last goal if available.")
                if self.last_goal_world is not None and self.publish_goal_when_missing:
                    self._publish_goal_marker_and_pose(msg.header.frame_id, self.last_goal_world, used_heading=used_heading_for_vis)
                return

        if not inside_points:
            self.get_logger().info("No contour points inside cone.")
            # republish last goal if available and requested
            if self.last_goal_world is not None and self.publish_goal_when_missing:
                self._publish_goal_marker_and_pose(msg.header.frame_id, self.last_goal_world, used_heading=used_heading_for_vis)
            return

        result = self.find_goal_from_contours(inside_points)
        if result is None:
            self.get_logger().info("Only one contour visible in cone (obstacle) or no valid pairs. Re-publishing last goal if available.")
            if self.last_goal_world is not None and self.publish_goal_when_missing:
                self._publish_goal_marker_and_pose(msg.header.frame_id, self.last_goal_world, used_heading=used_heading_for_vis)
            return

        goal_pixel, closest_pair, min_dist = result

        goal_world_x = origin_x + float(goal_pixel[0]) * resolution
        goal_world_y = origin_y + float(goal_pixel[1]) * resolution

        # store last goal
        self.last_goal_pixel = goal_pixel
        self.last_goal_world = (goal_world_x, goal_world_y)
        self.last_goal_time = self.get_clock().now()

        # Publish goal marker and PoseStamped
        self._publish_goal_marker_and_pose(msg.header.frame_id, self.last_goal_world, used_heading=used_heading_for_vis)

        # publish closest pair small markers (always publish these for debugging)
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

        # cone visualization (optional—cheap). visualize using the heading actually used to find the goal
        if self.publish_cone:
            cone_marker = Marker()
            cone_marker.header.frame_id = msg.header.frame_id
            cone_marker.header.stamp = self.get_clock().now().to_msg()
            cone_marker.ns = "cone"
            cone_marker.id = 1
            cone_marker.type = Marker.LINE_STRIP
            cone_marker.action = Marker.ADD
            cone_marker.scale.x = 0.2
            cone_marker.color.r = 1.0
            cone_marker.color.g = 0.0
            cone_marker.color.b = 0.0
            cone_marker.color.a = 1.0

            cone_radius_m = self.cone_radius * resolution
            half_angle = np.deg2rad(self.cone_angle / 2.0)

            cone_marker.points.append(Point(x=bot_world_x, y=bot_world_y, z=0.05))
            robot_heading = float(used_heading_for_vis)
            for angle_offset in np.linspace(-half_angle, half_angle, 15):
                angle = robot_heading + angle_offset
                x = bot_world_x + cone_radius_m * np.cos(angle)
                y = bot_world_y + cone_radius_m * np.sin(angle)
                cone_marker.points.append(Point(x=x, y=y, z=0.05))
            cone_marker.points.append(Point(x=bot_world_x, y=bot_world_y, z=0.05))
            self.marker_pub.publish(cone_marker)

        self.get_logger().info(f"Published gap goal at pixel {goal_pixel}, map ({goal_world_x:.2f}, {goal_world_y:.2f}), gap {min_dist:.2f}")

    def _publish_goal_marker_and_pose(self, frame_id: str, world_xy: tuple, used_heading=None):
        gx, gy = world_xy
        goal_marker = Marker()
        goal_marker.header.frame_id = frame_id
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
        goal_marker.pose.position.x = gx
        goal_marker.pose.position.y = gy
        goal_marker.pose.position.z = 0.1
        goal_marker.pose.orientation.w = 1.0
        self.marker_pub.publish(goal_marker)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = gx
        pose_msg.pose.position.y = gy
        pose_msg.pose.position.z = 0.0
        # keep orientation simple: identity (w=1). If you want a heading in the pose, set pose.orientation accordingly.
        pose_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(pose_msg)

        # optional: visualize the heading used to pick this goal as a small arrow or by republishing the cone (done elsewhere)

    def points_in_cone_world_vectorized(self, contours, origin_x, origin_y, resolution, bot_world_x, bot_world_y, chosen_heading):
        half_angle = np.deg2rad(self.cone_angle / 2.0)
        cone_radius_m = self.cone_radius * resolution
        inside_points = {}

        for i, contour in enumerate(contours):
            pts = contour.reshape(-1, 2).astype(np.float32)  # shape (N, 2)
            if pts.shape[0] == 0:
                continue

            # pixel -> world (vectorized)
            wx = origin_x + pts[:, 0] * resolution
            wy = origin_y + pts[:, 1] * resolution

            dx = wx - bot_world_x
            dy = wy - bot_world_y
            dists = np.hypot(dx, dy)

            # mask by distance first (cheap)
            mask_d = dists <= cone_radius_m
            if not np.any(mask_d):
                continue

            angles = np.arctan2(dy, dx)
            angle_diffs = np.arctan2(np.sin(angles - float(chosen_heading)), np.cos(angles - float(chosen_heading)))
            mask_ang = np.abs(angle_diffs) <= half_angle

            mask = mask_d & mask_ang
            if not np.any(mask):
                continue

            selected = pts[mask]
            # Convert back to list of tuples in pixel coordinates (float)
            inside_points[i] = [(float(x), float(y)) for x, y in selected]

        return inside_points

    def find_goal_from_contours(self, points_dict):
        contour_ids = list(points_dict.keys())
        if len(contour_ids) == 0:
            return None
        if len(contour_ids) == 1:
            return None

        closest_pair = None
        min_dist = float('inf')

        # check pairs of distinct contours
        for id1, id2 in combinations(contour_ids, 2):
            pts1 = np.array(points_dict[id1])
            pts2 = np.array(points_dict[id2])
            if pts1.size == 0 or pts2.size == 0:
                continue

            # efficient pairwise distances
            diff = pts1[:, None, :] - pts2[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            dist = dists[idx]
            if dist < min_dist:
                min_dist = float(dist)
                closest_pair = (pts1[idx[0]], pts2[idx[1]])

        if closest_pair is None:
            return None

        goal = (closest_pair[0] + closest_pair[1]) / 2.0
        return (goal, closest_pair, min_dist)

    def _norm(self, a: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    rclpy.init(args=args)
    node = ContourPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
