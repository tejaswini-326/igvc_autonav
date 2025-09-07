#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN

LOOK_AHEAD_DISTANCE = 8.0
LANE_WIDTH = 3.0   # fixed lane width

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')

        # Parameters
        self.lookahead_distance = LOOK_AHEAD_DISTANCE   
        self.lane_width = LANE_WIDTH
        self.goal_tolerance = 0.5       # meters

        # State
        self.current_goal = None
        self.robot_pose = None
        self.costmap = None
        self.white_points = None
        self.last_marker = None   # <-- store last search area marker

        # Subscribers
        self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)
        self.create_subscription(PointCloud2, '/white_lane_points', self.white_points_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_point', 10)
        self.marker_pub = self.create_publisher(Marker, '/search_area_marker', 10)

        # Timer to republish last marker at 1 Hz
        self.create_timer(1.0, self.republish_marker)

    # -------------------
    # Callbacks
    # -------------------
    def odom_cb(self, msg: Odometry):
        self.robot_pose = msg.pose.pose

        if self.current_goal is None:
            self.compute_next_goal()
        else:
            dx = self.robot_pose.position.x - self.current_goal.pose.position.x
            dy = self.robot_pose.position.y - self.current_goal.pose.position.y
            if np.hypot(dx, dy) < self.goal_tolerance:
                self.get_logger().info("Reached goal → computing next")
                self.compute_next_goal()

    def costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg

    def white_points_cb(self, msg: PointCloud2):
        self.white_points = np.array([[p[0], p[1]] for p in pc2.read_points(msg, skip_nans=True)])

    def republish_marker(self):
        """Keep publishing the last marker so it stays visible in RViz"""
        if self.last_marker is not None:
            self.last_marker.header.stamp = self.get_clock().now().to_msg()
            self.marker_pub.publish(self.last_marker)

    # -------------------
    # Core function
    # -------------------
    def compute_next_goal(self):
        if self.robot_pose is None or self.white_points is None or self.costmap is None:
            return

        lookahead_x = self.robot_pose.position.x + self.lookahead_distance

        if len(self.white_points) < 3:
            self.get_logger().warn("Not enough points for clustering!")
            return

        clustering = DBSCAN(eps=0.5, min_samples=3).fit(self.white_points)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            self.get_logger().warn("No clusters found!")
            return
        elif len(unique_labels) == 1:
            cluster_pts = self.white_points[labels == list(unique_labels)[0]]
            idx = np.argmin(np.abs(cluster_pts[:, 0] - lookahead_x))
            lane_pt = cluster_pts[idx]

            if self.robot_pose.position.y < lane_pt[1]:
                left_pt = lane_pt
                right_pt = (left_pt[0], left_pt[1] + self.lane_width)
            else:
                right_pt = lane_pt
                left_pt = (right_pt[0], right_pt[1] - self.lane_width)
        else:
            cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            main_labels = [cluster_sizes[0][0], cluster_sizes[1][0]]

            pts0 = self.white_points[labels == main_labels[0]]
            pts1 = self.white_points[labels == main_labels[1]]

            pt0 = pts0[np.argmin(np.abs(pts0[:, 0] - lookahead_x))]
            pt1 = pts1[np.argmin(np.abs(pts1[:, 0] - lookahead_x))]

            if pt0[1] < pt1[1]:
                left_pt, right_pt = pt0, pt1
            else:
                left_pt, right_pt = pt1, pt0

        mid_x = (left_pt[0] + right_pt[0]) / 2.0
        mid_y = (left_pt[1] + right_pt[1]) / 2.0

        res = self.costmap.info.resolution
        origin = self.costmap.info.origin.position
        w = self.costmap.info.width
        h = self.costmap.info.height

        min_x = min(left_pt[0], right_pt[0]) - 0.5
        max_x = max(left_pt[0], right_pt[0]) + 0.5
        min_y = min(left_pt[1], right_pt[1]) - 0.5
        max_y = max(left_pt[1], right_pt[1]) + 0.5

        gx_min = max(0, int((min_x - origin.x) / res))
        gx_max = min(w - 1, int((max_x - origin.x) / res))
        gy_min = max(0, int((min_y - origin.y) / res))
        gy_max = min(h - 1, int((max_y - origin.y) / res))

        # -------------------
        # Create and store search area marker
        # -------------------
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "search_area"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = (min_x + max_x) / 2.0
        marker.pose.position.y = (min_y + max_y) / 2.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = (max_x - min_x)
        marker.scale.y = (max_y - min_y)
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.3
        self.marker_pub.publish(marker)
        self.last_marker = marker  # <-- save for republishing

        # collect candidate cells (exclude unknown -1)
        candidates = []
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                idx = gy * w + gx
                val = self.costmap.data[idx]
                if val >= 0:  # skip unknown
                    wx = gx * res + origin.x + res / 2.0
                    wy = gy * res + origin.y + res / 2.0
                    candidates.append((wx, wy, val))

        if not candidates:
            self.get_logger().warn("No valid cells in search area!")
            return

        # choose point with lowest cost, break ties by closeness to midpoint
        goal_x, goal_y, _ = min(
            candidates,
            key=lambda p: (p[2], np.hypot(p[0] - mid_x, p[1] - mid_y))
        )

        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.w = 1.0

        self.goal_pub.publish(goal)
        self.current_goal = goal
        self.get_logger().info(f"New goal at ({goal_x:.2f}, {goal_y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
