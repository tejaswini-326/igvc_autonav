#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import math
import random
import tf2_ros
import tf_transformations

LOOK_AHEAD_DISTANCE = 5.0
LANE_WIDTH = 3.0   # fixed lane width
OCCUPIED_THRESHOLD = 60
ROBOT_WIDTH_M = 2.2
ROBOT_RADIUS_M = ROBOT_WIDTH_M / 2.0


class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        
        # Parameters
        self.lookahead_distance = LOOK_AHEAD_DISTANCE
        self.lane_width = LANE_WIDTH
        self.goal_tolerance = 0.5

        # State
        self.current_goal = None
        self.robot_pose = None
        self.costmap = None
        self.white_points = None
        self.last_marker = None
        self.last_left_marker = None
        self.last_right_marker = None
        self.last_cluster_markers = None

        # Subscribers
        self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)
        self.create_subscription(PointCloud2, '/white_lane_points', self.white_points_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_point', 10)
        self.marker_pub = self.create_publisher(Marker, '/search_area_marker', 10)
        self.left_pt_pub = self.create_publisher(Marker, '/debug_left_pt', 10)
        self.right_pt_pub = self.create_publisher(Marker, '/debug_right_pt', 10)
        self.cluster_pub = self.create_publisher(MarkerArray, '/lane_clusters', 10)

        # TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer to republish last markers
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
        pts = np.array([[p[0], p[1], 0.0] for p in pc2.read_points(msg, skip_nans=True)])
        try:
            t = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time())
            self.white_points = np.array([self.transform_point(pt, t) for pt in pts])
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")

    def transform_point(self, pt, transform):
        # Convert quaternion to rotation matrix
        q = transform.transform.rotation
        t = transform.transform.translation
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        p = np.array(pt)
        p_transformed = R @ p + np.array([t.x, t.y, t.z])
        return p_transformed[:2]

    def republish_marker(self):
        now = self.get_clock().now().to_msg()
        if self.last_marker is not None:
            self.last_marker.header.stamp = now
            self.marker_pub.publish(self.last_marker)
        if self.last_left_marker is not None:
            self.last_left_marker.header.stamp = now
            self.left_pt_pub.publish(self.last_left_marker)
        if self.last_right_marker is not None:
            self.last_right_marker.header.stamp = now
            self.right_pt_pub.publish(self.last_right_marker)
        if self.last_cluster_markers is not None:
            for m in self.last_cluster_markers.markers:
                m.header.stamp = now
            self.cluster_pub.publish(self.last_cluster_markers)

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

        # ------------------- DBSCAN clustering -------------------
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(self.white_points)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        # ------------------- Marker publishing -------------------
        marker_array = MarkerArray()
        for label in unique_labels:
            pts = self.white_points[labels == label]
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lane_clusters"
            marker.id = int(label)
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = 0.2
            marker.color.r = random.random()
            marker.color.g = random.random()
            marker.color.b = random.random()
            marker.color.a = 1.0
            for pt in pts:
                p = Point()
                p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.0
                marker.points.append(p)
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.cluster_pub.publish(marker_array)
            self.last_cluster_markers = marker_array

        # ------------------- Select lane points -------------------
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

        # ------------------- Costmap-aware goal selection -------------------
        num_samples = 20
        candidate_points = np.linspace(left_pt, right_pt, num_samples)
        best_point = None
        best_cost = 1e6
        mid_pt = (left_pt + right_pt) / 2.0

        for pt in candidate_points:
            idx = self.world_to_costmap(pt[0], pt[1])
            if idx is None:
                continue
            mx, my = idx
            cost_idx = my * self.costmap.info.width + mx
            cost_val = self.costmap.data[cost_idx]
            if cost_val < best_cost:
                best_cost = cost_val
                best_point = pt
            elif cost_val == best_cost:
                if np.linalg.norm(pt - mid_pt) < np.linalg.norm(best_point - mid_pt):
                    best_point = pt

        if best_point is None:
            self.get_logger().warn("No valid free point in lane. Skipping goal.")
            return

        # ------------------- Publish goal -------------------
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(best_point[0])
        goal.pose.position.y = float(best_point[1])
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.current_goal = goal
        self.goal_pub.publish(goal)

    # ------------------- Helper function -------------------
    def world_to_costmap(self, x, y):
        info = self.costmap.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)
        if 0 <= mx < info.width and 0 <= my < info.height:
            return mx, my
        else:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
