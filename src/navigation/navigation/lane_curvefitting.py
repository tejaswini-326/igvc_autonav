import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time

# ================= CONFIGURABLE CONSTANTS =================
VERBOSE_UNNESSARY_THINGS = False
MIN_CLUSTERING_DISTANCE = 0.6
MIN_CLUSTERING_POINTS = 20
MY_HZ = 30

# thresholds
MIN_X_ALLOWED = -0.2             # drop points behind the robot
MAX_DISPLAY_X = 8.0              # max forward distance
MARKER_LIFETIME = 0.2            # RViz marker lifetime
MIN_POINTS_PER_CLUSTER = 100     # minimum cluster size


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')
        self.subscription_white = self.create_subscription(
            PointCloud2,
            '/igvc/white_points',
            self.white_pointcloud_callback,
            10
        )

        self.white_curve_pub = self.create_publisher(MarkerArray, '/lane_fitted_white', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/lane_visualization', 10)
        self.white_pub = self.create_publisher(PointCloud2, "/white_lane_points", 10)

        self.timer = self.create_timer(1.0 / MY_HZ, self.timer_callback)

        self.white_msg = None

    def extract_xyz(self, msg):
        return [
            [x, y, z]
            for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        ]

    def publish_lane_visualization(self, msg, cluster_curves, white_ground_points):
        marker_array = MarkerArray()
        for i, (label, coeffs, color_type, cluster_xy) in enumerate(cluster_curves):
            curve_marker = Marker()
            curve_marker.header.frame_id = msg.header.frame_id
            curve_marker.header.stamp = Time(sec=0, nanosec=0)
            curve_marker.ns = "lane_curves"
            curve_marker.type = Marker.LINE_STRIP
            curve_marker.action = Marker.ADD
            curve_marker.scale.x = 0.05
            curve_marker.color.a = 1.0
            curve_marker.lifetime.sec = int(MARKER_LIFETIME)
            curve_marker.lifetime.nanosec = int((MARKER_LIFETIME - int(MARKER_LIFETIME)) * 1e9)

            if cluster_xy is not None and len(cluster_xy) > 0:
                x_vals = cluster_xy[:, 0]
                x_min, x_max = np.min(x_vals), np.max(x_vals)
                x_line = np.linspace(x_min, x_max, 50)
            else:
                x_line = np.linspace(0.0, 4.0, 50)

            a, b, c = coeffs
            for x_val in x_line:
                y_val = a * x_val**2 + b * x_val + c
                pt = Point()
                pt.x = float(x_val)
                pt.y = float(y_val)
                pt.z = -1.35
                curve_marker.points.append(pt)

            # color white lanes blue
            curve_marker.color.r = 0.0
            curve_marker.color.g = 0.0
            curve_marker.color.b = 1.0
            curve_marker.id = i
            marker_array.markers.append(curve_marker)

        self.markers_pub.publish(marker_array)
        self.white_curve_pub.publish(marker_array)

    def white_pointcloud_callback(self, msg):
        self.white_msg = msg

    def timer_callback(self):
        if self.white_msg is None:
            empty_markers = MarkerArray()
            self.markers_pub.publish(empty_markers)
            self.white_curve_pub.publish(empty_markers)
            return

        white_msg = self.white_msg
        raw_white_points = self.extract_xyz(white_msg)
        if not raw_white_points:
            empty_markers = MarkerArray()
            self.markers_pub.publish(empty_markers)
            self.white_curve_pub.publish(empty_markers)
            return

        points_np_white = np.array(raw_white_points)

        # filter out behind and too far ahead
        mask = (points_np_white[:, 0] >= MIN_X_ALLOWED) & (points_np_white[:, 0] <= MAX_DISPLAY_X)
        points_np_white = points_np_white[mask]

        clustered_white_points = []
        cluster_curves = []

        if len(points_np_white) >= MIN_CLUSTERING_POINTS:
            points_xy_white = points_np_white[:, :2]
            clustering_white = DBSCAN(
                eps=MIN_CLUSTERING_DISTANCE,
                min_samples=MIN_CLUSTERING_POINTS
            ).fit(points_xy_white)

            labels_white = clustering_white.labels_
            unique_labels_white = set(labels_white)

            # collect valid clusters
            clusters = []
            for label in unique_labels_white:
                if label == -1:
                    continue
                cluster_indices = np.where(labels_white == label)[0]
                cluster_points = points_np_white[cluster_indices]
                if len(cluster_points) < MIN_POINTS_PER_CLUSTER:
                    continue
                clusters.append(cluster_points)

            CENTER_THRESHOLD = 0.3   # ignore lanes starting too close to center

            # --- Pick longest cluster on each side (at most 2 total) ---
            left_clusters = []
            right_clusters = []

            for c in clusters:
                # rear-most point (smallest x)
                idx_start = np.argmin(c[:, 0])
                x_start, y_start = c[idx_start, 0], c[idx_start, 1]

                # ignore clusters starting too close to centerline
                if abs(y_start) < CENTER_THRESHOLD:
                    continue

                if y_start < 0:
                    left_clusters.append(c)
                else:
                    right_clusters.append(c)

            def pick_longest(cluster_list):
                if not cluster_list:
                    return None
                return max(cluster_list, key=lambda c: np.max(c[:, 0]) - np.min(c[:, 0]))

            final_clusters = []
            left_best = pick_longest(left_clusters)
            right_best = pick_longest(right_clusters)
            if left_best is not None:
                final_clusters.append(left_best)
            if right_best is not None:
                final_clusters.append(right_best)



            # publish chosen clusters
            for fc in final_clusters:
                clustered_white_points.extend(fc.tolist())
                coeffs = np.polyfit(fc[:, 0], fc[:, 1], deg=2)
                cluster_curves.append(("final", coeffs, 'white', fc[:, :2]))

            if len(clustered_white_points) > 0:
                white_msg = pc2.create_cloud_xyz32(white_msg.header, clustered_white_points)
                self.white_pub.publish(white_msg)
        else:
            empty_markers = MarkerArray()
            self.markers_pub.publish(empty_markers)
            self.white_curve_pub.publish(empty_markers)
            return

        self.publish_lane_visualization(white_msg, cluster_curves, points_np_white)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
