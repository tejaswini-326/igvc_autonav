import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
from sklearn.cluster import DBSCAN
import math
from sklearn.decomposition import PCA
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String, ColorRGBA
from nav_msgs.msg import Odometry
import cv2
import time
import ctypes


MIN_CLUSTERING_DISTANCE = 0.6
MIN_CLUSTERING_POINTS = 20

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')
        self.subscription_white = self.create_subscription(
            PointCloud2,
            '/igvc/white_points',
            self.white_pointcloud_callback,
            10
        )
        self.subsctrption_yellow= self.create_subscription(
            PointCloud2,
            '/igvc/yellow_points',
            self.yellow_pointcloud_callback,
            10
        )
        self.white_curve_pub = self.create_publisher(MarkerArray, '/lane_fitted_white', 10)
        self.yellow_curve_pub = self.create_publisher(MarkerArray, '/lane_fitted_yellow', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/lane_visualization', 10)
        
        self.white_pub = self.create_publisher(PointCloud2, "/white_lane_points", 10)
        self.yellow_pub = self.create_publisher(PointCloud2, "/yellow_lane_points", 10)    
        self.timer_period = 0.033  # 30 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.yellow_points_history = []
        self.history_length = 1  # Keep last 5 frames of yellow points
        
        self.white_msg = None
        self.yellow_msg = None
        self.closest_y = 0.0

    def extract_xyz(self, msg):
        return [
            [x, y, z]
            for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        ]
    
        # FUNCTION TO PUBLISH STUFF FOR LANE VISUALISATION IN RVIZ
    def publish_lane_visualization(self, msg, target_point, cluster_curves, white_ground_points, yellow_ground_points):
        marker_array = MarkerArray()
        white_markers = []
        yellow_closest = 0


        for i, (label, coeffs, color_type, cluster_xy) in enumerate(cluster_curves):
            curve_marker = Marker()
            curve_marker.header.frame_id = msg.header.frame_id
            curve_marker.header.stamp = self.get_clock().now().to_msg()
            curve_marker.ns = "lane_curves"
            curve_marker.type = Marker.LINE_STRIP
            curve_marker.action = Marker.ADD
            curve_marker.scale.x = 0.05
            curve_marker.color.a = 1.0

            if cluster_xy is not None and len(cluster_xy) > 0:
                x_vals = cluster_xy[:, 0]
                x_min, x_max = np.min(x_vals), np.max(x_vals)
                x_line = np.linspace(x_min, x_max, 50)

                distances = np.linalg.norm(cluster_xy, axis=1)
                closest_index = np.argmin(distances)
                closest_y = cluster_xy[closest_index][1]

            else:
                x_line = np.linspace(0.0, 4.0, 50)  # fallback
                closest_y = 0.0

            a, b, c = coeffs
            y_values = []
            for x_val in x_line:
                y_val = a * x_val**2 + b * x_val + c
                y_values.append(y_val)
                pt = Point()
                pt.x = float(x_val)
                pt.y = float(y_val)
                pt.z = -1.35  # Ground level
                curve_marker.points.append(pt)

            if color_type == 'white':
                avg_y = closest_y
                white_markers.append((closest_y, curve_marker))
            else:  # yellow
                curve_marker.color.r = 0.0
                curve_marker.color.g = 1.0
                curve_marker.color.b = 0.0
                curve_marker.id = 1
                marker_array.markers.append(curve_marker)
                yellow_closest= closest_y
                

        for point_y, marker in white_markers:
            if yellow_closest > point_y:
                marker.id = 2
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.id = 0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            marker_array.markers.append(marker)

        print("length of marker array: ")
        print(len(marker_array.markers))
        self.markers_pub.publish(marker_array)

        
    def white_pointcloud_callback(self, msg):
        self.white_msg=msg
    
    def yellow_pointcloud_callback(self, msg):
        self.yellow_msg=msg
    
    def timer_callback(self):
        if self.white_msg is None or self.yellow_msg is None:
            return  # Wait until we have both clouds

        # Use latest messages
        white_msg = self.white_msg
        yellow_msg = self.yellow_msg

        # Extract and store points
        self.white_ground_points = self.extract_xyz(white_msg)
        self.yellow_ground_points = self.extract_xyz(yellow_msg)

        msg = white_msg  # Use white_msg for consistent header (needed for publishing)
    
        # ==== Yellow History Buffer ====
        # self.yellow_points_history.append(self.yellow_ground_points)
        # if len(self.yellow_points_history) > self.history_length:
        #     self.yellow_points_history.pop(0)

        # # Combine history
        # combined_yellow_points = [p for frame in self.yellow_points_history for p in frame]

        start = time.time()

        # ==== Spatial Dilation ====
        dilation_range = 0.02  
        dilated_points = []
        for p in self.yellow_ground_points:
            x, y, z = p
            dilated_points.extend([
                [x, y, z],
                [x + dilation_range, y, z],
                [x - dilation_range, y, z],
                [x, y + dilation_range, z],
                [x, y - dilation_range, z]
            ])

        # Final yellow points used for clustering and publishing
        final_yellow_points = dilated_points

        # self.get_logger().info(f"[Benchmark] Yellow Dilation took {time.time() - start:.3f} sec")
        # self.get_logger().info(f"no of final yellow ground points is : {len(final_yellow_points)}")

        # === WHITE DBSCAN and clustering ===
        start = time.time()
        points_np_white = np.array(self.white_ground_points)
        clustered_white_points = []
        cluster_curves = []
        white_cluster_centers = []  # CHANGE 1: Store white cluster centers

        if len(points_np_white) >= MIN_CLUSTERING_POINTS:
            points_xy_white = points_np_white[:, :2]
            clustering_white = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy_white)
            labels_white = clustering_white.labels_
            unique_labels_white = set(labels_white)

            # Compute yellow stats (used for proximity filtering)
            yellow_y_mean = None
            yellow_point_count = 0
            if len(final_yellow_points) >= 10:
                y_vals_yellow = np.array(final_yellow_points)[:, 1]
                yellow_y_mean = np.mean(y_vals_yellow)
                yellow_point_count = len(final_yellow_points)

            all_cluster_infos = []
            for label in unique_labels_white:
                if label == -1:
                    continue

                cluster_indices = np.where(labels_white == label)[0]
                cluster_points = points_np_white[cluster_indices]
                points_xy_cluster = cluster_points[:, :2]
                num_white_pts = len(points_xy_cluster)

                if num_white_pts < 100:
                    continue

                # PCA for elongation
                pca = PCA(n_components=2)
                pca.fit(points_xy_cluster)
                eigenvalues = pca.explained_variance_ratio_
                elongation_ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] != 0 else float('inf')

                if elongation_ratio < 2.0:
                    # self.get_logger().info(f"Skipping white cluster {label} (pothole-like): elongation_ratio = {elongation_ratio:.2f}")
                    continue

                center_y_white = np.mean(points_xy_cluster[:, 1])
                if yellow_y_mean is not None:
                    if abs(center_y_white - yellow_y_mean) < 0.5 and num_white_pts < yellow_point_count and num_white_pts < 180:
                        # self.get_logger().info(f"Skipping white cluster {label} near yellow (y={yellow_y_mean:.2f}) with only {num_white_pts} pts")
                        continue

                # Passed all checks: add points to publish
                clustered_white_points.extend(cluster_points.tolist())

                # Curve fit
                coeffs = np.polyfit(points_xy_cluster[:, 0], points_xy_cluster[:, 1], deg=2)

                # Compute distances from origin for all points
                distances = np.linalg.norm(points_xy_cluster, axis=1)
                # Get indices of the 100 closest points
                closest_indices = np.argsort(distances)[:100]
                # Extract those points
                subset = points_xy_cluster[closest_indices]
                center_x = np.mean(subset[:, 0])
                center_y = np.mean(subset[:, 1])
                white_cluster_centers.append((center_x, center_y))  # CHANGE 2: Store center coordinates
                # self.get_logger().info(f"White cluster {label}: center = ({center_x:.2f}, {center_y_white:.2f}), points = {num_white_pts}")
                # cluster_curves.append((label, coeffs, 'white', points_xy_cluster))
                all_cluster_infos.append((label, coeffs, 'white', points_xy_cluster, center_y))

            if len(white_cluster_centers) > 0:
                y_values = [cy for _, cy in white_cluster_centers]
                leftmost_white_x = min(y_values)
                rightmost_white_x = max(y_values)
            else:
                # self.get_logger().warn("No white cluster centers found — setting left/right bounds to default")
                leftmost_white_x = -1.5  # fallback default
                rightmost_white_x = 1.5

            for cy in [c[1] for c in white_cluster_centers]:
                if cy < 0:
                    if cy > leftmost_white_x:
                        leftmost_white_x = cy
                elif cy > 0:
                    if cy < rightmost_white_x:
                        rightmost_white_x = cy
            # self.get_logger().info(f"leftmost_white cluster = {leftmost_white_x} and rightmost white cluster = {rightmost_white_x}")
            
            # Apply filtering logic based on cluster centers
            if len(white_cluster_centers) >= 2 and leftmost_white_x < 0 and rightmost_white_x > 0:
                # self.get_logger().info("Filtering to only closest left/right white clusters")
                for cluster_info in all_cluster_infos:
                    label, coeffs, color, pts, cy = cluster_info
                    if abs(cy - leftmost_white_x) < 1e-3 or abs(cy - rightmost_white_x) < 1e-3:
                        cluster_curves.append((label, coeffs, color, pts))

            elif len(white_cluster_centers) >= 2 and leftmost_white_x > 0:
                # self.get_logger().info("Appending only the closest white cluster on right side (cy > 0)")
                min_cy = float('inf')
                closest_cluster = None
                for cluster_info in all_cluster_infos:
                    label, coeffs, color, pts, cy = cluster_info
                    if cy > 0 and cy < min_cy:
                        min_cy = cy
                        closest_cluster = (label, coeffs, color, pts)
                if closest_cluster:
                    cluster_curves.append(closest_cluster)

            elif len(white_cluster_centers) >= 2 and rightmost_white_x < 0:
                # self.get_logger().info("Appending only the closest white cluster on left side (cy < 0)")
                max_cy = -float('inf')
                closest_cluster = None
                for cluster_info in all_cluster_infos:
                    label, coeffs, color, pts, cy = cluster_info
                    if cy < 0 and cy > max_cy:
                        max_cy = cy
                        closest_cluster = (label, coeffs, color, pts)
                if closest_cluster:
                    cluster_curves.append(closest_cluster)

            else:
                # self.get_logger().info("Appending all white clusters (did not meet left/right condition)")
                for cluster_info in all_cluster_infos:
                    label, coeffs, color, pts, _ = cluster_info
                    cluster_curves.append((label, coeffs, color, pts))



            # === Publish filtered white points only ===
        if len(clustered_white_points) > 0:
            white_msg = pc2.create_cloud_xyz32(msg.header, clustered_white_points)
            self.white_pub.publish(white_msg)
        # self.get_logger().info(f"[Benchmark] White DBSCAN took {time.time() - start:.3f} sec")

        # === YELLOW DBSCAN and clustering ===
        start = time.time()
        points_np_yellow = np.array(final_yellow_points)
        clustered_yellow_points = []
        if len(points_np_yellow) >= MIN_CLUSTERING_POINTS:
            points_xy_yellow = points_np_yellow[:, :2]
            clustering_yellow = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy_yellow)
            labels_yellow = clustering_yellow.labels_

            # CHANGE 3: Modified logic - filter based on white cluster count
            if len(white_cluster_centers) >= 2 and rightmost_white_x > 0 and leftmost_white_x < 0:
                # Find yellow clusters that are between the white clusters
                unique_labels_yellow = set(labels_yellow)
                between_clusters_points = []

                for label in unique_labels_yellow:
                    if label == -1:
                        continue

                    cluster_indices = np.where(labels_yellow == label)[0]
                    cluster_points = points_np_yellow[cluster_indices]
                    cluster_center_x = np.mean(cluster_points[:, 1])

                    # Check if this yellow cluster is between white clusters
                    if leftmost_white_x <= cluster_center_x <= rightmost_white_x:
                        between_clusters_points.extend(cluster_points.tolist())
                        # self.get_logger().info(f"Yellow cluster {label} is between white clusters at x={cluster_center_x:.2f}")

                clustered_yellow_points = np.array(between_clusters_points) if between_clusters_points else np.array([])

            elif len(cluster_curves) == 1:
                # self.get_logger().info("Only one white cluster curve found, using closest yellow cluster to origin")
                unique_labels_yellow = set(labels_yellow)
                closest_cluster = None
                min_dist = float('inf')

                for label in unique_labels_yellow:
                    if label == -1:
                        continue

                    cluster_indices = np.where(labels_yellow == label)[0]
                    cluster_points = points_np_yellow[cluster_indices]

                    if cluster_points.size == 0:
                        continue

                    mean_x = np.mean(cluster_points[:, 0])
                    mean_y = np.mean(cluster_points[:, 1])
                    dist = np.sqrt(mean_x**2 + mean_y**2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = cluster_points

                clustered_yellow_points = closest_cluster if closest_cluster is not None else np.array([])

            else:
                # CHANGE 4: If less than 2 white clusters, combine all yellow clusters (original behavior)
                clustered_yellow_points = points_np_yellow[labels_yellow != -1]
                # self.get_logger().info(f"Only {len(white_cluster_centers)} white cluster(s) found, using all yellow clusters")


            if len(clustered_yellow_points) > 0:
                yellow_msg = pc2.create_cloud_xyz32(msg.header, clustered_yellow_points.tolist())
                self.yellow_pub.publish(yellow_msg)

            unique_labels_yellow = set(labels_yellow)
            n_clusters_y = len(unique_labels_yellow) - (1 if -1 in labels_yellow else 0)
        #     self.get_logger().info(f"The number of yellow clusters : {n_clusters_y}")
        # self.get_logger().info(f"[Benchmark] Yellow DBSCAN took {time.time() - start:.3f} sec")

        # === Yellow Curve Fitting: Single global fit on filtered yellow points ===
        # CHANGE 5: Curve fitting for yellow points (filtered or all based on white cluster count)
        if len(clustered_yellow_points) >= 10:
            x_vals_y = clustered_yellow_points[:, 0]
            y_vals_y = clustered_yellow_points[:, 1]
            coeffs_yellow = np.polyfit(x_vals_y, y_vals_y, deg=2)
            cluster_curves.append(('yellow_global', coeffs_yellow, 'yellow', clustered_yellow_points[:, :2]))
        else:
            self.get_logger().info(f"Not enough yellow points for curve fitting and no of points are: {len(clustered_yellow_points)}")


        # === Final Lane Visualization ===
        start = time.time()
        self.publish_lane_visualization(msg, None, cluster_curves, self.white_ground_points, final_yellow_points)
        # self.get_logger().info(f"[Benchmark] Marker publishing took {time.time() - start:.3f} sec")

        

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()