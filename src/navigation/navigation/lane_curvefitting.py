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


MIN_CLUSTERING_DISTANCE = 0.3
MIN_CLUSTERING_POINTS = 20

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points_downsampled',
            self.pointcloud_callback,
            10
        )
        self.white_curve_pub = self.create_publisher(MarkerArray, '/lane_fitted_white', 10)
        self.yellow_curve_pub = self.create_publisher(MarkerArray, '/lane_fitted_yellow', 10)
        self.markers_pub = self.create_publisher(MarkerArray, '/lane_visualization', 10)
        self.white_pub = self.create_publisher(PointCloud2, "/white_lane_points", 10)
        self.yellow_pub = self.create_publisher(PointCloud2, "/yellow_lane_points", 10)    

        self.yellow_points_history = []
        self.history_length = 1  # Keep last 5 frames of yellow points

    # FUNCTION TO PUBLISH STUFF FOR LANE VISUALISATION IN RVIZ
    def publish_lane_visualization(self, msg, target_point, cluster_curves, white_ground_points, yellow_ground_points):
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = msg.header.frame_id
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Marker for white ground points (detected lane pixels)
        if white_ground_points:
            points_marker = Marker()
            points_marker.header.frame_id = msg.header.frame_id
            points_marker.header.stamp = self.get_clock().now().to_msg()
            points_marker.ns = "white_points"
            points_marker.id = 0
            points_marker.type = Marker.POINTS
            points_marker.action = Marker.ADD
            points_marker.scale.x = 0.02
            points_marker.scale.y = 0.02
            points_marker.color.r = 1.0
            points_marker.color.g = 1.0
            points_marker.color.b = 1.0
            points_marker.color.a = 0.6
            
            for point in white_ground_points:
                pt = Point()
                pt.x = float(point[0])
                pt.y = float(point[1])
                pt.z = float(point[2])
                points_marker.points.append(pt)
            
            marker_array.markers.append(points_marker)
        
        # Marker for yellow ground points (detected lane pixels)
        if yellow_ground_points:
            yellow_points_marker = Marker()
            yellow_points_marker.header.frame_id = msg.header.frame_id
            yellow_points_marker.header.stamp = self.get_clock().now().to_msg()
            yellow_points_marker.ns = "yellow_points"
            yellow_points_marker.id = 1
            yellow_points_marker.type = Marker.POINTS
            yellow_points_marker.action = Marker.ADD
            yellow_points_marker.scale.x = 0.02
            yellow_points_marker.scale.y = 0.02
            yellow_points_marker.color.r = 1.0
            yellow_points_marker.color.g = 1.0
            yellow_points_marker.color.b = 0.0
            yellow_points_marker.color.a = 0.8
            
            for point in yellow_ground_points:
                pt = Point()
                pt.x = float(point[0])
                pt.y = float(point[1])
                pt.z = float(point[2])
                yellow_points_marker.points.append(pt)
            
            marker_array.markers.append(yellow_points_marker)
        
        # Markers for fitted curves
        for i, (label, coeffs, color_type) in enumerate(cluster_curves):
            curve_marker = Marker()
            curve_marker.header.frame_id = msg.header.frame_id
            curve_marker.header.stamp = self.get_clock().now().to_msg()
            curve_marker.ns = "lane_curves"
            curve_marker.id = i + 2
            curve_marker.type = Marker.LINE_STRIP
            curve_marker.action = Marker.ADD
            curve_marker.scale.x = 0.05
            
            # Different colors for different curves and types
            if color_type == 'white':
                if i == 0:
                    curve_marker.color.r = 1.0
                    curve_marker.color.g = 0.0
                    curve_marker.color.b = 0.0
                elif i == 1:
                    curve_marker.color.r = 0.0
                    curve_marker.color.g = 1.0
                    curve_marker.color.b = 0.0
                else:
                    curve_marker.color.r = 0.0
                    curve_marker.color.g = 0.0
                    curve_marker.color.b = 1.0
            else:  # yellow
                if i == 0:
                    curve_marker.color.r = 1.0
                    curve_marker.color.g = 1.0
                    curve_marker.color.b = 0.0
                elif i == 1:
                    curve_marker.color.r = 1.0
                    curve_marker.color.g = 0.5
                    curve_marker.color.b = 0.0
                else:
                    curve_marker.color.r = 0.8
                    curve_marker.color.g = 0.8
                    curve_marker.color.b = 0.0
            
            curve_marker.color.a = 1.0
            
            # Generate curve points
            x_line = np.linspace(0.0, 4.0, 50)
            a, b, c = coeffs
            
            for x_val in x_line:
                y_val = a * x_val**2 + b * x_val + c  
                pt = Point()
                pt.x = float(x_val)
                pt.y = float(y_val)
                pt.z = -1.35  # Ground level
                curve_marker.points.append(pt)
            
            marker_array.markers.append(curve_marker)
        
        # Marker for target point
        if target_point is not None:
            target_marker = Marker()
            target_marker.header.frame_id = msg.header.frame_id
            target_marker.header.stamp = self.get_clock().now().to_msg()
            target_marker.ns = "target_point"
            target_marker.id = 100
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.scale.x = 0.2
            target_marker.scale.y = 0.2
            target_marker.scale.z = 0.2
            target_marker.color.r = 1.0
            target_marker.color.g = 1.0
            target_marker.color.b = 0.0
            target_marker.color.a = 1.0
            
            target_marker.pose.position.x = float(target_point[0])
            target_marker.pose.position.y = float(target_point[1])
            target_marker.pose.position.z = -1.3
            target_marker.pose.orientation.w = 1.0
            
            marker_array.markers.append(target_marker)
        white_array = MarkerArray()
        yellow_array = MarkerArray()

        for i, (label, coeffs, color_type) in enumerate(cluster_curves):
            curve_marker = Marker()
            curve_marker.header.frame_id = msg.header.frame_id
            curve_marker.header.stamp = self.get_clock().now().to_msg()
            curve_marker.ns = "lane_curves"
            curve_marker.id = i + 2
            curve_marker.type = Marker.LINE_STRIP
            curve_marker.action = Marker.ADD
            curve_marker.scale.x = 0.05
            curve_marker.color.a = 1.0

            # Color and curve assignment
            if color_type == 'white':
                curve_marker.color.r = 1.0 if i == 0 else 0.0
                curve_marker.color.g = 0.0 if i == 0 else (1.0 if i == 1 else 0.0)
                curve_marker.color.b = 0.0 if i == 0 else (0.0 if i == 1 else 1.0)
                white_array.markers.append(curve_marker)
            else:
                curve_marker.color.r = 1.0
                curve_marker.color.g = 1.0 if i == 0 else (0.5 if i == 1 else 0.8)
                curve_marker.color.b = 0.0
                yellow_array.markers.append(curve_marker)

            # Points along the curve
            x_line = np.linspace(0.0, 4.0, 50)
            a, b, c = coeffs
            for x_val in x_line:
                y_val = a * x_val**2 + b * x_val + c
                pt = Point()
                pt.x = float(x_val)
                pt.y = float(y_val)
                pt.z = -1.35
                curve_marker.points.append(pt)

        self.white_curve_pub.publish(white_array)
        self.yellow_curve_pub.publish(yellow_array)

        # Publish marker array
        self.markers_pub.publish(marker_array)
    
    def pointcloud_callback(self, msg):
        height = msg.height
        width = msg.width

    # ==========INSTEAD OF USING THE COMMEENTED PART BELOW (A NORAL FOR LOOP) USE NUMPY FOR BETTER SPEED....NUMPY AURA=========
        # white_ground_points = []
        # yellow_ground_points = []
        # index = 0

        # wh_h_min = 0
        # wh_h_max = 180
        # wh_s_min = 0
        # wh_s_max = 20
        # wh_v_min = 95
        # wh_v_max = 255

        # yl_h_min = 20
        # yl_h_max = 30
        # yl_s_min = 140
        # yl_s_max = 255
        # yl_v_min = 90
        # yl_v_max = 255

        # start = time.time()

        # for x, y, z, rgb in pc2.read_points(msg, field_names=('x','y','z','rgb'), skip_nans=True):
        #     # unpack RGB float
        #     rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
        #     b = rgb_int & 0xFF
        #     g = (rgb_int >> 8) & 0xFF
        #     r = (rgb_int >> 16) & 0xFF

        #     # convert BGR to HSV
        #     hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        #     h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        #     # ground-plane filter
        #     if not (-2.0 < z < -1.3 and 0.0 < x < 10.0):
        #         continue

        #     # white?
        #     if (wh_h_min <= h <= wh_h_max and
        #         wh_s_min <= s <= wh_s_max and
        #         wh_v_min <= v <= wh_v_max):
        #         white_ground_points.append([x, y, z])

        #     # yellow?
        #     elif (yl_h_min <= h <= yl_h_max and
        #         yl_s_min <= s <= yl_s_max and
        #         yl_v_min <= v <= yl_v_max):
        #         yellow_ground_points.append([x, y, z])
        #     index += 1

        # self.get_logger().info(f"[Benchmark] PointCloud parsing took {time.time() - start:.3f} sec")

        white_ground_points = []
        yellow_ground_points = []

        start = time.time()
        # PointCloud to numpy array
        cloud_points = np.fromiter(
            pc2.read_points(msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True),
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')]
        )
        cloud_points_np = np.vstack([cloud_points['x'], cloud_points['y'], cloud_points['z'], cloud_points['rgb']]).T


        if cloud_points.shape[0] == 0:
            return

        xyz = np.stack([cloud_points['x'], cloud_points['y'], cloud_points['z']], axis=-1)
        rgb_floats = cloud_points['rgb'].astype(np.float32)

        # Fast float -> uint32 conversion
        rgb_uint32 = rgb_floats.view(np.uint32)
        r = (rgb_uint32 >> 16) & 0xFF
        g = (rgb_uint32 >> 8) & 0xFF
        b = rgb_uint32 & 0xFF

        # Stack as BGR image and convert to HSV
        start = time.perf_counter()
        #hsv_image = fast_rgb_to_hsv(r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8))
        bgr_image = np.stack([b, g, r], axis=1).astype(np.uint8).reshape(-1, 1, 3)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        end= time.perf_counter()
        elapsed = end - start
        self.get_logger().info(f"[Benchmark] Block took {elapsed:.6f} seconds")
        h, s, v = hsv_image[:, 0], hsv_image[:, 1], hsv_image[:, 2]

        # Ground plane mask
        z = xyz[:, 2]
        x_vals = xyz[:, 0]
        ground_mask = (-2.0 < z) & (z < -1.3) & (0.0 < x_vals) & (x_vals < 10.0)

        # White and yellow masks
        white_mask = (
            (0 <= h) & (h <= 180) &
            (0 <= s) & (s <= 20) &
            (95 <= v) & (v <= 255)
        )

        yellow_mask = (
            (20 <= h) & (h <= 30) &
            (140 <= s) & (s <= 255) &
            (90 <= v) & (v <= 255)
        )

        # Apply masks
        white_ground_points = xyz[ground_mask & white_mask].tolist()
        yellow_ground_points = xyz[ground_mask & yellow_mask].tolist()

        self.get_logger().info(f"[Benchmark] PointCloud parsing took {time.time() - start:.3f} sec")
        self.get_logger().info(f"no of white ground points : {len(white_ground_points)} and no of yellow ground points is : {len(yellow_ground_points)}")

        # ==== Yellow History Buffer ====
        self.yellow_points_history.append(yellow_ground_points)
        if len(self.yellow_points_history) > self.history_length:
            self.yellow_points_history.pop(0)

        # Combine history
        combined_yellow_points = [p for frame in self.yellow_points_history for p in frame]

        start = time.time()

        # ==== Spatial Dilation ====
        dilation_range = 0.02  
        dilated_points = []
        for p in combined_yellow_points:
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

        self.get_logger().info(f"[Benchmark] Yellow Dilation took {time.time() - start:.3f} sec")
        self.get_logger().info(f"no of final yellow ground points is : {len(final_yellow_points)}")



        # ==== Clustering ====
        # all_lane_points = white_ground_points + final_yellow_points

        # if len(all_lane_points) < 10:
        #     self.get_logger().warn(f"Not enough lane points for detection (white: {len(white_ground_points)}, yellow: {len(final_yellow_points)})")
        #     self.publish_lane_visualization(msg, None, [], white_ground_points, final_yellow_points)
        #     return

        # points_np_all = np.array(all_lane_points)
        # points_xy_all = points_np_all[:, :2]

        # # DBSCAN on all lane points together (white + yellow)
        # clustering_all = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy_all)
        # labels_all = clustering_all.labels_

        # unique_labels_all = set(labels_all)
        # n_clusters_all = len(unique_labels_all) - (1 if -1 in labels_all else 0)
        # self.get_logger().info(f"Total lane clusters (white + yellow): {n_clusters_all}")

        # Now do white and yellow pointcloud publishing separately

        # === WHITE DBSCAN and clustering ===
        start = time.time()
        points_np_white = np.array(white_ground_points)
        clustered_white_points = []
        cluster_curves = []

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
                    self.get_logger().info(f"Skipping white cluster {label} (pothole-like): elongation_ratio = {elongation_ratio:.2f}")
                    continue

                center_y_white = np.mean(points_xy_cluster[:, 1])
                if yellow_y_mean is not None:
                    if abs(center_y_white - yellow_y_mean) < 0.5 and num_white_pts < yellow_point_count and num_white_pts < 180:
                        self.get_logger().info(f"Skipping white cluster {label} near yellow (y={yellow_y_mean:.2f}) with only {num_white_pts} pts")
                        continue

                # Passed all checks: add points to publish
                clustered_white_points.extend(cluster_points.tolist())

                # Curve fit
                coeffs = np.polyfit(points_xy_cluster[:, 0], points_xy_cluster[:, 1], deg=2)
                cluster_curves.append((label, coeffs, 'white'))

                center_x = np.mean(points_xy_cluster[:, 0])
                self.get_logger().info(f"White cluster {label}: center = ({center_x:.2f}, {center_y_white:.2f}), points = {num_white_pts}")

            # === Publish filtered white points only ===
        if len(clustered_white_points) > 0:
            white_msg = pc2.create_cloud_xyz32(msg.header, clustered_white_points)
            self.white_pub.publish(white_msg)
        self.get_logger().info(f"[Benchmark] White DBSCAN took {time.time() - start:.3f} sec")

        # === YELLOW DBSCAN and clustering ===
        start = time.time()
        points_np_yellow = np.array(final_yellow_points)
        clustered_yellow_points = []
        if len(points_np_yellow) >= MIN_CLUSTERING_POINTS:
            points_xy_yellow = points_np_yellow[:, :2]
            clustering_yellow = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy_yellow)
            labels_yellow = clustering_yellow.labels_

            # Combine all non-noise yellow cluster points
            clustered_yellow_points = points_np_yellow[labels_yellow != -1]

            if len(clustered_yellow_points) > 0:
                yellow_msg = pc2.create_cloud_xyz32(msg.header, clustered_yellow_points.tolist())
                self.yellow_pub.publish(yellow_msg)

            unique_labels_yellow = set(labels_yellow)
            n_clusters_y = len(unique_labels_yellow) - (1 if -1 in labels_yellow else 0)
            self.get_logger().info(f"The number of yellow clusters : {n_clusters_y}")
        self.get_logger().info(f"[Benchmark] Yellow DBSCAN took {time.time() - start:.3f} sec")

        # === Yellow Curve Fitting: Single global fit on all yellow points ===
        if len(clustered_yellow_points) >= 10:
            x_vals_y = clustered_yellow_points[:, 0]
            y_vals_y = clustered_yellow_points[:, 1]
            coeffs_yellow = np.polyfit(x_vals_y, y_vals_y, deg=2)
            cluster_curves.append(('yellow_global', coeffs_yellow, 'yellow'))

        # === Final Lane Visualization ===
        start = time.time()
        self.publish_lane_visualization(msg, None, cluster_curves, white_ground_points, final_yellow_points)
        self.get_logger().info(f"[Benchmark] Marker publishing took {time.time() - start:.3f} sec")

        

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()