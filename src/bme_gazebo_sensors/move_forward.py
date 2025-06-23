import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import math
from geometry_msgs.msg import Twist
# x forward, y left, z upward

class WhitePointImageVisualizer(Node):
    def __init__(self):
        super().__init__('white_point_image_visualizer')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.pointcloud_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_cmd = Twist()
        self.last_cmd.linear.x = 1.0
        self.last_cmd.angular.z = 0.0
        self.stopped = False

        # Set the variable below to 'left' or 'right' depending on which lane you want the robot to follow
        self.which_lane = 'right'

    def publish(self, cmd, target=None):
        if self.stopped:
            # Override any move command with a full stop
            self.get_logger().warn("Robot is stopped — blocking velocity command.")
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.cmd_pub.publish(stop_cmd)
        else:
            self.cmd_pub.publish(cmd)

        print('')

    def debug_time_yo_yo_yo(self, x, y, msg, img, centers):
        self.get_logger().info(f"DEBUG")
        height = msg.height
        width = msg.width

        white_img = img
        index = 0
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
            px, py, pz, rgb = point
            if abs(px - x) < 0.02 and abs(py - y) < 0.02:
                row = index // width
                col = index % width
                cv2.circle(white_img, (col, row), 5, (0, 0, 255), -1)
            for label, center in centers:
                cx = center[0]
                cy = center[1]
                if abs(px - cx) < 0.02 and abs(py - cy) < 0.02:
                    row = index // width
                    col = index % width
                    cv2.circle(white_img, (col, row), 5, (255, 0, 0), -1)
                    text = f"({cx:.2f}, {cy:.2f})"
                    cv2.putText(
                        white_img, text, (col + 10, row - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                    )
            index += 1

        cv2.imshow("Target", white_img)
        cv2.waitKey(1) 
        return
         
    def calculate_normal_velocity(self, target, msg, white_img, centers):
        cmd = Twist()
        self.debug_time_yo_yo_yo(target[0], target[1], msg, white_img, centers)

        # Compute direction to target
        angle_to_target = math.atan2(target[1], target[0])  # direction from (0,0) to target in radians

        # Move toward target
        cmd.linear.x = 5.0  # Forward speed
    
        # Small angle threshold to avoid jitter
        if abs(angle_to_target) > 0.05:
            cmd.angular.z = angle_to_target  # Steer towards target
            if abs(angle_to_target) > 0.4:
                cmd.linear.x = 0.2
            self.get_logger().info(f"Turning: angle to target = {math.degrees(angle_to_target):.2f}°")
        else:
            cmd.angular.z = 0.0
            self.get_logger().info("Target straight ahead")
        return cmd
    
    def detect_horizontal_lines_2d(self, msg):
        """Detect stop line by checking dense spread in Y across 2–3m ahead."""
        white_y_vals = []

        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
            x, y, z, rgb = point
            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            except:
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            white_threshold = 100  # Increased threshold
            color_balance_threshold = 25  # Colors should be similar for true white
            
            # Check if pixel is white (high intensity + balanced RGB)
            avg_color = (r + g + b) / 3
            if (r > white_threshold and g > white_threshold and b > white_threshold and
                abs(r - avg_color) < color_balance_threshold and 
                abs(g - avg_color) < color_balance_threshold and 
                abs(b - avg_color) < color_balance_threshold):

                if 3.0 < x < 3.5 and -1.5 < z < -0.5:
                    white_y_vals.append(y)

        if len(white_y_vals) < 300:
            return

        # Create histogram over Y axis
        hist, bin_edges = np.histogram(white_y_vals, bins=20, range=(-2.0, 2.0))

        # Find contiguous bin groups with high density
        dense_threshold = 5  # min points per bin
        dense_bins = [bin_edges[i] for i in range(len(hist)) if hist[i] >= dense_threshold]

        if len(dense_bins) < 2:
            return

        # Dense region span = difference between leftmost and rightmost high bins
        y_dense_min = min(dense_bins)
        y_dense_max = max(dense_bins)
        dense_y_range = y_dense_max - y_dense_min

        if dense_y_range > 0.7:
            self.get_logger().warn(
                f"STOP LINE DETECTED: dense y-range = {dense_y_range:.2f}m with {len(white_y_vals)} points"
            )
            self.stopped = True



    
    def pointcloud_callback(self, msg):
        self.detect_horizontal_lines_2d(msg)
        # if self.stopped == True:
        #     self.destroy_node()

        height = msg.height
        width = msg.width

        white_img = np.zeros((height, width, 3), dtype=np.uint8)
        white_ground_points = []

        index = 0
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
            x, y, z, rgb = point
            row = index // width
            col = index % width

            # Skip invalid points
            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            except:
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            # Improved white detection
            white_threshold = 100  # Increased threshold
            color_balance_threshold = 25  # Colors should be similar for true white
            
            # Check if pixel is white (high intensity + balanced RGB)
            avg_color = (r + g + b) / 3
            if (r > white_threshold and g > white_threshold and b > white_threshold and
                abs(r - avg_color) < color_balance_threshold and 
                abs(g - avg_color) < color_balance_threshold and 
                abs(b - avg_color) < color_balance_threshold):

                # Ground level filtering
                if -1.4 < z < -1.3 and 0.0 < x < 3.0:  # Adjusted range
                    white_img[row, col] = (255, 255, 255)
                    white_ground_points.append([x, y, z])  # Store x,y,z coordinates
            index += 1

        # Only do 3D clustering for lane following if we have enough points
        if len(white_ground_points) < 10:
            self.get_logger().warn("Not enough white points for lane detection")
            self.publish(self.last_cmd)
            return

        points_np = np.array(white_ground_points)
        
        # Use only x,y coordinates for clustering (ignore z)
        points_xy = points_np[:, :2]  # Extract x,y coordinates
        
        # Debug: Print point distribution
        x_coords = points_xy[:, 0]
        y_coords = points_xy[:, 1]
        z_coords = points_np[:, 2]
        # self.get_logger().info(f"X range: {x_coords.min():.2f} to {x_coords.max():.2f}")
        # self.get_logger().info(f"Y range: {y_coords.min():.2f} to {y_coords.max():.2f}")
        # self.get_logger().info(f"Z range: {z_coords.min():.2f} to {z_coords.max():.2f}")

        # Better clustering parameters
        eps = 0.75  # Reduced eps for tighter clusters
        min_samples = 20  # Increased min_samples to reduce noise
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
        labels = clustering.labels_
        
        # Count clusters and noise points
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # self.get_logger().info(f"Clusters found: {n_clusters}, Noise points: {n_noise}")

        # Process lane clusters
        centers = []
        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = points_xy[labels == label]
            cluster_size = len(cluster_points)
            if cluster_size < 10:
                continue

            center = np.mean(cluster_points, axis=0)
            centers.append((label, center))

        # Lane following logic
        cmd = Twist()
        if len(centers) == 2:
            centers.sort(key=lambda c: c[1][1])  # sort by y coordinate
            self.get_logger().info(f"{centers}")

            left_lane = centers[1][1]   # leftmost cluster
            right_lane = centers[0][1]  # rightmost cluster
            
            target = (left_lane + right_lane) / 2
            # self.get_logger().info(f"Target point: ({target[0]:.2f}, {target[1]:.2f})")
            cmd = self.calculate_normal_velocity(target, msg, white_img, centers)
            
            self.publish(cmd, target)
            self.last_cmd = cmd
            self.last_cmd.linear.x = 0.0
            return

        if len(centers) >= 3:
            centers.sort(key=lambda c: c[1][1])  # sort by y coordinate
            # self.get_logger().info(f"{centers}")

            right_lane = centers[0][1]
            middle_lane = centers[1][1]
            left_lane = centers[2][1]

            target = ((middle_lane + right_lane) / 2) if self.which_lane == 'right' else ((middle_lane + left_lane) / 2)
            # self.get_logger().info(f"Target point: ({target[0]:.2f}, {target[1]:.2f})")
            cmd = self.calculate_normal_velocity(target, msg, white_img, centers)

            self.publish(cmd, target)
            self.last_cmd = cmd
            self.last_cmd.linear.x = 0.0
            return
        
        elif len(centers) < 2:
            if len(centers) == 1:
                # self.get_logger().info("Only one lane cluster found")
                single_lane = centers[0][1]
                self.debug_time_yo_yo_yo(0, 0, msg, white_img, centers)
                # self.get_logger().info(f"Single lane at: ({single_lane[0]:.2f}, {single_lane[1]:.2f})")
            # else:
            #     self.get_logger().warn("No valid clusters found for lane detection")
            self.publish(self.last_cmd)
            return


def main(args=None):
    rclpy.init(args=args)
    node = WhitePointImageVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
