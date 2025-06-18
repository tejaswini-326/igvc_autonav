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

class WhitePointImageVisualizer(Node):
    def __init__(self):
        super().__init__('white_point_image_visualizer')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.pointcloud_callback,
            10
        )
        self.cmd_pu b = self.create_publisher(Twist, '/cmd_vel', 10)

    def pointcloud_callback(self, msg):
        height = msg.height
        width = msg.width

        white_img = np.zeros((height, width, 3), dtype=np.uint8)
        white_ground_points = []

        index = 0
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
            x, y, z, rgb = point
            row = index // width
            col = index % width
            index += 1

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
            white_threshold = 120  # Increased threshold
            color_balance_threshold = 25  # Colors should be similar for true white
            
            # Check if pixel is white (high intensity + balanced RGB)
            avg_color = (r + g + b) / 3
            if (r > white_threshold and g > white_threshold and b > white_threshold and
                abs(r - avg_color) < color_balance_threshold and 
                abs(g - avg_color) < color_balance_threshold and 
                abs(b - avg_color) < color_balance_threshold):

                # Ground level filtering
                if -1.5 < z < -0.5:  # Adjusted range
                    white_img[row, col] = (255, 255, 255)
                    white_ground_points.append([x, y, z])  # Store x,y,z coordinates

        self.get_logger().info(f"White ground points: {len(white_ground_points)}")

        # Display only the white thresholded image
        cv2.imshow("White Lane Clusters", white_img)
        cv2.waitKey(1)

        if len(white_ground_points) < 10:  # Increased minimum threshold
            self.get_logger().warn("Not enough white points for clustering")
            return

        points_np = np.array(white_ground_points)
        
        # Use only x,y coordinates for clustering (ignore z)
        points_xy = points_np[:, :2]  # Extract x,y coordinates
        
        # Debug: Print point distribution
        x_coords = points_xy[:, 0]
        y_coords = points_xy[:, 1]
        z_coords = points_np[:, 2]
        self.get_logger().info(f"X range: {x_coords.min():.2f} to {x_coords.max():.2f}")
        self.get_logger().info(f"Y range: {y_coords.min():.2f} to {y_coords.max():.2f}")
        self.get_logger().info(f"Z range: {z_coords.min():.2f} to {z_coords.max():.2f}")

        # Better clustering parameters
        eps = 0.75  # Reduced eps for tighter clusters
        min_samples = 8  # Increased min_samples to reduce noise
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
        labels = clustering.labels_
        
        # Count clusters and noise points
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.get_logger().info(f"Clusters found: {n_clusters}, Noise points: {n_noise}")
        self.get_logger().info(f"Cluster labels: {unique_labels}")

        # Process valid clusters
        centers = []
        cluster_info = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            cluster_points = points_xy[labels == label]  # Use x,y coordinates
            cluster_size = len(cluster_points)
            
            # Filter out very small clusters
            if cluster_size < 15:
                self.get_logger().info(f"Cluster {label} too small ({cluster_size} points), skipping")
                continue
                
            center = np.mean(cluster_points, axis=0)
            std_dev = np.std(cluster_points, axis=0)
            
            # Filter out clusters that are too spread out (likely noise)
            if std_dev[0] > 0.5 or std_dev[1] > 0.5:
                self.get_logger().info(f"Cluster {label} too spread out (std: {std_dev}), skipping")
                continue
            
            centers.append((label, center))
            cluster_info.append({
                'label': label,
                'center': center,
                'size': cluster_size,
                'std': std_dev
            })
            
            self.get_logger().info(f"Valid cluster {label}: center=({center[0]:.2f}, {center[1]:.2f}), size={cluster_size}, std=({std_dev[0]:.2f}, {std_dev[1]:.2f})")

        # Lane detection logic (processing only, no visualization)
        if len(centers) >= 2:
            centers.sort(key=lambda c: c[1][0])  # sort by x coordinate
            left_lane = centers[0][1]   # leftmost cluster
            right_lane = centers[-1][1]  # rightmost cluster

            # Validate lane separation
            lane_separation = abs(right_lane[0] - left_lane[0])
            self.get_logger().info(f"Lane separation: {lane_separation:.2f}m")
            self.get_logger().info(f"Left lane: ({left_lane[0]:.2f}, {left_lane[1]:.2f})")
            self.get_logger().info(f"Right lane: ({right_lane[0]:.2f}, {right_lane[1]:.2f})")
            
            target = (left_lane + right_lane) / 2
            self.get_logger().info(f"Target point: ({target[0]:.2f}, {target[1]:.2f})")

            # Compute direction to target
            angle_to_target = math.atan2(target[1], target[0])  # direction from (0,0) to target

            # Move toward target
            cmd = Twist()
            cmd.linear.x = 0.3  # Forward speed

            # Small angle threshold to avoid jitter
            if abs(angle_to_target) > 0.05:
                cmd.angular.z = angle_to_target  # Steer towards target
                self.get_logger().info(f"Turning: angle to target = {math.degrees(angle_to_target):.2f}°")
            else:
                cmd.angular.z = 0.0
                self.get_logger().info("Target straight ahead")

            self.cmd_pub.publish(cmd)

        
        elif len(centers) == 1:
            self.get_logger().info("Only one lane cluster found")
            single_lane = centers[0][1]
            self.get_logger().info(f"Single lane at: ({single_lane[0]:.2f}, {single_lane[1]:.2f})")
        else:
            self.get_logger().warn("No valid clusters found for lane detection")


def main(args=None):
    rclpy.init(args=args)
    node = WhitePointImageVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()