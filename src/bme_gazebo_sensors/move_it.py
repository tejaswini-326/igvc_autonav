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
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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

            # White detection for lane boundaries
            white_threshold = 90
            color_balance_threshold = 40
            
            avg_color = (r + g + b) / 3
            if (r > white_threshold and g > white_threshold and b > white_threshold and
                abs(r - avg_color) < color_balance_threshold and 
                abs(g - avg_color) < color_balance_threshold and 
                abs(b - avg_color) < color_balance_threshold):

                # Ground level filtering
                if -2.0 < z < 0.2:
                    white_img[row, col] = (255, 255, 255)
                    white_ground_points.append([x, y, z])

        self.get_logger().info(f"White boundary points: {len(white_ground_points)}")

        cv2.imshow("White Lane Boundaries", white_img)
        cv2.waitKey(1)

        if len(white_ground_points) < 10:
            self.get_logger().warn("Not enough white points for lane boundary detection")
            # Continue straight when no boundaries detected
            cmd = Twist()
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        points_np = np.array(white_ground_points)
        points_xy = points_np[:, :2]  # Extract x,y coordinates
        
        # Clustering to find lane boundaries
        eps = 0.6
        min_samples = 8
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        
        self.get_logger().info(f"Lane boundaries found: {n_clusters}")

        # Get valid lane boundaries
        lane_boundaries = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            cluster_points = points_xy[labels == label]
            cluster_size = len(cluster_points)
            
            if cluster_size < 15:  # Filter small clusters
                continue
                
            center = np.mean(cluster_points, axis=0)
            std_dev = np.std(cluster_points, axis=0)
            
            # Filter scattered clusters
            if std_dev[0] > 0.5 or std_dev[1] > 0.5:
                continue
            
            lane_boundaries.append({
                'center': center,
                'size': cluster_size,
                'x_pos': center[0]  # x position for sorting
            })
            
            self.get_logger().info(f"Lane boundary at x={center[0]:.2f}, y={center[1]:.2f}, size={cluster_size}")

        # Lane maintenance logic - stay between two boundaries
        cmd = Twist()
        cmd.linear.x = 0.75  # Always move forward
        
        if len(lane_boundaries) < 2:
            # Need at least 2 boundaries to define a lane
            cmd.angular.z = 0.0
            self.get_logger().warn("Need at least 2 lane boundaries - going straight")
            
        else:
            # Sort boundaries by x-coordinate (left to right)
            lane_boundaries.sort(key=lambda b: b['x_pos'])
            
            # Find the most appropriate pair of boundaries using distance-based approach
            # This doesn't assume robot position, just finds the "best" lane to follow
            
            best_lane_pair = None
            min_score = float('inf')
            
            # Evaluate all possible boundary pairs
            for i in range(len(lane_boundaries) - 1):
                left_boundary = lane_boundaries[i]
                right_boundary = lane_boundaries[i + 1]
                
                left_x = left_boundary['x_pos']
                right_x = right_boundary['x_pos']
                
                # Calculate lane properties
                lane_center = (left_x + right_x) / 2
                lane_width = right_x - left_x
                
                # Scoring function (lower is better):
                # 1. Prefer lanes with centers closer to straight ahead (x=0)
                # 2. Prefer reasonable lane widths (not too narrow/wide)
                # 3. Prefer larger boundary clusters (more reliable detection)
                
                center_distance_penalty = abs(lane_center) * 2.0  # Prefer center lanes
                
                # Width penalty (prefer reasonable lane widths, e.g., 1.5-4.0 meters)
                ideal_width = 2.5
                width_penalty = abs(lane_width - ideal_width) * 0.5
                
                # Size bonus (larger clusters are more reliable)
                size_bonus = -(left_boundary['size'] + right_boundary['size']) * 0.01
                
                # Total score
                score = center_distance_penalty + width_penalty + size_bonus
                
                if score < min_score:
                    min_score = score
                    best_lane_pair = (left_boundary, right_boundary)
                
                self.get_logger().info(f"Lane pair {i}: center={lane_center:.2f}, width={lane_width:.2f}, score={score:.2f}")
            
            if best_lane_pair:
                left_boundary, right_boundary = best_lane_pair
            
            if left_boundary and right_boundary:
                # Calculate lane center (midpoint between boundaries)
                left_x = left_boundary['x_pos']
                right_x = right_boundary['x_pos']
                lane_center_x = (left_x + right_x) / 2
                
                # Calculate how far robot is from lane center
                lateral_error = lane_center_x  # Positive means lane center is to the right
                
                # Proportional control to stay centered in lane
                kp = 1.2  # Tunable gain
                cmd.angular.z = -lateral_error * kp  # Negative because ROS angular.z convention
                
                # Limit turning rate to prevent oscillation
                max_turn_rate = 1.0
                cmd.angular.z = max(-max_turn_rate, min(max_turn_rate, cmd.angular.z))
                
                lane_width = right_x - left_x
                self.get_logger().info(f"Lane maintenance: left_boundary={left_x:.2f}, right_boundary={right_x:.2f}, lane_center={lane_center_x:.2f}, width={lane_width:.2f}, error={lateral_error:.2f}, turn_rate={cmd.angular.z:.2f}")
                
            else:
                # Fallback - go straight
                cmd.angular.z = 0.0
                self.get_logger().warn("Could not determine lane boundaries - going straight")

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = WhitePointImageVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()