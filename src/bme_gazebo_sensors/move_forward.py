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

    def publish(self, cmd):
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
            index += 1

        cv2.imshow("Target", white_img)
        cv2.waitKey(1)  

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

<<<<<<< HEAD:src/bme_gazebo_sensors/move_it.py
            # White detection for lane boundaries
            white_threshold = 90
            color_balance_threshold = 40
=======
            # Improved white detection
            white_threshold = 100  # Increased threshold
            color_balance_threshold = 25  # Colors should be similar for true white
>>>>>>> 546608dd1c7f8468e25822a19edae7af6717a999:src/bme_gazebo_sensors/move_forward.py
            
            avg_color = (r + g + b) / 3
            if (r > white_threshold and g > white_threshold and b > white_threshold and
                abs(r - avg_color) < color_balance_threshold and 
                abs(g - avg_color) < color_balance_threshold and 
                abs(b - avg_color) < color_balance_threshold):

                # Ground level filtering
                if -2.0 < z < 0.2:
                    white_img[row, col] = (255, 255, 255)
<<<<<<< HEAD:src/bme_gazebo_sensors/move_it.py
                    white_ground_points.append([x, y, z])
=======
                    white_ground_points.append([x, y, z])  # Store x,y,z coordinates
            index += 1
>>>>>>> 546608dd1c7f8468e25822a19edae7af6717a999:src/bme_gazebo_sensors/move_forward.py

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
            
<<<<<<< HEAD:src/bme_gazebo_sensors/move_it.py
            if cluster_size < 15:  # Filter small clusters
=======
            # Filter out very small clusters
            if cluster_size < 10:
                self.get_logger().info(f"Cluster {label} too small ({cluster_size} points), skipping")
>>>>>>> 546608dd1c7f8468e25822a19edae7af6717a999:src/bme_gazebo_sensors/move_forward.py
                continue
                
            center = np.mean(cluster_points, axis=0)
            std_dev = np.std(cluster_points, axis=0)
            
<<<<<<< HEAD:src/bme_gazebo_sensors/move_it.py
            # Filter scattered clusters
            if std_dev[0] > 0.5 or std_dev[1] > 0.5:
                continue
=======
>>>>>>> 546608dd1c7f8468e25822a19edae7af6717a999:src/bme_gazebo_sensors/move_forward.py
            
            lane_boundaries.append({
                'center': center,
                'size': cluster_size,
                'x_pos': center[0]  # x position for sorting
            })
            
<<<<<<< HEAD:src/bme_gazebo_sensors/move_it.py
            self.get_logger().info(f"Lane boundary at x={center[0]:.2f}, y={center[1]:.2f}, size={cluster_size}")
=======
            self.get_logger().info(f"Valid cluster {label}: center=({center[0]:.2f}, {center[1]:.2f}), size={cluster_size}, std=({std_dev[0]:.2f}, {std_dev[1]:.2f})")

        cmd = Twist()
        # Lane detection logic (processing only, no visualization)
        if len(centers) == 2:
            centers.sort(key=lambda c: c[1][0])  # sort by x coordinate
            left_lane = centers[0][1]   # leftmost cluster
            right_lane = centers[-1][1]  # rightmost cluster

            lane_separation = abs(right_lane[0] - left_lane[0])
            self.get_logger().info(f"Lane separation: {lane_separation:.2f}m")
            self.get_logger().info(f"Left lane: ({left_lane[0]:.2f}, {left_lane[1]:.2f})")
            self.get_logger().info(f"Right lane: ({right_lane[0]:.2f}, {right_lane[1]:.2f})")
            
            target = (left_lane + right_lane) / 2
            self.debug_time_yo_yo_yo(target[0], target[1], msg, white_img, centers)
            self.get_logger().info(f"Target point: ({target[0]:.2f}, {target[1]:.2f})")

            # Compute direction to target
            angle_to_target = math.atan2(target[1], target[0])  # direction from (0,0) to target

            # Move toward target
            cmd.linear.x = 0.3  # Forward speed

            # Small angle threshold to avoid jitter
            if abs(angle_to_target) > 0.05:
                cmd.angular.z = angle_to_target  # Steer towards target
                self.get_logger().info(f"Turning: angle to target = {math.degrees(angle_to_target):.2f}°")
            else:
                cmd.angular.z = 0.0
                self.get_logger().info("Target straight ahead")

            self.publish(cmd)

        if len(centers) >= 3:
            centers.sort(key=lambda c: c[1][0])  # sort by x coordinate

            right_lane = centers[0][1]
            middle_lane = centers[1][1]
            left_lane = centers[2][1]

            target = (middle_lane + right_lane) / 2

            lane_separation = abs(right_lane[0] - left_lane[0])
            self.get_logger().info(f"Lane separation: {lane_separation:.2f}m")
            self.get_logger().info(f"Left lane: ({left_lane[0]:.2f}, {left_lane[1]:.2f})")
            self.get_logger().info(f"Right lane: ({right_lane[0]:.2f}, {right_lane[1]:.2f})")

            self.debug_time_yo_yo_yo(target[0], target[1], msg, white_img, centers)
            self.get_logger().info(f"Target point: ({target[0]:.2f}, {target[1]:.2f})")

            # Compute direction to target
            angle_to_target = math.atan2(target[1], target[0])  # direction from (0,0) to target

            # Move toward target
            cmd.linear.x = 0.3  # Forward speed

            # Small angle threshold to avoid jitter
            if abs(angle_to_target) > 0.05:
                cmd.angular.z = angle_to_target  # Steer towards target
                self.get_logger().info(f"Turning: angle to target = {math.degrees(angle_to_target):.2f}°")
            else:
                cmd.angular.z = 0.0
                self.get_logger().info("Target straight ahead")

            self.publish(cmd)
>>>>>>> 546608dd1c7f8468e25822a19edae7af6717a999:src/bme_gazebo_sensors/move_forward.py

        # Lane maintenance logic - stay between two boundaries
        cmd = Twist()
        cmd.linear.x = 0.75  # Always move forward
        
<<<<<<< HEAD:src/bme_gazebo_sensors/move_it.py
        if len(lane_boundaries) < 2:
            # Need at least 2 boundaries to define a lane
            cmd.angular.z = 0.0
            self.get_logger().warn("Need at least 2 lane boundaries - going straight")
            
=======
        elif len(centers) == 1:
            self.get_logger().info("Only one lane cluster found")
            single_lane = centers[0][1]
            self.debug_time_yo_yo_yo(0, 0, msg, white_img, centers)
            self.get_logger().info(f"Single lane at: ({single_lane[0]:.2f}, {single_lane[1]:.2f})")

            self.publish(cmd)
>>>>>>> 546608dd1c7f8468e25822a19edae7af6717a999:src/bme_gazebo_sensors/move_forward.py
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