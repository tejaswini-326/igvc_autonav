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

class YellowParkingDetector(Node):
    def __init__(self):
        super().__init__('yellow_parking_detector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.pointcloud_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parking logic state
        self.state = "SEARCHING"
        self.first_line_y = None
        self.turn_start_time = None

        self.get_logger().info("Yellow parking detector initialized")


    def detect_yellow_color(self, r, g, b):
        """Detect pale/faded yellow parking lines - adapted from white detection logic"""
        # These are pale yellow/cream colored lines, not bright yellow
        # Use similar logic to white detection but with yellow tint
        
        pale_threshold = 80  # Lower threshold for faded lines
        color_balance_threshold = 30  # Allow more variation for faded colors
        
        # Calculate average color
        avg_color = (r + g + b) / 3
        
        # Check if pixel is pale/faded yellow:
        # 1. Should be reasonably bright (but not as bright as white)
        # 2. Red and Green should be higher than Blue (yellow tint)
        # 3. Colors should be somewhat balanced (not too saturated)
        
        is_bright_enough = (r > pale_threshold and g > pale_threshold)
        has_yellow_tint = (r > b and g > b)  # Yellow = more red+green, less blue
        is_balanced = (abs(r - avg_color) < color_balance_threshold and 
                      abs(g - avg_color) < color_balance_threshold)
        
        # Additional check: it should look more yellow/cream than pure white or gray
        yellow_factor = (r + g) / (2 * max(b, 1))  # Ratio of yellow components to blue
        
        return (is_bright_enough and 
                has_yellow_tint and 
                is_balanced and
                yellow_factor > 1.1 and  # Should have some yellow tint
                avg_color > 70)  # Minimum brightness

    def detect_horizontal_stop_line(self, msg):
        """Detect horizontal yellow stop line using histogram density in y-direction"""
        yellow_y_vals = []

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

            if self.detect_yellow_color(r, g, b):
                if 3.0 < x < 4.0 and -1.4 < z < -1.2:
                    yellow_y_vals.append(y)

        if len(yellow_y_vals) < 100:
            return False

        # Create histogram over Y axis
        hist, bin_edges = np.histogram(yellow_y_vals, bins=20, range=(-2.0, 2.0))
        dense_threshold = 5  # Minimum number of points per bin
        dense_bins = [bin_edges[i] for i in range(len(hist)) if hist[i] >= dense_threshold]

        if len(dense_bins) < 2:
            return False

        y_dense_min = min(dense_bins)
        y_dense_max = max(dense_bins)
        dense_y_range = y_dense_max - y_dense_min

        self.get_logger().info(f"[STOP CHECK] Yellow dense y-span: {dense_y_range:.2f}, total points: {len(yellow_y_vals)}")

        if dense_y_range > 2.0 and len(yellow_y_vals)  > 700:
            self.get_logger().warn("DENSE HORIZONTAL YELLOW STOP LINE DETECTED")
            return True

        return False

    def find_parking_slot(self, msg):
        """Find parking slot using clustering like in move_forward.py"""
        height = msg.height
        width = msg.width
        
        if height == 0 or width == 0:
            return None, None

        yellow_img = np.zeros((height, width, 3), dtype=np.uint8)
        yellow_ground_points = []
        debug_count = 0

        index = 0
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
            x, y, z, rgb = point
            row = index // width
            col = index % width

            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                index += 1
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            except:
                index += 1
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            # Debug: Print some ground pixel colors to understand what we're dealing with
            if 1.0 < x < 3.0 and -1.4 < z < -1.3 and debug_count % 1000 == 0:
                #self.get_logger().info(f"Ground pixel RGB: R={r}, G={g}, B={b} at ({x:.1f}, {y:.1f})")
                debug_count += 1

            # Check if pixel is yellow/pale yellow
            if self.detect_yellow_color(r, g, b):
                # Look for parking lines on the ground ahead of robot
                if -1.4 < z < -1.3 and 1.0 < x < 3.0:  # Ground level, ahead of robot
                    if 0 <= row < height and 0 <= col < width:
                        yellow_img[row, col] = (0, 255, 255)  # Yellow in BGR
                    yellow_ground_points.append([x, y, z])
                    
                    # Debug: Print detected yellow pixels
                    #if len(yellow_ground_points) % 50 == 0:
                        #self.get_logger().info(f"DETECTED pale yellow: R={r}, G={g}, B={b} at Y={y:.2f}")
            
            index += 1

        # Show debug visualization
        try:
            if yellow_img.size > 0:
                cv2.imshow("Pale Yellow Parking Lines", yellow_img)
                cv2.waitKey(1)
        except:
            pass

        #self.get_logger().info(f"Total pale yellow pixels detected: {len(yellow_ground_points)}")

        # Need enough points for clustering
        if len(yellow_ground_points) < 20:
            return None, yellow_img

        # If we're getting too many detections, the threshold might be too loose
        # if len(yellow_ground_points) > 1000:
        #     self.get_logger().warn(f"Too many detections ({len(yellow_ground_points)}) - color threshold might be too loose")

        # Cluster the yellow points (similar to move_forward.py clustering)
        points_np = np.array(yellow_ground_points)
        points_xy = points_np[:, :2]  # Use only x,y for clustering

        eps = 0.3  # Smaller eps for parking lines
        min_samples = 15
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
        labels = clustering.labels_
        unique_labels = set(labels)

        # Find parking slot boundaries
        line_centers = []
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            cluster_points = points_xy[labels == label]
            if len(cluster_points) < 10:
                continue
                
            center = np.mean(cluster_points, axis=0)
            line_centers.append(center)

        return line_centers, yellow_img
    
    def pointcloud_callback(self, msg):
        if self.state == "STOPPED":
            stop = Twist()
            self.cmd_pub.publish(stop)
            return

        line_centers, _ = self.find_parking_slot(msg)

        # STOP CONDITION (if already aligned and moving)
        if self.state == "DRIVING_IN_SLOT":
            if self.detect_horizontal_stop_line(msg):
                self.get_logger().warn("STOP line detected. Stopping bot.")
                self.state = "STOPPED"
                stop = Twist()
                self.cmd_pub.publish(stop)
                return
            else:
                cmd = Twist()
                cmd.linear.x = 0.5
                cmd.angular.z = 0.0
                self.get_logger().warn("No side lanes found — moving straight cautiously.")
                self.cmd_pub.publish(cmd)
                return


        # FORCED TURN HANDLING
        if self.state == "TURNING":
            # Keep turning until sidelanes are balanced around bot
            line_centers, _ = self.find_parking_slot(msg)
            
            if line_centers and len(line_centers) >= 2:
                # Sort by y (left to right)
                line_centers.sort(key=lambda c: c[1])
                left_y = line_centers[-1][1]
                right_y = line_centers[0][1]
                center_y = (left_y + right_y) / 2
                y_gap = abs(left_y - right_y)

                self.get_logger().info(f"[TURNING] left_y={left_y:.2f}, right_y={right_y:.2f}, center_y={center_y:.2f}, gap={y_gap:.2f}")

                # Check if they are symmetric around y=0 and far enough apart
                if abs(center_y) < 1.0 and y_gap > 2.0 and y_gap < 4.0:
                    self.get_logger().info("Aligned between two yellow lines — starting to drive in.")
                    self.state = "DRIVING_IN_SLOT"
                    return

            # Keep turning left
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
            self.cmd_pub.publish(cmd)
            return

        # MAIN STATE MACHINE
        if self.state == "SEARCHING":
            if line_centers and len(line_centers) >= 1:
                line_centers.sort(key=lambda c: c[1])
                self.first_line_y = line_centers[0][1]
                # if (self.first_line_y <1.5 and self.first_line_y  >0.5):
                self.state = "FOUND_ONE_LINE"
                self.get_logger().info(f"First yellow line found at y={self.first_line_y:.2f}")
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = 0.5
            self.cmd_pub.publish(cmd)

        elif self.state == "FOUND_ONE_LINE":
            if line_centers and len(line_centers) >= 2:
                for c in line_centers:
                    y = c[1]
                    if abs(y - self.first_line_y) < 0.6:
                        self.get_logger().info(f"Second yellow line at y={y:.2f} matches first.")
                        self.state = "TURNING"
                        self.turn_start_time = None
                        return
            # Still go forward until second line found
            cmd = Twist()
            cmd.linear.x = 0.7
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = YellowParkingDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()