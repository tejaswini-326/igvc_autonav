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
        
        # NEW: Enhanced parking parameters
        self.target_slot_center_y = None
        self.alignment_count = 0
        self.max_alignment_attempts = 15

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

    # NEW: Helper function to validate parking slot
    def validate_parking_slot(self, line_centers):
        """Check if detected lines form a valid parking slot"""
        if not line_centers or len(line_centers) < 2:
            return False, None, None
            
        # Sort by y coordinate
        line_centers.sort(key=lambda c: c[1])
        
        # Check all possible pairs for valid parking slots
        for i in range(len(line_centers)):
            for j in range(i+1, len(line_centers)):
                left_line = line_centers[i]
                right_line = line_centers[j]
                
                slot_width = abs(right_line[1] - left_line[1])
                slot_center_y = (left_line[1] + right_line[1]) / 2
                
                # Valid parking slot criteria
                if 4.0 < slot_width < 6.0:  # Reasonable width
                    return True, slot_center_y, slot_width
                    
        return False, None, None

    # NEW: Calculate precise control commands
    def calculate_alignment_command(self, line_centers):
        """Calculate precise steering to center between parking lines"""
        cmd = Twist()
        
        if not line_centers or len(line_centers) < 2:
            # Default behavior when lines not visible
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
            return cmd
        
        # Find the best pair of lines for parking slot
        is_valid, slot_center_y, slot_width = self.validate_parking_slot(line_centers)
        
        if not is_valid:
            # Keep searching
            cmd.linear.x = 0.3
            cmd.angular.z = 0.1
            return cmd
        
        # Store target for consistency
        if self.target_slot_center_y is None:
            self.target_slot_center_y = slot_center_y
        else:
            # Smooth update to avoid jitter
            self.target_slot_center_y = 0.8 * self.target_slot_center_y + 0.2 * slot_center_y
        
        # Calculate steering to center robot in slot
        centering_error = self.target_slot_center_y  # Robot should be at y=0
        
        # P-controller for centering
        kp_angular = 0.8
        kp_linear = 0.6
        
        # Linear speed based on how well centered we are
        if abs(centering_error) < 0.3:  # Well centered
            cmd.linear.x = 0.4
        elif abs(centering_error) < 0.8:  # Moderately centered  
            cmd.linear.x = 0.3
        else:  # Poorly centered
            cmd.linear.x = 0.2
            
        # Angular velocity for centering
        cmd.angular.z = -kp_angular * centering_error
        
        # Limit commands
        cmd.linear.x = max(0.1, min(0.5, cmd.linear.x))
        cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z))
        
        self.get_logger().info(f"[ALIGNMENT] Center error: {centering_error:.2f}, Width: {slot_width:.2f}, Cmd: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}")
        
        return cmd

    # NEW: Enhanced turning logic
    def calculate_turning_command(self, line_centers):
        """Enhanced turning to properly align with parking slot"""
        cmd = Twist()
        
        if not line_centers or len(line_centers) < 2:
            # Keep turning if lines not visible
            cmd.linear.x = 0.0
            cmd.angular.z = 0.15
            return cmd
            
        # Check alignment quality
        is_valid, slot_center_y, slot_width = self.validate_parking_slot(line_centers)
        
        if is_valid:
            # Check if we're well aligned
            if abs(slot_center_y) < 0.5 and 2.0 < slot_width < 3.0:
                self.get_logger().info(f"Good alignment achieved! Center: {slot_center_y:.2f}, Width: {slot_width:.2f}")
                return None  # Signal to move to next state
        
        # Continue turning with slight forward motion
        cmd.linear.x = 0.1
        cmd.angular.z = 0.12
        
        self.alignment_count += 1
        if self.alignment_count > self.max_alignment_attempts:
            self.get_logger().warn("Max alignment attempts reached, proceeding anyway")
            return None  # Force proceed
            
        return cmd

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
                # ENHANCED: Use alignment control even while driving in slot
                cmd = self.calculate_alignment_command(line_centers)
                if cmd.angular.z != 0.0:  # Still need centering
                    self.get_logger().info("Fine-tuning alignment while driving in slot")
                else:
                    cmd.linear.x = 0.4  # Steady forward motion
                    cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return

        # ENHANCED TURNING HANDLING
        if self.state == "TURNING":
            turn_cmd = self.calculate_turning_command(line_centers)
            
            if turn_cmd is None:  # Alignment achieved
                self.get_logger().info("Alignment complete - starting to drive into slot")
                self.state = "DRIVING_IN_SLOT"
                self.alignment_count = 0  # Reset counter
                return
            else:
                self.cmd_pub.publish(turn_cmd)
                return

        # MAIN STATE MACHINE (keeping your original logic)
        if self.state == "SEARCHING":
            if line_centers and len(line_centers) >= 1:
                line_centers.sort(key=lambda c: c[1])
                self.first_line_y = line_centers[0][1]
                self.state = "FOUND_ONE_LINE"
                self.get_logger().info(f"First yellow line found at y={self.first_line_y:.2f}")
            cmd = Twist()
            cmd.linear.x = 0.7
            cmd.angular.z = 0.1
            self.cmd_pub.publish(cmd)

        elif self.state == "FOUND_ONE_LINE":
            if line_centers and len(line_centers) >= 2:
                # ENHANCED: Check for valid parking slot instead of just any second line
                is_valid, slot_center_y, slot_width = self.validate_parking_slot(line_centers)
                
                if is_valid:
                    self.get_logger().info(f"Valid parking slot found! Center: {slot_center_y:.2f}, Width: {slot_width:.2f}")
                    self.state = "TURNING"
                    self.target_slot_center_y = slot_center_y
                    self.alignment_count = 0
                    return
                else:
                    # Keep looking for a valid slot
                    for c in line_centers:
                        y = c[1]
                        if abs(y - self.first_line_y) > 0.5:
                            self.get_logger().info(f"Second line at y={y:.2f} found but slot not valid yet")
                            break
                            
            # Still go forward until valid slot found  
            cmd = Twist()
            cmd.linear.x = 0.5
            cmd.angular.z = 0.3
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