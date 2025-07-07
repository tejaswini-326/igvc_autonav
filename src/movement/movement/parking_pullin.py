import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import math
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# The max length of the stop line
MAX_STOP_LINE_LENGTH = 100 
# Constants for velocity calculation
LINEAR_SPEED = 1.0
ANGLE_FACTOR = 1.1

class YellowParkingDetector(Node):
    def __init__(self):
        super().__init__('yellow_parking_detector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/igvc/yellow_points',
            self.pointcloud_callback,
            10
        )
        
        # subscribing to camera/image for stop line detection and not using pointcloud data at all
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        #Storing latest image for stop line detection
        self.latest_image = None

        # Parking logic state
        self.state = "SEARCHING"
        self.first_line_y = None
        self.turn_start_time = None
        
        # Parking parameters
        self.target_slot_center_y = None
        # fallback parameters to not idle away from real world situations
        self.alignment_count = 0
        self.max_alignment_attempts = 15

        self.get_logger().info("Yellow parking detector initialized")

    def image_callback(self, msg):
        # Store the latest image for stop line detection
        self.latest_image = msg


    # FUNCTION TO DETECT YELLOW COLOUR FOR DETECTING PARKING LOT
    def detect_yellow_color(self, r, g, b):
        pale_threshold = 80  
        color_balance_threshold = 30  
        avg_color = (r + g + b) / 3
        
        # Conditions for the yellow in parking in this world file
        is_bright_enough = (r > pale_threshold and g > pale_threshold)
        has_yellow_tint = (r > b and g > b)  
        is_balanced = (abs(r - avg_color) < color_balance_threshold and abs(g - avg_color) < color_balance_threshold)
        yellow_factor = (r + g) / (2 * max(b, 1)) 
        
        return (is_bright_enough and has_yellow_tint and is_balanced and yellow_factor > 1.1 and avg_color > 70)  

    # FUNCTION TO DETECT HORIZONTAL STOP LINE
    def detect_horizontal_stop_line(self):
        if self.latest_image is None:
            return False
            
        try:
            # Convert ROS Image to OpenCV BGR format
            frame1 = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            original = frame1.copy()
            height = frame1.shape[0]

            # Slice bottom 30% of the image (same as stop_sign.py)
            frame = frame1[int(0.7 * height):, :]

            # Convert cropped region to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Yellow mask (for stop lines)
            lower_yellow = np.array([15, 30, 60])
            upper_yellow = np.array([30, 120, 200])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # White mask (alternative stop line color, excluding yellow)
            lower_white = np.array([0, 0, 120])
            upper_white = np.array([25, 60, 255])
            mask_white_raw = cv2.inRange(hsv, lower_white, upper_white)

            # Clean white by subtracting yellow
            mask_white = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_yellow))

            # Stop line detection from yellow mask
            stop_edges_yellow = cv2.Canny(mask_yellow, 50, 150)
            stop_lines_yellow = cv2.HoughLinesP(stop_edges_yellow, 1, np.pi / 180, threshold=30,
                                         minLineLength=20, maxLineGap=10)

            # Stop line detection from white mask
            stop_edges_white = cv2.Canny(mask_white, 50, 150)
            stop_lines_white = cv2.HoughLinesP(stop_edges_white, 1, np.pi / 180, threshold=30,
                                         minLineLength=20, maxLineGap=10)

            # Check yellow stop lines
            if stop_lines_yellow is not None:
                for line in stop_lines_yellow:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if (angle < 10 or angle > 170) and length < MAX_STOP_LINE_LENGTH:
                        self.get_logger().warn("HORIZONTAL YELLOW STOP LINE DETECTED")
                        # Drawing line for debugging
                        cv2.line(original, (x1, y1 + int(0.7 * height)), (x2, y2 + int(0.7 * height)), (0, 255, 0), 3)
                        cv2.imshow("Stop Line Detection", original)
                        cv2.waitKey(1)
                        return True

            # Check white stop lines
            if stop_lines_white is not None:
                for line in stop_lines_white:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if (angle < 10 or angle > 170) and length < MAX_STOP_LINE_LENGTH:
                        self.get_logger().warn("HORIZONTAL WHITE STOP LINE DETECTED")
                        # Optional: Draw line for debugging
                        cv2.line(original, (x1, y1 + int(0.7 * height)), (x2, y2 + int(0.7 * height)), (255, 0, 0), 3)
                        cv2.imshow("Stop Line Detection", original)
                        cv2.waitKey(1)
                        return True

            # Show debug window 
            cv2.imshow("Stop Line Detection", original)
            cv2.waitKey(1)
            
            return False
            
        except Exception as e:
            self.get_logger().error(f"Error in stop line detection: {e}")
            return False
    
    # FUNCTION TO FIND PARKING SLOT BASED ON CLUSTERING OF POINTS
    def find_parking_slot(self, msg):
        height = msg.height
        width = msg.width

        if height == 0 or width == 0:
            return None, None

        raw_points = []

        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False):
            x, y, z = point

            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            if -2.0 < z < -1.3 and 1.0 < x < 3.0:  # ground-level + in front
                raw_points.append((x, y, z))

        if len(raw_points) < 20:
            return None, None

        # Step 1: Downsample using voxel filter
        voxel_size = 0.05  # meters (5cm grid)
        voxel_grid = {}
        
        for pt in raw_points:
            voxel_key = (
                int(pt[0] / voxel_size),
                int(pt[1] / voxel_size),
                int(pt[2] / voxel_size),
            )
            if voxel_key not in voxel_grid:
                voxel_grid[voxel_key] = pt

        downsampled_points = list(voxel_grid.values())
        if len(downsampled_points) < 20:
            return None, None

        points_np = np.array(downsampled_points)
        points_xy = points_np[:, :2]

        # Step 2: Cluster
        eps = 0.3
        min_samples = 15
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
        labels = clustering.labels_
        unique_labels = set(labels)

        line_centers = []

        # Step 3: Visualization
        debug_img = np.zeros((500, 500, 3), dtype=np.uint8)
        scale = 100
        offset_x = 100
        offset_y = 250

        import random
        random.seed(42)
        color_map = {}

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = points_xy[labels == label]
            if len(cluster_points) < 10:
                continue

            center = np.mean(cluster_points, axis=0)
            line_centers.append(center)

            color = tuple(random.randint(100, 255) for _ in range(3))
            color_map[label] = color

            for x, y in cluster_points:
                img_x = int(x * scale) + offset_x
                img_y = int(y * scale) + offset_y
                if 0 <= img_x < 500 and 0 <= img_y < 500:
                    debug_img[img_y, img_x] = color

        # try:
        #     cv2.imshow("Yellow Clusters Debug", debug_img)
        #     cv2.waitKey(1)
        # except:
        #     pass

        return line_centers, debug_img


    # HELPER FUNCTION TO VALIDATE PARKING SLOT
    def validate_parking_slot(self, line_centers):
        if not line_centers or len(line_centers) < 2:
            return None, None, 0, None
            
        # Sort by y coordinate
        line_centers.sort(key=lambda c: c[1])
        
        # Check all possible pairs for valid parking slots
        for i in range(len(line_centers)):
            for j in range(i+1, len(line_centers)):
                left_line = line_centers[i]  
                right_line = line_centers[j]  
                
                slot_width = abs(right_line[1] - left_line[1])
                lane_center_x = (left_line[0] + right_line[0]) / 2
                lane_center_y = (left_line[1] + right_line[1]) / 2
                lane_center = [lane_center_x, lane_center_y]
                num_lines = len(line_centers)
                
                return lane_center, slot_width, num_lines, line_centers
                    

    # FUNCTION FOR CALCULATING CONTROL COMMANDS
    def calculate_alignment_command(self, line_centers):
        cmd = Twist()
        if not line_centers or len(line_centers) < 2:
            # Default behavior when lines not visible
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
            return cmd
        
        # Find the best pair of lines for parking slot
        lane_center, slot_width, num_lines, all_centers = self.validate_parking_slot(line_centers)
        self.get_logger().info(f" lane center : {lane_center}, slot width : {slot_width}, No of lines detected: {num_lines}")
        
        if num_lines > 2:
            cmd.linear.x = 0.3
            cmd.angular.z = 0.1
            return cmd
        
        # Store target for consistency and avoiding jitteryness
        if self.target_slot_center_y is None:
            self.target_slot_center_y = lane_center[1]
        else:
            self.target_slot_center_y = 0.8 * self.target_slot_center_y + 0.2 * lane_center[1]
        
        # Move towards target
        target = [lane_center[0], lane_center[1]]
        cmd = self.calculate_normal_velocity(target)
        
        self.get_logger().info(f"[ALIGNMENT] Lines: {num_lines}, Lane Center: ({lane_center[0]:.2f}, {lane_center[1]:.2f}), Width: {slot_width:.2f}, Cmd: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}")
        return cmd
    
    # FUNCTION TO CALCULATE VELOCITY
    def calculate_normal_velocity(self, target):
        cmd = Twist()
        # Compute direction to target
        angle_to_target = math.atan2(target[1], target[0]) 
        
        # Move toward target
        if(target[0] < 2.2 and target[0] > 1.8):
            if (abs(angle_to_target) > 0.2 ):
                cmd.linear.x = (LINEAR_SPEED - abs(angle_to_target) * ANGLE_FACTOR)/3 
                cmd.angular.z = angle_to_target/2
            else:
                cmd.linear.x = LINEAR_SPEED - abs(angle_to_target) * ANGLE_FACTOR 
                cmd.angular.z = angle_to_target/5 
        else:
            cmd.linear.x = LINEAR_SPEED - abs(angle_to_target) * ANGLE_FACTOR 
            cmd.angular.z = angle_to_target  
        
        return cmd

    # TURNING LOGIC TO ALIGN WITH THE PARKING LOT
    def calculate_turning_command(self, line_centers):
        cmd = Twist()
        if not line_centers or len(line_centers) < 2:
            # Keep turning if lines not visible
            cmd.linear.x = 0.0
            cmd.angular.z = 0.15
            return cmd
            
        lane_center, slot_width, num_lines, all_centers = self.validate_parking_slot(line_centers)
        
        if num_lines == 2:
            # Check if we're well aligned
            if abs(lane_center[1]) < 0.5 and 3.0 < slot_width < 6.0:
                self.get_logger().info(f"Good alignment achieved! Lane Center: ({lane_center[0]:.2f}, {lane_center[1]:.2f}), Width: {slot_width:.2f}")
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

        if self.state == "DRIVING_IN_SLOT":
            if self.detect_horizontal_stop_line():  
                self.get_logger().warn("STOP line detected. Stopping bot.")
                self.state = "STOPPED"
                stop = Twist()
                self.cmd_pub.publish(stop)
                return
            else:
                cmd = self.calculate_alignment_command(line_centers)
                if cmd.angular.z != 0.0:  
                    self.get_logger().info("Fine-tuning alignment while driving in slot")
                else:
                    cmd.linear.x = 0.4  
                    cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                return

        if self.state == "TURNING":
            turn_cmd = self.calculate_turning_command(line_centers)
            
            if turn_cmd is None: 
                self.get_logger().info("Alignment complete - starting to drive into slot")
                self.state = "DRIVING_IN_SLOT"
                self.alignment_count = 0  
                return
            else:
                self.cmd_pub.publish(turn_cmd)
                return

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
                # Check for valid parking slot instead of just any second line
                lane_center, slot_width, num_lines, all_centers = self.validate_parking_slot(line_centers)
                
                if num_lines == 2:
                    self.get_logger().info(f"Valid parking slot found! Lane Center: ({lane_center[0]:.2f}, {lane_center[1]:.2f}), Width: {slot_width:.2f}")
                    self.state = "TURNING"
                    self.target_slot_center_y = lane_center[1]
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