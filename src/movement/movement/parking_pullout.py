import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import math

MAX_STOP_LINE_LENGTH = 100  
WHITE_THRESHOLD = 100
COLOR_BALANCE_THRESHOLD = 25
MIN_CLUSTERING_DISTANCE = 0.971
MIN_CLUSTERING_POINTS = 20

class ParkingPullout(Node):
    def __init__(self):
        super().__init__('yellow_parking_detector')
        self.subscription1 = self.create_subscription(Image, '/camera/image', self.image_callback_1, 10)
        self.subscription2 = self.create_subscription(PointCloud2, '/bcamera/points_downsampled', self.pointcloud_callback, 10)
        self.bridge = CvBridge()
        self.latest_image = None  
        self.pmsg = None
        self.direction = 'left'
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state = "IN_SLOT"
        self.k = 0
        self.turn_timer = None
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.barrel_found = False

    def pull_back(self):
        cmd = Twist()
        cmd.linear.x = -0.5
        cmd.angular.z = 0.0
        self.get_logger().info("Pulling back to see lines")
        self.cmd_pub.publish(cmd)

    def turn(self, direction_factor):
        cmd = Twist()
        cmd.linear.x = -0.5
        cmd.angular.z = direction_factor*0.4
        self.cmd_pub.publish(cmd) 

    def stop_robot(self):
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_pub.publish(stop_cmd)
        self.get_logger().info("Turn complete, robot stopped")
        self.state = "FINISHED"

        if self.turn_timer is not None:
            self.turn_timer.cancel()
            self.turn_timer = None

    def image_callback_1(self, msg):
        self.latest_image = msg
    
    def pointcloud_callback(self, msg):
        self.pmsg = msg 

    def detect_horizontal_stop_line(self):
        if self.latest_image is None:
            return False
        try:
            frame1 = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            original = frame1.copy()
            height = frame1.shape[0]
            frame = frame1[int(0.5 * height):, :]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([15, 30, 60])
            upper_yellow = np.array([30, 120, 200])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            stop_edges_yellow = cv2.Canny(mask_yellow, 50, 150)
            stop_lines_yellow = cv2.HoughLinesP(stop_edges_yellow, 1, np.pi / 180, threshold=30,
                                         minLineLength=20, maxLineGap=10)

            if stop_lines_yellow is not None:
                for line in stop_lines_yellow:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if (angle < 10 or angle > 170) and length < MAX_STOP_LINE_LENGTH:
                        self.get_logger().warn("HORIZONTAL YELLOW STOP LINE DETECTED")
                        cv2.line(original, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.imshow("Stop Line Detection", original)
                        cv2.waitKey(1)
                        return True
                    
            cv2.imshow("Stop Line Detection", original)
            cv2.waitKey(1)
            return False
            
        except Exception as e:
            self.get_logger().error(f"Error in stop line detection: {e}")
            return False

    def detect_yellow_color(self, r, g, b):
        pale_threshold = 80  
        color_balance_threshold = 30  
        avg_color = (r + g + b) / 3
        is_bright_enough = (r > pale_threshold and g > pale_threshold)
        has_yellow_tint = (r > b and g > b)  
        is_balanced = (abs(r - avg_color) < color_balance_threshold and abs(g - avg_color) < color_balance_threshold)
        yellow_factor = (r + g) / (2 * max(b, 1)) 
        
        return (is_bright_enough and has_yellow_tint and is_balanced and yellow_factor > 1.1 and avg_color > 70)  

    def find_parking_slot(self, msg):
        height = msg.height
        width = msg.width
        
        if height == 0 or width == 0:
            return None, None

        yellow_img = np.zeros((height, width, 3), dtype=np.uint8)
        yellow_ground_points = []

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

            if self.detect_yellow_color(r, g, b):
                if -2.0 < z < -1.3 and 1.0 < x < 3.0:
                    if 0 <= row < height and 0 <= col < width:
                        yellow_img[row, col] = (0, 255, 255)  
                    yellow_ground_points.append([x, y, z])
            index += 1

        try:
            if yellow_img.size > 0:
                cv2.imshow("Pale Yellow Parking Lines", yellow_img)
                cv2.waitKey(1)
        except:
            pass

        if len(yellow_ground_points) < 20:
            return None, yellow_img

        points_np = np.array(yellow_ground_points)
        points_xy = points_np[:, :2]  

        eps = 0.3  
        min_samples = 15
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
        labels = clustering.labels_
        unique_labels = set(labels)

        line_centers = []
        for label in unique_labels:
            if label == -1: 
                continue
                
            cluster_points = points_xy[labels == label]
            if len(cluster_points) < 10:
                continue
                
            center = np.mean(cluster_points, axis=0)
            line_centers.append(center)

        return line_centers, yellow_img

    def control_loop(self):
        if self.state == "IN_SLOT":
            if self.detect_horizontal_stop_line():
                self.pull_back()
            else:
                self.state = "NEARLY_OUT"
        
        if self.state == "NEARLY_OUT":
            if self.pmsg is not None:
                line_centers, _ = self.find_parking_slot(self.pmsg)
                if line_centers:
                    self.pull_back()
                else:
                    stop_cmd = Twist()
                    stop_cmd.linear.x = 0.0
                    stop_cmd.angular.z = 0.0 
                    self.get_logger().info("Nearly outside the parking lot")
                    self.cmd_pub.publish(stop_cmd)
                    self.state = "TURNING"
            else:
                self.get_logger().warn("No point cloud data available yet.")

        if self.state == "TURNING":
            if self.direction == 'right':
                self.k = 1
            else:
                self.k = -1
            self.turn(self.k)
            self.turn_timer = self.create_timer(23, self.stop_robot)    
            self.state = "TURNING_IN_PROGRESS"
        
        elif self.state == "TURNING_IN_PROGRESS":
            if self.detect_horizontal_stop_line():
                self.get_logger().info("Stop line re-detected — turn complete.")
                self.stop_robot()
            else:
                self.turn(self.k)  

def main(args=None):
    rclpy.init(args=args)
    node = ParkingPullout()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
