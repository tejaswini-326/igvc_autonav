#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

MAX_STOP_LINE_LENGTH = 100 
class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image',  # Update this to match your camera topic
            self.image_callback,
            10)
        
        self.bridge = CvBridge()
        self.get_logger().info("LaneDetectorNode initialized.")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV BGR format
        # Convert ROS Image to OpenCV BGR format
        frame1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        original = frame1.copy()

        # Get image height
        height = frame1.shape[0]

        # Slice bottom 30% of the image
        frame = frame1[int(0.7 * height):, :]

        # Convert cropped region to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # Yellow mask (dashed center lane)
        lower_yellow = np.array([15, 30, 60])
        upper_yellow = np.array([30, 120, 200])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # White mask (side lanes + stop line, excluding yellow)
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([25, 60, 255])
        mask_white_raw = cv2.inRange(hsv, lower_white, upper_white)

        # Clean white by subtracting yellow
        mask_white = cv2.bitwise_and(mask_white_raw, cv2.bitwise_not(mask_yellow))


        # Combined filtered mask
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        filtered = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # Canny edge detection
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Lane lines detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=40, maxLineGap=40)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Identify if line is yellow or white
                if mask_yellow[y1, x1] > 0:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 255, 255)  # White
                # cv2.line(original, (x1, y1), (x2, y2), color, 2)

        # Stop line detection (from white mask only)
        stop_edges = cv2.Canny(mask_white, 50, 150)
        stop_lines = cv2.HoughLinesP(stop_edges, 1, np.pi / 180, threshold=30,
                                     minLineLength=20, maxLineGap=10)

        if stop_lines is not None:
            for line in stop_lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if (angle < 10 or angle > 170) and length < MAX_STOP_LINE_LENGTH:
                    cv2.line(original, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue

        stop_edges_yellow = cv2.Canny(mask_yellow, 50, 150)
        stop_lines_yellow = cv2.HoughLinesP(stop_edges_yellow, 1, np.pi / 180, threshold=30,
                                     minLineLength=20, maxLineGap=10)

        if stop_lines_yellow is not None:
            for line in stop_lines_yellow:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if (angle < 10 or angle > 170) and length < MAX_STOP_LINE_LENGTH:
                    cv2.line(original, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Blue

                

        # Show the final result (for debugging only)
        cv2.imshow("Lane Detection", original)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
	main()