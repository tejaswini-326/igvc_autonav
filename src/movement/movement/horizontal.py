#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg


MAX_STOP_LINE_LENGTH = 100 
THRESHOLD_ANGLE = 20


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
        self.depth_sub = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/horizontal_line', 10)
        self.fx = 246.4928
        self.fy = 246.4928
        self.cx = 300.0
        self.cy = 300.0

    def depth_callback(self, msg):
            try:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1') #load and decode depth image
            except Exception as e:
                self.get_logger().error(f'Error converting depth image: {e}')

    def image_callback(self, msg):
        
        # Convert ROS Image to OpenCV BGR format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        original = frame.copy()

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow mask (dashed center lane)
        lower_yellow = np.array([18, 80, 80])
        upper_yellow = np.array([30, 255, 255])
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
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=40)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Identify if line is yellow or white
                if mask_yellow[y1, x1] > 0:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 255, 255)  # White
                cv2.line(original, (x1, y1), (x2, y2), color, 2)

        # Stop line detection (from white mask only)
        stop_edges = cv2.Canny(mask_white, 50, 150)
        stop_lines = cv2.HoughLinesP(stop_edges, 1, np.pi / 180, threshold=30,
                                     minLineLength=20, maxLineGap=10)
        stop_points_3d = []

        if stop_lines is not None:
            for line in stop_lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                if (angle < THRESHOLD_ANGLE or angle > 180 - THRESHOLD_ANGLE) and length < MAX_STOP_LINE_LENGTH:
                    cv2.line(original, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    # Sample N points along the line between (x1, y1) and (x2, y2)
                    N = int(length)  # One pixel per unit length
                    for i in range(N + 1):
                        u = int(x1 + (x2 - x1) * i / N)
                        v = int(y1 + (y2 - y1) * i / N)

                        # Check pixel bounds
                        if 0 <= v < self.depth_image.shape[0] and 0 <= u < self.depth_image.shape[1]:
                            z = self.depth_image[v, u]
                            if np.isfinite(z) and z > 0.1:  # Ignore NaNs or very low values
                                x = (u - self.cx) * z / self.fx
                                y = (v - self.cy) * z / self.fy
                                stop_points_3d.append([z, -x, -y])  # Your preferred coordinate frame

            if stop_points_3d:
                stop_points_3d = np.array(stop_points_3d)
                header = std_msgs.msg.Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "camera_link"  # Set this to your TF frame

                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
                ]

                pointcloud_msg = pc2.create_cloud(header, fields, stop_points_3d)
                self.pointcloud_pub.publish(pointcloud_msg)

                # Find xmin, xmax
                xmin = np.min(stop_points_3d[:, 1])  # x is second in [z, -x, -y]
                xmax = np.max(stop_points_3d[:, 1])
                self.get_logger().info(f"Stop line X range: xmin={xmin:.2f}, xmax={xmax:.2f}, Total points: {len(stop_points_3d)}")

    
                

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