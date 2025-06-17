#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LaneOverlayNode(Node):
    def __init__(self):
        super().__init__('lane_overlay_node')
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/lane_overlay/image',
            10
        )

        # Parameters for lane detection
        self.roi_height_ratio = 0.69  # Focus on bottom 50% of image
        self.get_logger().info("Lane Overlay Node initialized - focusing on bottom half of image")

    def image_callback(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Get image dimensions
        height, width = bgr.shape[:2]
        
        # Define ROI - bottom half of the image
        roi_start = int(height * (1 - self.roi_height_ratio))
        roi_bgr = bgr[roi_start:height, :]
        
        # Convert ROI to HSV for better color detection
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Also work with grayscale for better dim white/grey detection
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Method 1: HSV-based detection for whiter lanes
        # Adjusted ranges to catch dim white and grey lanes
        lower_white_hsv = np.array([0, 0, 120])    # Lower brightness threshold for dim whites
        upper_white_hsv = np.array([180, 60, 255]) # Higher saturation tolerance
        mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

        # Method 2: Grayscale threshold for grey lanes
        # Detect pixels that are brighter than the average in the ROI
        avg_brightness = np.mean(gray)
        brightness_threshold = max(100, int(avg_brightness * 0.8))  # Adaptive threshold
        _, mask_gray = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Method 3: Selective edge detection - only edges of bright/white areas
        edges = cv2.Canny(gray, 30, 150)
        
        # Create a mask for bright areas (potential white lanes)
        _, bright_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        
        # Only keep edges that are near bright/white areas
        # Dilate the bright mask to include nearby edges
        kernel = np.ones((5, 5), np.uint8)
        dilated_bright = cv2.dilate(bright_mask, kernel, iterations=2)
        
        # Filter edges: only keep edges that are within dilated bright areas
        filtered_edges = cv2.bitwise_and(edges, dilated_bright)
        
        # Make the filtered edges thicker for better visibility
        kernel = np.ones((3, 3), np.uint8)
        filtered_edges = cv2.dilate(filtered_edges, kernel, iterations=1)

        # Combine all detection methods
        combined_mask = cv2.bitwise_or(mask_hsv, mask_gray)
        combined_mask = cv2.bitwise_or(combined_mask, filtered_edges)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Create full-size mask (same size as original image)
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[roi_start:height, :] = combined_mask

        # Create overlay with blue color for detected lanes
        overlay = bgr.copy()
        overlay[full_mask > 0] = [255, 100, 0]  # Blue color (BGR format)

        # Blend the original image with the overlay
        blended = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
        
        # Draw ROI boundary for visualization
        cv2.rectangle(blended, (0, roi_start), (width, height), (0, 255, 0), 2)
        
        # Add text information
        # cv2.putText(blended, f"ROI: Bottom {int(self.roi_height_ratio*100)}%", 
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(blended, f"Avg Brightness: {int(avg_brightness)}", 
        #            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(blended, f"Threshold: {brightness_threshold}", 
        #            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Publish the processed image
        output_msg = self.bridge.cv2_to_imgmsg(blended, encoding='bgr8')
        self.image_pub.publish(output_msg)

        # Display the result
        cv2.imshow("Lane Overlay", blended)
        
        # Optional: Show the mask for debugging
        cv2.imshow("Lane Mask", full_mask)
        
        cv2.waitKey(1)

    def update_roi_ratio(self, new_ratio):
        """Method to dynamically adjust ROI if needed"""
        self.roi_height_ratio = max(0.1, min(1.0, new_ratio))
        self.get_logger().info(f"ROI height ratio updated to: {self.roi_height_ratio}")


def main(args=None):
    rclpy.init(args=args)
    node = LaneOverlayNode()
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