import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import threading

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        
        self.subscription = self.create_subscription(
            Image,
            'lane_overlay/mask',
            self.image_callback,
            1  
        )
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()          
        self.running = True
        self.spin_thread = threading.Thread(target=self.spin_thread_func)
        self.spin_thread.start()
        self.store = []
        self.index = 0

    def spin_thread_func(self):
        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def stop(self):
        self.running = False
        self.spin_thread.join()

    def image_callback(self, msg):
        with self.frame_lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.store.append(self.latest_frame)
            self.index += 1

    def display_image(self):
        """Main loop to process and display the latest frame."""
        # Create a single OpenCV window
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 800,600)

        while rclpy.ok():
            # Check if there is a new frame available
            if self.latest_frame is not None:
                if (self.index > 1):
                    main_img = cv2.resize(self.store[self.index-2], (160, 120))
                    # Process the current image
                    mask, contour, crosshair = self.process_image(main_img)

                    # Add processed images as small images on top of main image
                    result = self.add_small_pictures(main_img, [mask, contour, crosshair])

                    # Show the latest frame
                    cv2.imshow("frame", result)
                    self.latest_frame = None  # Clear the frame after displaying

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        # Close OpenCV window after quitting
        cv2.destroyAllWindows()
        self.running = False

    def process_image(self, img):
        msg = Twist()

        rows, cols = img.shape[:2]
        binary = img.copy()

        # Use bottom strip of the image
        strip_start = int(rows * 0.5)
        strip_end = rows
        strip = binary[strip_start:strip_end, :]

        # Collapse to 1D by summing vertically
        strip_sum = np.sum(strip, axis=0)
        white_indices = np.where(strip_sum > 0)[0]

        if len(white_indices) >= 2:
            left_edge = white_indices[0]
            right_edge = white_indices[-1]
            lane_center = (left_edge + right_edge) // 2

            crosshair_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.circle(crosshair_mask, (lane_center, (strip_start + strip_end) // 2), 5, (0, 0, 255), -1)
            cv2.line(crosshair_mask, (cols // 2, 0), (cols // 2, rows), (255, 0, 0), 2)
            cv2.line(crosshair_mask, (lane_center, 0), (lane_center, rows), (0, 255, 0), 2)

            # FIXED: Calculate error (positive = need to turn right, negative = need to turn left)
            error = lane_center - (cols // 2)
            
            # FIXED: Always move forward, adjust angular velocity based on error
            msg.linear.x = 0.15  # Constant forward speed (reduced for stability)
            
            # FIXED: Proportional control with reasonable gain
            Kp = 0.005  # Reduced gain to prevent overcorrection
            msg.angular.z = -Kp * error  # Negative because ROS angular.z: positive = left, negative = right
            
            # Optional: Add dead zone to prevent tiny oscillations
            if abs(error) < 5:  # Dead zone of 5 pixels
                msg.angular.z = 0.0
            
            # Optional: Limit maximum angular velocity
            max_angular = 0.6
            msg.angular.z = max(-max_angular, min(max_angular, msg.angular.z))
            
            print(f"Lane center: {lane_center}, Image center: {cols//2}, Error: {error}, Angular: {msg.angular.z:.3f}")
            
        else:
            # No lane detected - stop or search behavior
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            crosshair_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            print("No lanes detected")

        self.publisher.publish(msg)
        
        # Convert binary mask to 3-channel for consistent display
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return binary_color, crosshair_mask, crosshair_mask
    # Convert to RGB channels
    def convert2rgb(self, img):
        R = img[:, :, 2]
        G = img[:, :, 1]
        B = img[:, :, 0]

        return R, G, B

    # Apply threshold and result a binary image
    def threshold_binary(self, img, thresh=(200, 255)):
        binary = np.zeros_like(img)
        binary[(img >= thresh[0]) & (img <= thresh[1])] = 1

        return binary*255
    
    # Add small images to the top row of the main image
    def add_small_pictures(self, img, small_images, size=(160, 120)):
        x_base_offset = 40
        y_base_offset = 10

        x_offset = x_base_offset
        y_offset = y_base_offset

        for small in small_images:
            small = cv2.resize(small, size)
            if len(small.shape) == 2:
                small = np.dstack((small, small, small))

            # Ensure there's enough space in the original image to paste the small image
            if y_offset + size[1] <= img.shape[0] and x_offset + size[0] <= img.shape[1]:
                img[y_offset: y_offset + size[1], x_offset: x_offset + size[0]] = small
                x_offset += size[0] + x_base_offset
            else:
                self.get_logger().warn("Not enough space to display all small images.")
                break
        return img

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    
    try:
        node.display_image()  # Run the display loop
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()  # Ensure the spin thread and node stop properly
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
