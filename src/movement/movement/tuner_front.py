import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class BackCameraHSV(Node):
    def __init__(self):
        super().__init__('front_camera_hsv_tuner')
        
        # Subscribe to the back camera image topic from Gazebo
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',  
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        
        # Create trackbar window
        self.create_trackbar_window()

    def create_trackbar_window(self):
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("H_min", "HSV Tuner", 20, 180, lambda x: None)
        cv2.createTrackbar("H_max", "HSV Tuner", 40, 180, lambda x: None)
        cv2.createTrackbar("S_min", "HSV Tuner", 100, 255, lambda x: None)
        cv2.createTrackbar("S_max", "HSV Tuner", 255, 255, lambda x: None)
        cv2.createTrackbar("V_min", "HSV Tuner", 100, 255, lambda x: None)
        cv2.createTrackbar("V_max", "HSV Tuner", 255, 255, lambda x: None)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # Get HSV threshold values from trackbars
        h_min = cv2.getTrackbarPos("H_min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H_max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S_min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S_max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V_min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V_max", "HSV Tuner")

        # Convert to HSV and create mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Display both original and mask
        cv2.imshow("Back Camera Feed", frame)
        cv2.imshow("HSV Mask", mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = BackCameraHSV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
