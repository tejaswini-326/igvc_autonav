import rclpy  # use rospy for ROS 1
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import easyocr

class TextDetectorNode(Node):
    def __init__(self):
        super().__init__('text_detector_node')

        # EasyOCR Reader (English)
        self.reader = easyocr.Reader(['en'])

        # ROS <-> OpenCV converter
        self.bridge = CvBridge()

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',  # or '/camera/image_raw'
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Detect text with EasyOCR
        results = self.reader.readtext(cv_image)

        # Draw bounding boxes and labels
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            tl, tr, br, bl = map(lambda x: tuple(map(int, x)), [tl, tr, br, bl])
            cv2.rectangle(cv_image, tl, br, (0, 255, 0), 2)
            cv2.putText(cv_image, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show result
        cv2.imshow("Text Detection", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TextDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
