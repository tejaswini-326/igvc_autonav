import rclpy  # use rospy for ROS 1
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

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

        # Enhance brightness
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        v = np.clip(v, 0, 255)
        hsv_enhanced = cv2.merge((h, s, v))
        bright_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        # Apply CLAHE for contrast
        lab = cv2.cvtColor(bright_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        bright_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # OCR
        results = self.reader.readtext(bright_image)

        for (bbox, text, prob) in results:
            text = text.strip().upper()
            (tl, tr, br, bl) = map(lambda x: tuple(map(int, x)), bbox)

            cv2.rectangle(cv_image, tl, br, (0, 255, 0), 2)
            cv2.putText(cv_image, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            is_stop = "STOP" in text
            is_red = False

            if is_stop:
                print("🛑 STOP text detected")

                # Extend bounding box vertically
                x_min = min(tl[0], bl[0])
                x_max = max(tr[0], br[0])
                y_min = min(tl[1], tr[1])
                height = max(bl[1], br[1]) - y_min
                y_max = y_min + 2 * height
                y_max = min(y_max, cv_image.shape[0])  # Ensure within image

                # Crop extended region
                extended_region = cv_image[y_min:y_max, x_min:x_max]

                # Convert to HSV to check red
                hsv_ext = cv2.cvtColor(extended_region, cv2.COLOR_BGR2HSV)
                # Range 1: Red on the lower end of hue scale
                lower_red1 = np.array([0, 70, 50])     # hue, sat, val
                upper_red1 = np.array([10, 255, 255])

                # Range 2: Red on the upper end of hue scale
                lower_red2 = np.array([160, 70, 50])
                upper_red2 = np.array([180, 255, 255])
                mask1 = cv2.inRange(hsv_ext, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv_ext, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)

                red_pixels = cv2.countNonZero(red_mask)
                total_pixels = red_mask.shape[0] * red_mask.shape[1] + 1e-5
                red_ratio = red_pixels / total_pixels
                red_percent = red_ratio * 100

                print(f"🔍 Red ratio: {red_percent:.2f}%")

                if red_ratio > 0.2:
                    is_red = True
                    print("🔴 Red detected below STOP text")

                # Display cropped region using matplotlib
                plt.figure("Extended Region")
                rgb_crop = cv2.cvtColor(extended_region, cv2.COLOR_BGR2RGB)
                plt.imshow(rgb_crop)
                plt.title(f"Red Area: {red_percent:.1f}%")
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.1)
                plt.clf()

            if is_stop and is_red:
                print("✅ STOP SIGN detected!")

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
