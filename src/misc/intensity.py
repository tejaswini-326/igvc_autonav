#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('pixel_coord')
        self.subscription = self.create_subscription(Image, '/zed/zed_node/rgb/image_rect_color', self.image_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print(image.shape)
        cv2.imshow('Image', image)
        cv2.setMouseCallback('Image', self.on_mouse, image)
        cv2.waitKey(1)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            intensity = self.get_intensity(param, x, y)
            print(f"Clicked at pixel coordinates (x={x}, y={y}), Intensity={intensity}")
    
    def get_intensity(self, image, x, y):
        return image[y, x]

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
