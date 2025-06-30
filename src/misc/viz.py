#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
class VelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_visualizer')
        self.subscription = self.create_subscription(Twist, '/cmd_vel_nav', self.image_callback, 10)
        with open('dwb.csv', 'w') as file:
            pass
    def image_callback(self, msg):
        with open('dwb.csv', 'a') as file:
            spamwriter = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([msg.linear.x, msg.angular.z])

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = VelSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
