import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ConstantVelocityPublisher(Node):
    def __init__(self):
        super().__init__('constant_velocity_publisher')

        # Create publisher to /cmd_vel topic
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer to publish velocity every 0.1 seconds
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_velocity)

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = 0.2  # Move forward with 0.2 m/s
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        self.publisher_.publish(msg)
        self.get_logger().info('Publishing forward velocity: %.2f m/s' % msg.linear.x)

def main(args=None):
    rclpy.init(args=args)
    node = ConstantVelocityPublisher()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
