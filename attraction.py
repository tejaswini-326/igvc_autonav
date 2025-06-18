import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import math

class AttractiveForceBot(Node):
    def __init__(self):
        super().__init__('attractive_force_bot')
        
        # Debug: Confirm node creation
        self.get_logger().info('AttractiveForceBot node created')
        
        # Target point to be attracted to
        self.target_x = -20.34
        self.target_y = -1.03
        self.k = 1.0  # gain
        
        # Create publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Publisher created for /cmd_vel')
        
        # Try different QoS profiles
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscription with QoS
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            qos_profile
        )
        self.get_logger().info('Subscription created for /odom with BEST_EFFORT QoS')
        
        # Counter for received messages
        self.msg_count = 0
        
        # Timer for debugging
        self.create_timer(5.0, self.debug_status)
        
    def odom_callback(self, msg):
        self.msg_count += 1
        self.get_logger().info(f'Received odometry message #{self.msg_count}')
        print(f"Callback executed! Message count: {self.msg_count}")
        
        # Get current position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        self.get_logger().info(f'Current position: ({x:.2f}, {y:.2f})')
        
        # Calculate attractive force
        dx = self.target_x - x
        dy = self.target_y - y
        dist_sq = dx**2 + dy**2
        distance = math.sqrt(dist_sq)
        
        self.get_logger().info(f'Distance to target: {distance:.2f}')
        
        if dist_sq < 0.01:
            self.get_logger().info('Target reached!')
            return  # close enough, stop moving
        
        # ATTRACTIVE FORCE: Proportional to distance (not inverse!)
        # This creates a spring-like attraction to the target
        force_x = self.k * dx  # Simple proportional control
        force_y = self.k * dy
        
        # Optional: Cap maximum velocity to prevent overshooting
        max_speed = 0.5  # m/s
        force_magnitude = math.sqrt(force_x**2 + force_y**2)
        if force_magnitude > max_speed:
            force_x = force_x / force_magnitude * max_speed
            force_y = force_y / force_magnitude * max_speed
        
        # Create and publish twist message
        twist = Twist()
        twist.linear.x = force_x
        twist.linear.y = force_y  # if your robot supports holonomic motion
        
        self.cmd_pub.publish(twist)
        self.get_logger().info(f'Published force: ({force_x:.3f}, {force_y:.3f})')
        
    def debug_status(self):
        publisher_count = self.odom_sub.get_publisher_count()
        self.get_logger().info(f'Publisher count for /odom: {publisher_count}')
        self.get_logger().info(f'Total messages received: {self.msg_count}')
        
        if publisher_count == 0:
            self.get_logger().warn('No publishers found for /odom topic!')
            self.get_logger().info('Try running: ros2 topic list | grep odom')

def main(args=None):
    rclpy.init(args=args)
    
    node = AttractiveForceBot()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()