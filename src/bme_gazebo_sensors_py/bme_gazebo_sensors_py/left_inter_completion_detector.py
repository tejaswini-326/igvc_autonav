from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion
import math
import rclpy
from rclpy.node import Node


class LeftIntersectionDetector(Node):
    def __init__(self):
        super().__init__('LeftIntersectionDetector')
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(String, '/intersection', self.intersection_cb, 10)
        self.intersection_pub = self.create_publisher(String, '/intersection', 10)
        self.active = False
        self.turning = False
        self.turn_start_position = None
        self.turn_start_yaw = None
        self.target_displacement = 6 # the gazebo value is 6
        
        self.switched_to_b = False

    def intersection_cb(self, msg):
        if msg.data.lower() == "left":
            self.active = True
            self.get_logger().info("🟢 'left' received — LeftIntersectionDetector activated.")
        else:
            self.active = False
            self.get_logger().info(f"🔴 Received '{msg.data}' — ignoring in LeftIntersectionDetector.")

    def odom_cb(self, msg):
        if not self.active or self.switched_to_b:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if not self.turning:
            # 🚦 Start of turn detected — store pose
            self.turning = True
            self.turn_start_position = (x, y)
            self.turn_start_yaw = yaw
            self.get_logger().info("🔁 Starting right turn…")
            return

        if self.switched_to_b:
            return

        # Compute rightward displacement
        dx = x - self.turn_start_position[0]
        dy = y - self.turn_start_position[1]

        # Rotate displacement into robot's starting frame
        # (to project motion onto "right" axis)
        rightward_disp = dx * math.sin(-self.turn_start_yaw) + dy * math.cos(-self.turn_start_yaw)

        self.get_logger().info(f"📏 Rightward displacement: {rightward_disp:.2f} m")

        if rightward_disp >= self.target_displacement:
            self.get_logger().info("✅ Turn complete — launching file B")
            msg = String()
            msg.data = "None"  # or "none" or "done" — your choice
            self.intersection_pub.publish(msg)
            #you want to publish None to the topic /intersection
            self.switched_to_b = True
            self.active = False  # deactivate after completion
def main(args=None):
    rclpy.init(args=args)
    node = LeftIntersectionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 KeyboardInterrupt — shutting down…")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()