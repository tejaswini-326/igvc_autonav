from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import rclpy
import os
from rclpy.node import Node
import subprocess  # for launching external files
from std_msgs.msg import Bool

relative_path = "../../bme_gazebo_sensors/move_forward.py"
script_path = os.path.join(os.path.dirname(__file__), relative_path)

class LeftIntersectionDetector(Node):
    def __init__(self):
        super().__init__('LeftIntersectionDetector')
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.shutdown_pub = self.create_publisher(Bool, '/shutdown_signal', 10)
        self.turning = False
        self.turn_start_position = None
        self.turn_start_yaw = None
        self.target_displacement = 3.7 # the gazebo value is 3.7
        
        self.switched_to_b = False

    def odom_cb(self, msg):
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
            subprocess.Popen(["python3",script_path])
            self.shutdown_pub.publish(Bool(data=True))
            self.switched_to_b = True
            self.get_logger().info("🛑 Shutting down RightIntersectionDetector…")
            rclpy.shutdown()
def main(args=None):
    rclpy.init(args=args)
    node =LeftIntersectionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()