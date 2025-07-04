import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs
import os 

class Goal_visualizer(Node):
    def __init__(self):
        super().__init__("goal_viz_node")
        self.goal_sub = self.create_subscription(Point, '/goal_point',self.goal_cb, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/goal_point_viz', 10)
        self.get_logger().info("Goal Visualizer Node Initialized")
        
    def goal_cb(self, msg: Point):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "odom"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position = msg
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_stamped)

def main(args=None):
    rclpy.init(args=args)
    node = Goal_visualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()