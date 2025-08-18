import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
import math

class ScanToPointCloud(Node):

    def __init__(self):
        super().__init__('scan_to_pointcloud')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/lidar_pc2', 10)

    def scan_callback(self, msg: LaserScan):
        points = []
        angle = msg.angle_min

        for r in msg.ranges:
            # Ignore invalid ranges (e.g. 0.0 or inf)
            if r > msg.range_min and r < msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                z = 0.0
                points.append((x, y, z))
            angle += msg.angle_increment

        # Create PointCloud2 with XYZ fields (float32)
        pc2_msg = point_cloud2.create_cloud_xyz32(
            header=self._make_header(),
            points=points
        )

        self.publisher.publish(pc2_msg)

    def _make_header(self):
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"
        return header


def main(args=None):
    rclpy.init(args=args)
    node = ScanToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
