# lane_mask_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from sensor_msgs.msg import PointField
import struct

class LaneMaskNode(Node):
    def __init__(self):
        super().__init__('lane_mask_node')
        self.sub = self.create_subscription(PointCloud2, '/igvc/transformed_pointcloud', self.pc_callback, 10)
        self.pub = self.create_publisher(PointCloud2, '/igvc/lane_mask', 10)

    def pc_callback(self, msg):
        points = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        white_points = []

        for p in points:
            rgb = struct.unpack('I', struct.pack('f', p[3]))[0]
            r = (rgb >> 16) & 0xFF
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF

            if r > 170 and g > 170 and b > 170:  # White threshold
                white_points.append([p[0], p[1], p[2]])

        if not white_points:
            return

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        new_pc = pc2.create_cloud(header, fields, white_points)
        self.pub.publish(new_pc)

def main():
    rclpy.init()
    node = LaneMaskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
