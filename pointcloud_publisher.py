#!/usr/bin/env python3
"""
Subscribe to /camera/points, print a quick summary, rotate the cloud, and
re-publish it on /igvc/transformed_pointcloud, keeping RGB values.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

from std_msgs.msg import Header

import struct

# -90° rotation along Y
ROT_MATRIX_X = np.array([[0, 0, -1],
                         [0, 1, 0],
                         [1, 0, 0]], dtype=np.float32)

# -90° rotation along X
ROT_MATRIX_Y = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=np.float32)

# +90° rotation along Z
ROT_MATRIX_Z = np.array([[0, -1, 0],
                         [1, 0, 0],
                         [0, 0, 1]], dtype=np.float32)

ROT_MATRIX = ROT_MATRIX_Z @ ROT_MATRIX_Z @ ROT_MATRIX_X @ ROT_MATRIX_X @ ROT_MATRIX_Y @ ROT_MATRIX_Z

i = 0

class PointCloudRotator(Node):
    def __init__(self):
        super().__init__("pointcloud_rotator")
        self.subscription = self.create_subscription(
            PointCloud2, "/camera/points", self._callback, 10
        )
        self.publisher = self.create_publisher(PointCloud2,
                                               "/igvc/transformed_pointcloud", 10)

    def _callback(self, msg: PointCloud2):
        global i
        hdr = msg.header
        self.get_logger().info(
            f"[{hdr.stamp.sec}.{hdr.stamp.nanosec:09d}] "
            f'frame="{hdr.frame_id}"  {msg.width}×{msg.height}  '
            f'fields={[f.name for f in msg.fields]}'
        )

        # Read x, y, z, rgb from incoming cloud
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        if len(pts) == 0:
            return

        xyz = np.array([[x, y, z] for x, y, z, _ in pts], dtype=np.float32)
        rgb = [rgb_val for _, _, _, rgb_val in pts]

        # Apply rotation to xyz
        rotated_xyz = (ROT_MATRIX @ xyz.T).T

        # Re-pack into (x, y, z, rgb) tuples
        rotated_pts = [(x, y, z, rgb[i]) for i, (x, y, z) in enumerate(rotated_xyz)]

        if i == 0:
            print("Sample point before:", pts[0])
            print("Sample point after:", rotated_pts[0])
            i += 1

        # Define fields for x, y, z, rgb
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Create new cloud with RGB preserved
        cloud_out = pc2.create_cloud(hdr, fields, rotated_pts)
        self.publisher.publish(cloud_out)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudRotator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
