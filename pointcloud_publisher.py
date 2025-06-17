#!/usr/bin/env python3
"""
Subscribe to /camera/points, print a quick summary, rotate the cloud, and
re-publish it on /igvc/transformed_pointcloud.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from itertools import islice

# # -90° rotation along Y
# ROT_MATRIX = np.array([[0, 0, -1],
#                        [0, 1, 0],
#                        [1, 0, 0]], dtype=np.float32)

# # -90° rotation along X
# ROT_MATRIX = np.array([[1, 0, 0],
#                        [0, 0, 1],
#                        [0, -1, 0]], dtype=np.float32)

# -90° rotation along Z
ROT_MATRIX = np.array([[1, 0, 0],
                       [0, 0, 1],
                       [0, -1, 1]], dtype=np.float32)

class PointCloudRotator(Node):
    def __init__(self):
        super().__init__("pointcloud_rotator")
        self.subscription = self.create_subscription(
            PointCloud2, "/camera/points", self._callback, 10
        )
        self.publisher = self.create_publisher(PointCloud2,
                                               "/igvc/transformed_pointcloud", 10)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _transform_xyz(points_np: np.ndarray) -> np.ndarray:
        """Apply fixed 3×3 rotation to an (N, 3) float32 array."""
        return (ROT_MATRIX @ points_np.T).T

    @staticmethod
    def _first_finite(points_iter, n=5):
        """Yield first n finite XYZ triples for human-readable peek."""
        for x, y, z in points_iter:
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                yield x, y, z
                n -= 1
                if n == 0:
                    break

    # ------------------------------------------------------------------- main
    def _callback(self, msg: PointCloud2):
        # -------- console peek (unchanged) ------------------------------------
        hdr = msg.header
        self.get_logger().info(
            f"[{hdr.stamp.sec}.{hdr.stamp.nanosec:09d}] "
            f'frame="{hdr.frame_id}"  {msg.width}×{msg.height}  '
            f'fields={[f.name for f in msg.fields]}'
        )

        # sample five points for the log
        # pts_preview = pc2.read_points(msg, ("x", "y", "z"), skip_nans=True)
        # for i, (x, y, z) in enumerate(pts_preview):
        #     self.get_logger().info(f"  p{i}: ({x:.3f}, {y:.3f}, {z:.3f})")
        # self.get_logger().info("—" * 20)

        # -------- full transform & publish ------------------------------------
        pts_np = np.array([(x, y, z)
                        for x, y, z in pc2.read_points(
                            msg, ("x", "y", "z"), skip_nans=True)],
                        dtype=np.float32)               # <--  fix here
        if pts_np.size == 0:
            return

        pts_rot = (ROT_MATRIX @ pts_np.T).T
        out_cloud = pc2.create_cloud_xyz32(hdr, pts_rot)
        self.publisher.publish(out_cloud)



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