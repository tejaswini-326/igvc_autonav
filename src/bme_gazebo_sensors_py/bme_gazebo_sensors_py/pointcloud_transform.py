#!/usr/bin/env python3
"""
Subscribe to /camera/points, print a quick summary, broadcast a static TF
(camera_link_optical → camera_corrected) that performs the same rotation you
previously hard-coded, and re-publish the cloud on
/igvc/transformed_pointcloud with the new frame_id (RGB preserved).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import quaternion_from_matrix
import numpy as np
from copy import deepcopy

# --------------------------------------------------------------------------- #
#  Rotation matrices (same as your old script)                                #
# --------------------------------------------------------------------------- #
ROT_MATRIX_X = np.array([[0, 0, -1],
                         [0, 1,  0],
                         [1, 0,  0]], dtype=np.float32)   # -90° Y  (camera axis)

ROT_MATRIX_Y = np.array([[1, 0,  0],
                         [0, 0,  1],
                         [0,-1,  0]], dtype=np.float32)   # -90° X

ROT_MATRIX_Z = np.array([[0,-1,  0],
                         [1, 0,  0],
                         [0, 0,  1]], dtype=np.float32)   # +90° Z

ROT_MATRIX = (
    ROT_MATRIX_Z @ ROT_MATRIX_Z @   # (your original two Z’s)
    ROT_MATRIX_X @ ROT_MATRIX_X @   # two X’s
    ROT_MATRIX_Y @ ROT_MATRIX_Z     # one Y then one Z
)                                   # final 3×3 rotation

# Convert to quaternion once at startup
_H = np.eye(4, dtype=np.float32)
_H[:3, :3] = ROT_MATRIX
ROT_QUAT = quaternion_from_matrix(_H)   # (x, y, z, w)

# --------------------------------------------------------------------------- #
#  Node                                                                       #
# --------------------------------------------------------------------------- #
class PointCloudTFRepublisher(Node):
    def __init__(self):
        super().__init__("pointcloud_tf_republisher")

        # 1. TF broadcaster ---------------------------------------------------
        self.broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()

        # 2. Point-cloud I/O --------------------------------------------------
        self.sub = self.create_subscription(
            PointCloud2, "/camera/points", self._callback, 10
        )
        self.pub = self.create_publisher(
            PointCloud2, "/igvc/transformed_pointcloud", 10
        )

        self._first_log_done = False

    # --------------------------------------------------------------------- #
    #  STATIC TF (runs once)                                                #
    # --------------------------------------------------------------------- #
    def _broadcast_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id  = "camera_link_optical"   # parent  (unchanged)
        t.child_frame_id   = "camera_corrected"      # child   (rotated view)

        # no translation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # rotation → quaternion
        t.transform.rotation.x = float(ROT_QUAT[0])
        t.transform.rotation.y = float(ROT_QUAT[1])
        t.transform.rotation.z = float(ROT_QUAT[2])
        t.transform.rotation.w = float(ROT_QUAT[3])

        # Send once; tf2 will treat it as /tf_static
        self.broadcaster.sendTransform(t)
        self.get_logger().info("Published static TF camera_link_optical → camera_corrected")

    # --------------------------------------------------------------------- #
    #  Point-cloud callback                                                 #
    # --------------------------------------------------------------------- #
    def _callback(self, msg: PointCloud2):
        # quick header summary (unchanged)
        hdr = msg.header
        self.get_logger().info(
            f"[{hdr.stamp.sec}.{hdr.stamp.nanosec:09d}] "
            f'frame="{hdr.frame_id}"  {msg.width}×{msg.height}  '
            f'fields={[f.name for f in msg.fields]}'
        )

        if not self._first_log_done:
            # show one point just to prove RGB is intact
            pts = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
            if pts:
                self.get_logger().info(f"  sample point (raw): {pts[0]}")
            self._first_log_done = True

        # copy the message, change only the frame_id
        cloud_out = deepcopy(msg)
        cloud_out.header.frame_id = "camera_corrected"
        self.pub.publish(cloud_out)

# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main(args=None):
    rclpy.init(args=args)
    node = PointCloudTFRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()