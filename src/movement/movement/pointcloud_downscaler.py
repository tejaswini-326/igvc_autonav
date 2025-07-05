import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import time
import cv2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

WHITE_THRESH   = 90
BALANCE_THRESH = 50
Y_H_MIN, Y_H_MAX = 15, 40
Y_S_MIN = 80
Y_V_MIN = 120


def fast_xyz_white_yellow(msg):
    """
    Fast extractor returning three (N,3) float32 arrays of xyz for:
      1) white points
      2) yellow points
      3) not_black points (i.e. not in HSV range H:0-180, S:0-70, V:0-70)
    """
    n_pts   = msg.width * msg.height
    step    = msg.point_step
    endian  = '>' if msg.is_bigendian else '<'

    # ---- 1. reshape raw buffer to (points, fields) float32 view ----------
    row_floats = step // 4
    cloud_f32  = np.frombuffer(msg.data, dtype=endian + 'f4')
    cloud_f32  = cloud_f32.reshape(n_pts, row_floats)

    fld_idx = {f.name: f.offset // 4 for f in msg.fields}
    xs   = cloud_f32[:, fld_idx['x']]
    ys   = cloud_f32[:, fld_idx['y']]
    zs   = cloud_f32[:, fld_idx['z']]
    rgbf = cloud_f32[:, fld_idx['rgb']]

    # ---- 2. quick spatial window (ground slice just in front of robot) ---
    good = np.isfinite(zs) & (zs > -2.0) & (zs < -1.3)
    if not good.any():
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, empty

    xs, ys, zs, rgbf = xs[good], ys[good], zs[good], rgbf[good]

    # ---- 3. unpack BGR ----------------------------------------------
    rgb_u32 = rgbf.view(np.uint32)
    b = (rgb_u32 & 0xFF).astype(np.uint8)
    g = ((rgb_u32 >> 8)  & 0xFF).astype(np.uint8)
    r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)

    # ---- 4. white detection -----------------------------------------
    avg      = (r.astype(np.int16) + g + b) // 3
    is_white = (
        (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH) &
        (np.abs(r - avg) < BALANCE_THRESH) &
        (np.abs(g - avg) < BALANCE_THRESH) &
        (np.abs(b - avg) < BALANCE_THRESH)
    )

    # ---- 5. yellow detection (only on non-white) --------------------
    is_yellow = np.zeros_like(is_white)
    need_hsv  = ~is_white
    if need_hsv.any():
        bgr = np.stack([b[need_hsv], g[need_hsv], r[need_hsv]], axis=1)
        hsv = cv2.cvtColor(bgr.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
        h, s, v = hsv[:,0], hsv[:,1], hsv[:,2]
        is_yellow_sub = (
            (h >= Y_H_MIN) & (h <= Y_H_MAX) &
            (s >= Y_S_MIN) &
            (v >= Y_V_MIN)
        )
        is_yellow[need_hsv] = is_yellow_sub

    # ---- 6. not-black detection (convert all to HSV) ----------------
    # black HSV range: H:0-180 (all), S:0-70, V:0-70
    bgr_all = np.stack([b, g, r], axis=1)
    hsv_all = cv2.cvtColor(bgr_all.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
    s_all, v_all = hsv_all[:,1], hsv_all[:,2]
    is_black = (s_all <= 70) & (v_all <= 70)
    not_black_mask = ~is_black

    # ---- 7. build output arrays --------------------------------------
    white_xyz  = np.column_stack((xs[is_white],  ys[is_white],  zs[is_white])).astype(np.float32)
    yellow_xyz = np.column_stack((xs[is_yellow], ys[is_yellow], zs[is_yellow])).astype(np.float32)
    not_black_xyz = np.column_stack((xs[not_black_mask], ys[not_black_mask], zs[not_black_mask])).astype(np.float32)

    # ensure contiguous for downstream use
    return (
        np.ascontiguousarray(white_xyz),
        np.ascontiguousarray(yellow_xyz),
        np.ascontiguousarray(not_black_xyz),
    )

class PointCloudDownscaler(Node):
	def __init__(self):
		super().__init__("pointcloud_downscaler")
		self.create_subscription(PointCloud2, "/camera/points", self.pointcloud_cb, 10)
		self.white_publisher = self.create_publisher(PointCloud2, "/igvc/white_points", 10)
		self.yellow_publisher = self.create_publisher(PointCloud2, "/igvc/yellow_points", 10)
		self.notblack_publisher = self.create_publisher(PointCloud2, "/igvc/notblack_points", 10)
		self.timer = self.create_timer(0.05, self.publish_my_processed_points)

		self.clouds = None



	def pointcloud_cb(self, msg: PointCloud2):
		# now = self.get_clock().now()      # ROS Time, not perf_counter
		# if self.last_cb_time:
		# 	dt = (now - self.last_cb_time).nanoseconds * 1e-9
		# 	self.get_logger().info(f"Δt={dt:.3f}s  ≈ {1/dt:.2f} Hz")
		# self.last_cb_time = now

		header = Header()
		header.frame_id = msg.header.frame_id
		header.stamp = msg.header.stamp

		white_xyz, yellow_xyz, not_black_xyz = fast_xyz_white_yellow(msg)
		self.clouds = pc2.create_cloud_xyz32(msg.header, white_xyz), pc2.create_cloud_xyz32(msg.header, yellow_xyz), pc2.create_cloud_xyz32(msg.header, not_black_xyz)

	def publish_my_processed_points(self):
		if self.clouds:
			self.white_publisher.publish(self.clouds[0])
			self.yellow_publisher.publish(self.clouds[1])
			self.notblack_publisher.publish(self.clouds[2])
		

def main(args=None):
	rclpy.init(args=args)
	node = PointCloudDownscaler()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info("Shutting down…")
	finally:
		
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
