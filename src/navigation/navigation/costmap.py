#!/usr/bin/env python3
"""
Fast cost-map node **with verbose logging**.

Implements:
  •  NumPy point-cloud transform (no do_transform_cloud)
  •  Zero-copy PointCloud2 → NumPy
  •  Re-uses scratch buffers
  •  MultiThreadedExecutor
  •  Plenty of DEBUG/INFO logs so we can inspect every step
"""

import os
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import math
from collections import deque

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
import tf2_geometry_msgs
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf_transformations

VERBOSE_UNECESSARY_THINGS = False

FANCY_HISTORY_COSTMAP = False

OBJECT_HOLD_SEC = 1/15 # Just slightly more than the 20 Hz we receive objects at


def transform_to_matrix(tf_msg) -> np.ndarray:
	"""geometry_msgs/TransformStamped → 4×4 float32 homogeneous matrix"""
	q = tf_msg.transform.rotation
	t = tf_msg.transform.translation
	qx, qy, qz, qw = q.x, q.y, q.z, q.w
	R = np.array([
		[1 - 2 * (qy*qy + qz*qz),     2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
		[2 * (qx*qy + qz*qw),         1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
		[2 * (qx*qz - qy*qw),         2 * (qy*qz + qx*qw),     1 - 2 * (qx*qx + qy*qy)]
	], dtype=np.float32)
	M = np.eye(4, dtype=np.float32)
	M[:3, :3] = R
	M[:3, 3]  = np.array([t.x, t.y, t.z], dtype=np.float32)
	return M


def pc2_numpy_xyz(msg: PointCloud2) -> np.ndarray:
	"""
	Fast zero-copy extraction of an (N,3) float32 array (x,y,z) from a
	sensor_msgs/PointCloud2, regardless of point_step.
	"""
	# dtype whose itemsize == point_step so NumPy respects the stride
	dtype_xyz = np.dtype({'names': ('x', 'y', 'z'),
						  'formats': ('<f4', '<f4', '<f4'),
						  'itemsize': msg.point_step})
	cloud = np.frombuffer(msg.data, dtype=dtype_xyz,
						  count=msg.width * msg.height)
	return np.vstack((cloud['x'], cloud['y'], cloud['z'])).T.astype(np.float32)


class CostmapNode(Node):
	def __init__(self):
		super().__init__('costmap_node_fast_debug')

		# ----------------------- map parameters ---------------------------
		self.resolution = 0.067   # metres / cell
		self.width      = 300
		self.height     = 300
		self.frame_id   = 'odom'

		# --------------------- pre-allocated buffers ----------------------
		self._empty   = np.zeros((self.height, self.width),  np.uint8)
		self._scratch = np.zeros_like(self._empty)           # reusable work buf
		self.white_map  = self._empty.copy()
		self.yellow_map = self._empty.copy()
		self.object_map = self._empty.copy()
		self.imu_yaw = None
		self.yaw_buffer = deque(maxlen=6)
		self.last_object_msg_time = None

		# --------------------------- I/O ----------------------------------
		qos = 20
		self.create_subscription(PointCloud2, '/object_pc',self._object_cb, qos)
		self.create_subscription(PointCloud2, '/white_lane_points',self._white_cb,  qos)
		self.create_subscription(MarkerArray, '/lane_fitted_yellow',self._yellow_cb, qos)
		self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap', qos)
		self.create_subscription(Odometry, '/odom', self.odom_callback, qos)

		# ----------------------------- TF ---------------------------------
		self.tf_buffer   = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# --------------------------- state --------------------------------
		self._object_pc = self._white_pc = self._yellow_pc = None
		self._new_object = self._new_white = self._new_yellow = False
		self._T_odom_cam = np.eye(4, dtype=np.float32)
		self.origin_x = self.origin_y = 0.0
		self.pose = None

		if FANCY_HISTORY_COSTMAP: 
			# ------------ GLOBAL “memory” cost‑map (fixed origin) ------------
			self.g_size       = 2000              # 2000×2000 cells ≈134 m × 134 m
			self.g_map        = np.zeros((self.g_size, self.g_size), np.uint8)
			self.g_origin_set = False             # we’ll set it after first TF
			self.g_origin_x   = self.g_origin_y = 0.0

		# ---------------------------- timer -------------------------------
		self.create_timer(0.05, self._timer_cb)   # 10 Hz


		# --------------------- logger level tweak -------------------------
		# Set ROS_CONSOLE_STDOUT_LINE_BUFFERED=1 for live prints in docker
		# level = os.getenv('COSTMAP_DEBUG_LEVEL', 'info').lower()
		# self.get_logger().info(f'Cost-map node up (log level = {level})')
		# self.get_logger().set_level(
		#     rclpy.logging.LoggingSeverity.DEBUG
		#     if level == 'debug' else rclpy.logging.LoggingSeverity.INFO)

	# ---------------------- subscriber callbacks --------------------------
	def _object_cb(self, msg):
		self._object_pc, self._new_object = msg, True
		self._object_last_update = self.get_clock().now()
	def _white_cb(self,  msg):  self._white_pc,  self._new_white  = msg, True
	def _yellow_cb(self, msg: MarkerArray):
		points = []
		for marker in msg.markers:
			for p in marker.points:
				points.append([p.x, p.y, p.z])
		if points:
			self._yellow_pc = np.array(points, dtype=np.float32)
			self._new_yellow = True
		else:
			self._yellow_pc = None
			self._new_yellow = False


	def odom_callback(self, msg):
		self.pose = msg.pose.pose
		q = msg.pose.pose.orientation
		_, _, yaw = tf_transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
		self.imu_yaw = yaw
		self.yaw_buffer.append(yaw)

		if len(self.yaw_buffer) > 2:
			yaw_array = np.unwrap(np.array(self.yaw_buffer))

			median = np.median(yaw_array)
			deviation = np.abs(yaw_array - median)
			std_dev = np.std(yaw_array)

			if std_dev > 0.05:
				filtered_yaws = yaw_array[deviation < 1.5 * std_dev]
			else:
				filtered_yaws = yaw_array 

			if len(filtered_yaws) > 0:
				self.imu_yaw = float(np.mean(filtered_yaws))
			else:
				self.imu_yaw = float(median)
		else:
			self.imu_yaw = yaw



	def odom_to_costmap(self, x: float, y: float) -> tuple[int, int] | None:
		mx = int((x - self.origin_x) / self.resolution)
		my = int((y - self.origin_y) / self.resolution)

		if 0 <= mx < self.width and 0 <= my < self.height:
			return (mx, my)
		else:
			return None


	def draw_v_lines(self):
		x = self.pose.position.x
		y = self.pose.position.y
		costmap_coords = self.odom_to_costmap(x, y)
		if not costmap_coords:
			return np.zeros_like(self.white_map, dtype=np.uint8)

		xm, ym = costmap_coords

		# Offset 10 pixels in yaw direction
		dx = int(-15 * np.cos(self.imu_yaw))
		dy = int(np.sin(self.imu_yaw))
		center_x = xm + dx
		center_y = ym - dy  # invert dy due to image coords
		center = (center_x, center_y)

		# Define arms of the V
		spread_deg = 50
		left_angle = self.imu_yaw + np.radians(spread_deg)
		right_angle = self.imu_yaw - np.radians(spread_deg)
		line_length = 400

		def endpoint(angle):
			dx = int(np.cos(angle) * line_length)
			dy = int(np.sin(angle) * line_length)
			return (center_x + dx, center_y + dy)

		pt1 = endpoint(left_angle)
		pt2 = endpoint(right_angle)

		v_line_layer = np.zeros_like(self.white_map, dtype=np.uint8)
		cv2.line(v_line_layer, center, pt1, color=250, thickness=3)
		cv2.line(v_line_layer, center, pt2, color=250, thickness=3)

		return v_line_layer

	# ---------------------------- timer loop -----------------------------
	def _timer_cb(self):
		# ---------- TF lookup --------------------------------------------
		try:
			tf = self.tf_buffer.lookup_transform('odom', 'camera_link',
												 rclpy.time.Time())
			self._T_odom_cam = transform_to_matrix(tf)
			rx, ry = tf.transform.translation.x, tf.transform.translation.y
			self.origin_x = rx - (self.width  * self.resolution) / 2.0
			self.origin_y = ry - (self.height * self.resolution) / 2.0
		except Exception as e:
			self.get_logger().warn(f'TF lookup failed: {e}')
			return

		# self.get_logger().debug(
		#     f"TF OK  | origin=({self.origin_x:.2f},{self.origin_y:.2f})")

		# ---------- regenerate each layer --------------------------------
		if self._new_white and self._white_pc is not None:
			self.white_map[:] = self._make_layer(self._white_pc, 250, 'white')
			self._new_white = False
		if self._new_yellow and self._yellow_pc is not None:
			self.yellow_map[:] = self._make_layer_numpy(self._yellow_pc, 200, 'yellow')
			self._new_yellow = False

		now = self.get_clock().now()

		if self._object_pc is not None:
			age = (now - self._object_last_update).nanoseconds * 1e-9 if self._object_last_update else float('inf')
			if self._new_object:
				self.object_map[:] = self._make_layer(self._object_pc, 245, 'object')
				self._new_object = False
			elif age > OBJECT_HOLD_SEC:
				self.object_map.fill(0)
				self._object_pc = None
				self._object_last_update = None
		else:
			self.object_map.fill(0)

		# rear_mask_layer = np.zeros_like(self.white_map, dtype=np.uint8)
		# rear_mask_layer[:, :math.ceil(1.15*(self.width // 2))] = 250
		# v_layer = self.draw_v_lines()
		# ---------- fuse + distance penalty + publish ----------------------
		combined_raw = np.maximum.reduce([self.white_map,
									self.yellow_map,
									self.object_map]) # v_layer
		
		# ▸ Fancy “remember everything” mode
		if FANCY_HISTORY_COSTMAP:
			if not self.g_origin_set:
				# centre global map on the robot’s *starting* odom pose
				self.g_origin_x = tf.transform.translation.x - \
								(self.g_size * self.resolution) / 2.0
				self.g_origin_y = tf.transform.translation.y - \
								(self.g_size * self.resolution) / 2.0
				self.g_origin_set = True
				self.get_logger().info(
					f"Global map anchored at ({self.g_origin_x:.2f},"
					f"{self.g_origin_y:.2f}) in odom")
						# offset (cells) of this window within the global map
			gi0 = int((self.origin_x - self.g_origin_x) / self.resolution)
			gj0 = int((self.origin_y - self.g_origin_y) / self.resolution)

			# bounds‑checked slice targets
			i1 = gi0 + self.width
			j1 = gj0 + self.height
			si0, sj0 = max(0, gi0), max(0, gj0)
			si1 = min(self.g_size, i1)
			sj1 = min(self.g_size, j1)

			li0 = si0 - gi0            # how much of the local grid is clipped
			lj0 = sj0 - gj0
			li1 = self.width  - (i1 - si1)
			lj1 = self.height - (j1 - sj1)

			if li1 > li0 and lj1 > lj0:  # anything left after clipping?
				decay = 0.5                                # value to subtract every tick
				self.g_map = np.clip(self.g_map.astype(np.int16) - decay, 0, 255).astype(np.uint8)
				np.maximum(self.g_map[sj0:sj1, si0:si1],combined_raw[lj0:lj1, li0:li1],out=self.g_map[sj0:sj1, si0:si1])

			# --------- crop robot‑centred window back out --------------
			combined_raw = self.g_map[sj0:sj1, si0:si1].copy()
	

		penalty  = self._distance_penalty(combined_raw, thresh=200,radius_m=1.5, steepness=1.0)

		final    = np.maximum(combined_raw, penalty)      # 0-100 uint8
		self._publish_costmap(final, self.get_clock().now().to_msg())


	# ------------------------ layer construction -------------------------
	def _make_layer(self, pc_msg: PointCloud2, value: int, tag: str) -> np.ndarray:
		# 1) cloud → NumPy -------------------------------------------------
		pts = pc2_numpy_xyz(pc_msg)
		# self.get_logger().debug(f"[{tag}] cloud points: {pts.shape[0]}")
		if pts.size == 0:
			return self._empty

		# 2) transform -----------------------------------------------------
		xyz1 = np.hstack((pts, np.ones((pts.shape[0], 1), np.float32)))
		xyz  = (self._T_odom_cam @ xyz1.T).T[:, :3]          # (N,3)

		# 3) map indices ---------------------------------------------------
		mx_raw = (xyz[:, 0] - self.origin_x) / self.resolution
		my_raw = (xyz[:, 1] - self.origin_y) / self.resolution
		valid  = ((mx_raw >= 0) & (mx_raw < self.width) &
				  (my_raw >= 0) & (my_raw < self.height))
				#   (mx_raw > 100))             # ignore rear half
		vcount = int(np.count_nonzero(valid))
		# self.get_logger().debug(f"[{tag}] valid pts: {vcount}")
		if vcount == 0:
			return self._empty

		mx = mx_raw[valid].astype(np.int32)
		my = my_raw[valid].astype(np.int32)

		# 4) draw points ---------------------------------------------------
		layer = self._scratch
		layer.fill(0)
		layer[my, mx] = value

		# 5) gaussian blur -------------------------------------------------
		#blurred = layer
		blurred = cv2.GaussianBlur(layer.astype(np.float32),(5, 5), sigmaX=60.0)
		gmax = float(blurred.max())
		# self.get_logger().debug(f"[{tag}] gmax before scale: {gmax:.1f}")
		if gmax > 0.0:
			power = 2.0 if value == 245 else 0.8
			blurred = ((blurred / (gmax ** power)) * 100).astype(np.uint8)
			np.maximum(layer, blurred, out=layer)

		# copy because scratch will be reused next layer
		return layer.copy()
	
	def _make_layer_numpy(self, pts: np.ndarray, value: int, tag: str) -> np.ndarray:
		if pts is None or pts.size == 0:
			return self._empty

		xyz1 = np.hstack((pts, np.ones((pts.shape[0], 1), np.float32)))
		xyz  = (self._T_odom_cam @ xyz1.T).T[:, :3]

		mx_raw = (xyz[:, 0] - self.origin_x) / self.resolution
		my_raw = (xyz[:, 1] - self.origin_y) / self.resolution
		valid  = ((mx_raw >= 0) & (mx_raw < self.width) &
				(my_raw >= 0) & (my_raw < self.height))
		vcount = int(np.count_nonzero(valid))
		if vcount == 0:
			return self._empty

		mx = mx_raw[valid].astype(np.int32)
		my = my_raw[valid].astype(np.int32)

		layer = self._scratch
		layer.fill(0)
		layer[my, mx] = value

		blurred = cv2.GaussianBlur(layer.astype(np.float32), (5, 5), sigmaX=60.0)
		gmax = float(blurred.max())
		if gmax > 0.0:
			power = 2.0 if value == 245 else 0.8
			blurred = ((blurred / (gmax ** power)) * 100).astype(np.uint8)
			np.maximum(layer, blurred, out=layer)

		return layer.copy()

	def _distance_penalty(self, grid: np.ndarray,
						thresh: int = 200,
						radius_m: float = 1.5,
						steepness: float = 1) -> np.ndarray:
		"""
		Turn a binary 'obstacle mask' (anything ≥ `thresh`) into a smooth
		clearance penalty 0‥100 using an Euclidean distance transform.

		• `radius_m`    – how far (metres) the penalty should reach
		• `steepness`   – >1 ⇒ sharper walls, <1 ⇒ gentler slope
		"""
		# 1) binary obstacle mask  (lane pixels, objects, walls …)
		occ = (grid >= thresh).astype(np.uint8)

		# 2) OpenCV distance-transform on the *inverted* mask
		dist_cells = cv2.distanceTransform(255 - occ * 255,
										cv2.DIST_L2, 5).astype(np.float32)

		# 3) metres → normalised 0-1 “closeness”
		dist_m     = dist_cells * self.resolution
		closeness  = np.clip((radius_m - dist_m) / radius_m, 0.0, 1.0)

		# 4) non-linear mapping and scale to 0-100 int8
		penalty    = (closeness ** steepness) * 200.0
		return penalty.astype(np.uint8)

	# -------------------------- grid publisher ---------------------------
	@staticmethod
	def _scale_to_ogrid(grid: np.ndarray) -> np.ndarray:
		"""
		Convert uint8 [0,255] → int8 [0,100] as required by nav2.
		We use *float* scaling first to avoid the bug that produced only 0.
		"""
		return ((grid.astype(np.float32) / 255.0) * 100.0 + 0.5).astype(np.int8)

	def _publish_costmap(self, grid: np.ndarray, stamp):
		msg = OccupancyGrid()
		msg.header = Header(stamp=stamp, frame_id=self.frame_id)
		msg.info.resolution = self.resolution
		msg.info.width      = self.width
		msg.info.height     = self.height
		msg.info.origin.position.x = float(self.origin_x)
		msg.info.origin.position.y = float(self.origin_y)
		msg.info.origin.orientation.w = 1.0

		scaled = self._scale_to_ogrid(grid)
		nz = int(np.count_nonzero(scaled))
		# self.get_logger().debug(f"publish grid: non-zero cells = {nz}")
		msg.data = scaled.flatten().tolist()
		self.costmap_pub.publish(msg)

def main(args=None):
	rclpy.init(args=args)
	node = CostmapNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
