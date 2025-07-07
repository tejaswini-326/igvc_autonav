from math import pi
import math
import rclpy
from sensor_msgs.msg import PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from math import degrees
import struct
import cv2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import time

WHITE_THRESH   = 90
BALANCE_THRESH = 50
Y_H_MIN, Y_H_MAX = 15, 40
Y_S_MIN = 80
Y_V_MIN = 120

# You might want to change the z thresholds later to be exact

GRID_RES   = 0.1             # 0.1 metres per pixel
GRID_SIZE  = 200             # Number of pixels
CX = CY    = GRID_SIZE // 2  # robot centred

def normalise_angle(angle: float) -> float:
	"""Wrap `angle` to the interval [-π, π].""" 
	return (angle + pi) % (2 * pi) - pi

def get_xy_of_all_white_and_yellow_points_from_pointcloud_msg(msg):
    """
    Return all (x, y) pairs that lie below z = 0 m and whose RGB encodes
    either a white-paint pixel or a yellow-paint pixel.

    * Identical thresholds and logic as the original version
    * Uses a single vectorised BGR→HSV conversion for speed
    """
    height, width = msg.height, msg.width
    white_img = np.zeros((height, width, 3), dtype=np.uint8)   # debug only

    # Temporary storage ----------------------------------------------------
    rows, cols      = [], []
    xs, ys          = [], []
    rgb_ints_uint32 = []

    # ─────────────────── first pass — quick rejects, collect data ─────────
    index = 0
    for x, y, z, rgb in pc2.read_points(
            msg, field_names=("x", "y", "z", "rgb"),
            skip_nans=False):

        row, col = divmod(index, width)
        index += 1

        # Reject invalid / above-ground early (no colour work yet)
        if (rgb is None or not (-1.3 > z > -2.0) or not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z)):
            continue

        try:
            rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
        except struct.error:
            continue

        rows.append(row)
        cols.append(col)
        xs.append(x)
        ys.append(y)
        rgb_ints_uint32.append(rgb_int)

    if not rgb_ints_uint32:               # nothing survived the quick reject
        return np.empty((0, 2), dtype=np.float32)

    # ─────────────────── vectorised colour checks ─────────────────────────
    rgb_arr = np.asarray(rgb_ints_uint32, dtype=np.uint32)
    b = (rgb_arr & 0xFF).astype(np.uint8)
    g = ((rgb_arr >> 8) & 0xFF).astype(np.uint8)
    r = ((rgb_arr >> 16) & 0xFF).astype(np.uint8)

    bgr = np.stack([b, g, r], axis=1).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    # --- white test (RGB) -------------------------------------------------
    avg_rgb = (r.astype(np.int16) + g + b) / 3
    is_white = (
        (r > WHITE_THRESH) &
        (g > WHITE_THRESH) &
        (b > WHITE_THRESH) &
        (np.abs(r - avg_rgb) < BALANCE_THRESH) &
        (np.abs(g - avg_rgb) < BALANCE_THRESH) &
        (np.abs(b - avg_rgb) < BALANCE_THRESH)
    )

    # --- yellow test (HSV) ------------------------------------------------
    is_yellow = (
        (h >= Y_H_MIN) & (h <= Y_H_MAX) &
        (s >= Y_S_MIN) &
        (v >= Y_V_MIN)
    )

    keep_mask = is_white | is_yellow
    if not keep_mask.any():
        return np.empty((0, 2), dtype=np.float32)

    # ─────────────────── build outputs ────────────────────────────────────
    rows_np = np.asarray(rows, dtype=np.int32)[keep_mask]
    cols_np = np.asarray(cols, dtype=np.int32)[keep_mask]
    white_img[rows_np, cols_np] = (255, 255, 255)     # optional debug bitmap

    pts_xy = np.column_stack((np.asarray(xs, dtype=np.float32)[keep_mask],
                              np.asarray(ys, dtype=np.float32)[keep_mask]))
    return pts_xy


def fast_xy_white_yellow(msg):
    """
    Fast white-/yellow-point extractor for arbitrary PointCloud2 layouts.
    Returns (N,2) float32 array of (x,y).
    """
    n_pts   = msg.width * msg.height
    step    = msg.point_step          # bytes per point
    endian  = '>' if msg.is_bigendian else '<'   # ROS flag

    # --- build a float32 view of the buffer --------------------------------
    # One row = step//4 float32s (because 4 bytes == 1 float32)
    row_floats = step // 4
    cloud_f32  = np.frombuffer(msg.data, dtype=endian + 'f4')
    cloud_f32  = cloud_f32.reshape(n_pts, row_floats)

    # Map each field name to its float-index
    fld_idx = {f.name: f.offset // 4 for f in msg.fields}

    xs   = cloud_f32[:, fld_idx['x']]
    ys   = cloud_f32[:, fld_idx['y']]
    z    = cloud_f32[:, fld_idx['z']]
    rgbf = cloud_f32[:, fld_idx['rgb']]

    # --- spatial filter ----------------------------------------------------
    good = np.isfinite(z) & (z > -2.0) & (z < -1.3)
    if not good.any():
        return np.empty((0, 2), dtype=np.float32)

    xs, ys, rgbf = xs[good], ys[good], rgbf[good]

    # --- colour decode -----------------------------------------------------
    rgb_u32 = rgbf.view(np.uint32)
    b = (rgb_u32 & 0xFF).astype(np.uint8)
    g = ((rgb_u32 >> 8)  & 0xFF).astype(np.uint8)
    r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8)

    # white test (cheap, RGB only)
    avg = (r.astype(np.int16) + g + b) // 3
    is_white = (
        (r > WHITE_THRESH) &
        (g > WHITE_THRESH) &
        (b > WHITE_THRESH) &
        (np.abs(r - avg) < BALANCE_THRESH) &
        (np.abs(g - avg) < BALANCE_THRESH) &
        (np.abs(b - avg) < BALANCE_THRESH)
    )

    # yellow test (HSV, but only on non-white candidates)
    need_hsv = ~is_white
    is_yellow = np.zeros_like(is_white)
    if need_hsv.any():
        bgr = np.stack([b[need_hsv], g[need_hsv], r[need_hsv]],
                       axis=1).reshape(-1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        is_yellow_sub = (
            (h >= Y_H_MIN) & (h <= Y_H_MAX) &
            (s >= Y_S_MIN) &
            (v >= Y_V_MIN)
        )
        is_yellow[need_hsv] = is_yellow_sub

    keep = is_white | is_yellow
    if not keep.any():
        return np.empty((0, 2), dtype=np.float32)

    return np.column_stack((xs[keep], ys[keep])).astype(np.float32)


MAX_MAP_XY = (GRID_SIZE // 2 - 1) * GRID_RES   # 9.9 m for GRID_SIZE=200

def clean_xy(pts_xy: np.ndarray) -> np.ndarray:
    """Drop NaN/Inf and anything outside the map box."""
    pts_xy = np.asarray(pts_xy, dtype=np.float32)

    # A. keep only finite rows
    finite_mask = np.isfinite(pts_xy).all(axis=1)
    pts_xy = pts_xy[finite_mask]

    # B. clip to map extent (optional but saves work)
    in_map = (np.abs(pts_xy[:, 0]) <= MAX_MAP_XY) & \
             (np.abs(pts_xy[:, 1]) <= MAX_MAP_XY)
    return pts_xy[in_map]


def voxel_downsample_xy(pts_xy: np.ndarray,
						voxel: float = 0.05,
						max_points: int | None = None) -> np.ndarray:
	"""
	Keep at most one point per `voxel`-metre square cell in the XY plane.

	• pts_xy      –  (N,2) float32 or float64 array
	• voxel       –  edge length of the square voxel in metres  
					 (try 2–3× GRID_RES; tune to taste)
	• max_points  –  optional hard cap; a final uniform random draw
					 enforces this number if the voxel filter still leaves
					 too many points.
	"""
	# Quantise XY to integer voxel indices
	keys = np.floor_divide(pts_xy, voxel).astype(np.int32)

	# Unique rows → indices of the *first* occurrence of every voxel
	_, unique_idx = np.unique(keys, axis=0, return_index=True)

	pts_xy = pts_xy[unique_idx]

	if max_points is not None and pts_xy.shape[0] > max_points:
		idx = np.random.choice(pts_xy.shape[0], max_points, replace=False)
		pts_xy = pts_xy[idx]

	return pts_xy



def radial_scans(pts_xy, mode, yaw, turn_start_yaw, angle_tolerance, debug_stuff):
	'''
	ptx_xy - Output from previous funct
	mode - must be left, straight or horizontal line detect
	turn_start_yaw - as the name suggests
	angle_tolerance - from our wanted angle, what is the max delta in angle that we are willing to head in
	debug_stuff - a tuple of relevant topics etc needed for debug 
	'''
	# ----------------------------------------------------------------------
	# 2) Build a bird-eye binary image of white-floor points
	# ----------------------------------------------------------------------
	
	# We will create a slightly blurry bird's eye floor plan
	# As we shoot out our polar scans, if we hit a white pixel, we want to stop.
	# Because point cloud data has holes in it, we have intentionally set a poor resolution
	# So that every 'grid' in our representation contains at least some point of the pointcloud to create a continuous white stretch where the lanes are there
	# On top of this, we'll do cv2 morpology and dilation to ensure there are no holes
	# If you turn on DEBUG, you can see the grid made in the topic 'intersection_llane_scan_2d_debug'

	# ------------------------------------------------------------------
	# 0) Clean up raw XY points  (drop NaN / Inf / off-map outliers)
	# ------------------------------------------------------------------
	pts_xy = clean_xy(pts_xy)

	# --- SPEED HACK:  down-sample in XY before anything expensive -----------
	pts_xy = voxel_downsample_xy(
				 np.asarray(pts_xy, dtype=np.float32),
				 voxel       = 0.2,
				 max_points  = 5000      # or whatever feels safe
			 )

	binary = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

	for x, y in pts_xy:
		# shift robot to centre → map x (fwd), -y (left) to image coords
		u = int(CX +  x / GRID_RES)
		v = int(CY + -y / GRID_RES)
		if 0 <= u < GRID_SIZE and 0 <= v < GRID_SIZE:
			binary[v, u] = 255

	# Thicken the blobs to ensure no spaces are missed
	kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
	binary  = cv2.dilate(binary, None, iterations=1)

	# ----------------------------------------------------------------------
	# 3) Polar-clearance scan (“laser fan”) instead of Hough 
	#    Shoots virtual rays ahead and chooses the clearest heading
	# ----------------------------------------------------------------------

	# Figure out the angle we 'want to go to', this is just 90 degrees left from the original starting point
	# and as we turn, we update it according to the odometer readings
	if mode == 'horizontal line detect':
		angles = np.deg2rad(np.arange(-30, 30, 1))
	else:
		if mode == 'left':
			wanted =  pi/2 - normalise_angle(yaw - turn_start_yaw)
		elif mode == 'straight':
			wanted = normalise_angle(turn_start_yaw - yaw)
		angles = np.linspace(wanted-angle_tolerance, wanted+angle_tolerance, 2*int(degrees(angle_tolerance))+1)
	
	dirs = np.stack((np.cos(angles), np.sin(angles)), axis=1)

	MAX_RANGE_M  = 100000.0
	MAX_RANGE_PX = int(MAX_RANGE_M / GRID_RES)

	# initialise all rays as “clear to max range”
	dists     = np.full(len(angles), MAX_RANGE_PX, dtype=np.int32)
	never_hit = np.full(len(angles), True, dtype=bool)

	# March each ray
	for k, (ux, uy) in enumerate(dirs):
		for s in range(0, MAX_RANGE_PX + 1):
			u = int(CX + ux * s)
			v = int(CY - uy * s)                      # minus because image-y down
			if not (0 <= u < GRID_SIZE and 0 <= v < GRID_SIZE):
				# ran off map ⇒ keep MAX_RANGE_PX, no hit
				break
			if binary[v, u]:                          # hit white stripe
				dists[k] = s
				never_hit[k] = False
				break

	if mode == 'horizontal line detect':
		return dists	

	def pick_median_among_never_hit():
		nh_idx = np.flatnonzero(never_hit)
		best_idx = int(nh_idx[len(nh_idx) // 2])
		best_dist_px = MAX_RANGE_PX
		return best_idx, best_dist_px
	
	def pick_longest_ray():
		best_idx     = dists.argmax()
		best_dist_px = dists[best_idx]
		return best_idx, best_dist_px

	if mode in ('left', 'straight'):
		if never_hit.any():
			best_idx, best_dist_px = pick_median_among_never_hit()
		else:
			best_idx, best_dist_px = pick_longest_ray()

	best_theta  = angles[best_idx]

	# ----------------------------------------------------------------------
	# ----------------------------------------------------------------------
	# DEBUG STUF ONLY 
	# ----------------------------------------------------------------------
	# ----------------------------------------------------------------------


	if debug_stuff is None:
		return best_theta
	else:
		msg_header, marker_publisher, lane_scan_2d_debug_publisher, intersection_filtered_points_publisher, cv2_bridge, self = debug_stuff


	# Publish the white/yellow points as a blue point cloud
	if len(pts_xy) > 0:
		BLUE_RGB_INT   = 0x0000FF
		BLUE_RGB_FLOAT = struct.unpack('f',
									struct.pack('I', BLUE_RGB_INT))[0]

		cloud_pts = [[x, y, 0.0, BLUE_RGB_FLOAT]
					for (x, y) in pts_xy]

		fields = [PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
				PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
				PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
				PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]

		cloud_msg = pc2.create_cloud(msg_header, fields, cloud_pts)
		intersection_filtered_points_publisher.publish(cloud_msg)

	# ----------------------------------------------------------------------
	# 4) Publish all candidate rays in blue + highlight the chosen best one in green
	# ----------------------------------------------------------------------

	clear_mkr            = Marker()
	clear_mkr.header     = msg_header
	clear_mkr.ns         = "lane_all"
	clear_mkr.id         = 0
	clear_mkr.action     = Marker.DELETEALL
	marker_publisher.publish(clear_mkr)


	def ray_end(theta, dist_px):
		"""helper: pixel distance → (x,y) metres in robot frame"""
		return (float(dist_px * GRID_RES * np.cos(theta)),
				float(dist_px * GRID_RES * np.sin(theta)))

	# 1.  Publish every on-lane ray (except best) as thin blue
	for k, (theta, dist_px) in enumerate(zip(angles, dists)):
		# ----- publish *all* rays, even those with dist_px == -1 -------------
		colour_blue  = (0.0, 0.0, 1.0, 0.4)      # default: translucent blue
		if k == best_idx:
			continue                             # best ray handled later
		if dist_px < 0:                          # invalid / out of band
			colour_blue = (1.0, 0.0, 0.0, 0.3)   # show invalids in faint red
			dist_px     = MAX_RANGE_PX           # draw full-length for clarity

		mkr              = Marker()
		mkr.header       = msg_header
		mkr.ns           = "lane_all"
		mkr.id           = k + 1
		mkr.type         = Marker.LINE_STRIP
		mkr.action       = Marker.ADD
		mkr.scale.x      = 0.04
		mkr.color.r, mkr.color.g, mkr.color.b, mkr.color.a = colour_blue
		x_end, y_end     = ray_end(theta, dist_px)
		mkr.points       = [Point(x=0.0, y=0.0, z=0.0),
							Point(x=x_end, y=y_end, z=0.0)]
		mkr.lifetime     = rclpy.duration.Duration(seconds=0.2).to_msg()
		marker_publisher.publish(mkr)

	# 2.  Publish the *best* ray in thick green
	best_mkr            = Marker()
	best_mkr.header     = msg_header
	best_mkr.ns         = "lane_best"
	best_mkr.id         = 0
	best_mkr.type       = Marker.LINE_STRIP
	best_mkr.action     = Marker.ADD
	best_mkr.scale.x    = 0.05
	best_mkr.color.r    = 0.0
	best_mkr.color.g    = 1.0                    # green highlight
	best_mkr.color.b    = 0.0
	best_mkr.color.a    = 1.0
	x_best, y_best      = ray_end(best_theta, best_dist_px)
	best_mkr.points     = [Point(x=0.0, y=0.0, z=0.0),
						Point(x=x_best, y=y_best, z=0.0)]
	best_mkr.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
	marker_publisher.publish(best_mkr)


	# ----------------------------------------------------------------------
	# 5) Visual-debug image on the 2-D grid
	# ----------------------------------------------------------------------

	# A. build a white-lane bitmap → 3-ch image
	viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

	# helper (same pixel maths you used earlier)
	def grid_uv(theta, dist_px):
		u = int(CX + np.cos(theta) * dist_px)
		v = int(CY - np.sin(theta) * dist_px)   # image-y is down
		return u, v

	# 1) draw every on-lane ray very faintly
	for k, (theta, dist_px) in enumerate(zip(angles, dists)):
		if dist_px < 0:          # skip invalid / out-of-band rays
			continue
		u, v = grid_uv(theta, dist_px)		

	colour = (128, 128, 255)
	thickness = 1
	for k, (theta, dist_px) in enumerate(zip(angles, dists)):
		if dist_px < 0:                 # invalid/out-of-band ray
			continue

		# If this ray never hit anything, visualise it at full length
		if never_hit[k]:
			dist_px = MAX_RANGE_PX

		u_end, v_end = grid_uv(theta, dist_px)
		cv2.line(
			viz, (CX, CY), (u_end, v_end),
			colour, thickness, cv2.LINE_AA
		)

	# 2) draw the *best* ray nice and bold
	u_best, v_best = grid_uv(best_theta, best_dist_px)
	cv2.arrowedLine(
		viz, (CX, CY), (u_best, v_best),
		(0, 255, 0),              # bright green
		2, tipLength=0.08, line_type=cv2.LINE_AA
	)

	viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
	

	img_msg = cv2_bridge.cv2_to_imgmsg(viz, encoding="bgr8")
	img_msg.header = msg_header            # time-sync with point cloud
	lane_scan_2d_debug_publisher.publish(img_msg)

	return best_theta