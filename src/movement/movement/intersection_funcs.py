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

# You might want to change the z thresholds later to be exact

GRID_RES   = 0.1             # 0.1 metres per pixel
GRID_SIZE  = 200             # Number of pixels
CX = CY    = GRID_SIZE // 2  # robot centred


def normalise_angle(angle: float) -> float:
	"""Wrap `angle` to the interval [-π, π].""" 
	return (angle + pi) % (2 * pi) - pi



def get_xy_of_all_white_and_yellow_points_from_pointcloud_msg(msg):
	height, width = msg.height, msg.width
	white_img        = np.zeros((height, width, 3), dtype=np.uint8)
	pts_xy     = []

	index = 0
	for point in pc2.read_points(
			msg,
			field_names=("x", "y", "z", "rgb"),
			skip_nans=False):
		x, y, z, rgb = point
		row = index // width
		col = index % width
		index += 1

		# Skip invalid points
		if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
			continue
		try:
			rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
		except struct.error:
			continue

		# Unpack BGR as before
		r = (rgb_int >> 16) & 0xFF
		g = (rgb_int >> 8)  & 0xFF
		b =  rgb_int        & 0xFF

		# -----------------------------
		# 1) White detection (RGB)
		# -----------------------------
		WHITE_THRESH   = 90
		BALANCE_THRESH = 50
		avg_color = (r + g + b) / 3
		is_white = (
			r > WHITE_THRESH and
			g > WHITE_THRESH and
			b > WHITE_THRESH and
			abs(r - avg_color) < BALANCE_THRESH and
			abs(g - avg_color) < BALANCE_THRESH and
			abs(b - avg_color) < BALANCE_THRESH
		)

		# -----------------------------
		# 2) Yellow detection (HSV)
		# -----------------------------
		# Convert pixel to HSV once
		hsv = cv2.cvtColor(
			np.uint8([[[b, g, r]]]), 
			cv2.COLOR_BGR2HSV
		)[0][0]
		h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

		# generous yellow window
		Y_H_MIN, Y_H_MAX = 15, 40
		Y_S_MIN         = 80
		Y_V_MIN         = 120

		is_yellow = (
			Y_H_MIN <= h <= Y_H_MAX and
			s >= Y_S_MIN and
			v >= Y_V_MIN
		)

		# -----------------------------
		# 3) Ground-level filter
		# -----------------------------
		if z < 0:
			if is_white or is_yellow:
				white_img[row, col] = (255, 255, 255)
				pts_xy.append([x, y])

	return np.asarray(pts_xy)



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
		msg_header, marker_publisher, lane_scan_2d_debug_publisher, intersection_filtered_points_publisher, cv2_bridge = debug_stuff


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