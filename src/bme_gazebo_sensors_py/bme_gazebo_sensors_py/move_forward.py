import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import math
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import String
from nav_msgs.msg import Odometry

from object_detection.msg import ObjectData
from std_msgs.msg import String

'''obj detection stuff'''
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

# x forward, y left, z upward
LINEAR_SPEED = 1.5
REDUCED_LINEAR_SPEED = 0.5
THRESHOLD_ANGLE_TO_ROTATE = 0.2
THRESHOLD_ANGLE_TO_REDUCE_LINEAR_SPEED = 0.4
MIN_CLUSTERING_DISTANCE = 0.971
MIN_CLUSTERING_POINTS = 20
WHITE_THRESHOLD = 100
COLOR_BALANCE_THRESHOLD = 25
ANGLE_FACTOR = 1.1

class TrackedObstacle:   #constructor to store object data
	def __init__(self, label, odom_position, lane, detection_time):
		self.label = label
		self.odom_position = odom_position 
		self.lane = lane                   
		self.first_detected = detection_time
		self.position_history = []
		self.last_updated = detection_time
		self.is_being_avoided = False
		self.resume_timer_active = False 
	
	def update(self, update_time):
		self.last_updated = update_time
	
	def __repr__(self):
		return (f"<Obstacle: {self.lane} lane, "
				f"pos=({self.odom_position[0]:.2f},{self.odom_position[1]:.2f}), "
				f"avoided={self.is_being_avoided}>")
	
class LaneFollowerNode(Node):
	def __init__(self):
		super().__init__('lane_follower_node')
		self.subscription = self.create_subscription(
			PointCloud2,
			'/camera/points',
			self.pointcloud_callback,
			10
		)
		self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.marker_pub = self.create_publisher(Marker, '/lane_marker', 10)
		self.create_subscription(String, '/intersection', self.intersection_cb, 10)
		self.create_subscription(Odometry, "/odom", self.odom_cb, 50) 

		self.create_subscription(ObjectData, '/object_data', self.object_callback, 10)
		#self.obj_pos_history = {}
		#stuff for transform
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.latest_lane_centers = None 
		self.tracked_obstacles = []  # List of TrackedObstacle objects
		self.current_avoid_target = None  # ID of obstacle being avoided

		self.intersection_pub = self.create_publisher(String, '/intersection', 10)
		self.last_cmd = Twist()
		self.last_cmd.linear.x = LINEAR_SPEED
		self.last_cmd.angular.z = 0.0
		self.active=True
		self.stopping = False
		# Set the variable below to 'left' or 'right' depending on which lane you want the robot to follow
		self.which_lane = 'right'

		

		self.robot_x = 0.0  # initialize
		self.avoid_obstacle = None

	def intersection_cb(self, msg):
		if msg.data.lower() == "None":
			self.active = True
			# self.get_logger().info("🟢 'None' received — move_forward activated.")
		else:
			self.active = False
			
	def odom_cb(self, msg: Odometry):
		self.velocity_squared=(msg.twist.twist.linear.x)**2+(msg.twist.twist.linear.y)**2     
		self.robot_x = msg.pose.pose.position.x #current global posn of bot     

	def publish(self, cmd, target=None):
		if self.stopping:
			# Override any move command with a full stop
			self.get_logger().warn("Robot is stopped — blocking velocity command.")
			stop_cmd = Twist()
			stop_cmd.linear.x = 0.0
			stop_cmd.angular.z = 0.0
			self.cmd_pub.publish(stop_cmd)
			if(self.velocity_squared==0):
				msg = String()
				msg.data = "left"  # or "none" or "done" — your choice
				self.intersection_pub.publish(msg)
				self.stopping=False
		else:
			self.cmd_pub.publish(cmd)

		print('')

	def debug_time_yo_yo_yo(self, x, y, msg, img, centers):
		# self.get_logger().info(f"DEBUG")
		height = msg.height
		width = msg.width

		white_img = img
		index = 0
		for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
			px, py, pz, rgb = point
			if abs(px - x) < 0.02 and abs(py - y) < 0.02:
				row = index // width
				col = index % width
				cv2.circle(white_img, (col, row), 5, (0, 0, 255), -1)
			for label, center in centers:
				cx = center[0]
				cy = center[1]
				# self.get_logger().info(f"{cx}, {cy}")
				if abs(px - cx) < 0.027 and abs(py - cy) < 0.027:
					row = index // width
					col = index % width
					cv2.circle(white_img, (col, row), 5, (255, 0, 0), -1)
					text = f"({cx:.2f}, {cy:.2f})"
					cv2.putText(
						white_img, text, (col + 10, row - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
					)
					break
			index += 1

		cv2.imshow("Target", white_img)
		cv2.waitKey(1) 
		return
		 
	def calculate_normal_velocity(self, target, msg, white_img, centers):
		cmd = Twist()
		self.debug_time_yo_yo_yo(target[0], target[1], msg, white_img, centers)

		# Compute direction to target
		angle_to_target = math.atan2(target[1], target[0])  # direction from (0,0) to target in radians

		# self.get_logger().info(f"{target[0]}")
		# self.get_logger().info(f"{angle_to_target}")

		# Move toward target
		if(target[0] < 2.2 and target[0] > 1.8):
			if (abs(angle_to_target) > 0.2 ):
				cmd.linear.x = (LINEAR_SPEED - abs(angle_to_target) * ANGLE_FACTOR)/3 # Forward speed
				cmd.angular.z = angle_to_target/2
			else:
				cmd.linear.x = LINEAR_SPEED - abs(angle_to_target) * ANGLE_FACTOR # Forward speed
				cmd.angular.z = angle_to_target/5 # teer towards target
		else:
			cmd.linear.x = LINEAR_SPEED - abs(angle_to_target) * ANGLE_FACTOR # Forward speed
			cmd.angular.z = angle_to_target  # teer towards target
		return cmd
	
	def detect_horizontal_lines_2d(self, msg):
		
		white_y_vals = []

		for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
			x, y, z, rgb = point
			if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
				continue

			try:
				rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
			except:
				continue

			r = (rgb_int >> 16) & 0xFF
			g = (rgb_int >> 8) & 0xFF
			b = rgb_int & 0xFF

			white_threshold = WHITE_THRESHOLD  # Increased threshold
			color_balance_threshold = COLOR_BALANCE_THRESHOLD  # Colors should be similar for true white
			
			# Check if pixel is white (high intensity + balanced RGB)
			avg_color = (r + g + b) / 3
			if (r > white_threshold and g > white_threshold and b > white_threshold and
				abs(r - avg_color) < color_balance_threshold and 
				abs(g - avg_color) < color_balance_threshold and 
				abs(b - avg_color) < color_balance_threshold):

				if 3.0 < x < 6.5 and -1.4 < z < -1.3:
					white_y_vals.append(y)

		if len(white_y_vals) < 300:
			return

		# Create histogram over Y axis
		hist, bin_edges = np.histogram(white_y_vals, bins=20, range=(-2.0, 2.0))

		# Find contiguous bin groups with high density
		dense_threshold = 5  # min points per bin
		dense_bins = [bin_edges[i] for i in range(len(hist)) if hist[i] >= dense_threshold]

		# self.get_logger().warn(f"STOP LINE DETECTED: dense y-range = {dense_y_range:.2f}m with {len(white_y_vals)} points")
		# self.get_logger(f"The number of dense bins : {len(dense_bins)}")

		if len(dense_bins) < 2:
			return

		# Dense region span = difference between leftmost and rightmost high bins
		y_dense_min = min(dense_bins)
		y_dense_max = max(dense_bins)
		dense_y_range = y_dense_max - y_dense_min

		self.get_logger().warn(f"dense y-range = {dense_y_range:.2f}m with {len(white_y_vals)} points")

		if dense_y_range > 0.7 and len(white_y_vals) > 600:
			self.get_logger().warn(f"STOP LINE DETECTED: dense y-range = {dense_y_range:.2f}m with {len(white_y_vals)} points")
			self.stopping = True

	def pointcloud_callback(self, msg):
		if self.tracked_obstacles:
			# Sort and trim obstacles
			sorted_obstacles = sorted(self.tracked_obstacles, key=lambda obs: (obs.odom_position[0] -self.robot_x))
			closest = sorted_obstacles[0]
			self.get_logger().info(f"[Obstacle] Closest: x={closest.odom_position[0]:.2f}, label={closest.label}")

			# Always keep latest version of closest obstacle
			self.avoid_obstacle = closest

			if not closest.is_being_avoided:
				#find closest in tracked_obstacled and set being_avoided as True 
				closest.is_being_avoided = True
				if closest.label == 'stop':
					self.get_logger().info("Closest obstacle is STOP → stopping")
					self.handle_stop()
				elif closest.label == 'mannequin':
					self.get_logger().info("Closest obstacle is MANNEQUIN → handling")
					self.handle_mannequin()
				else:
					self.get_logger().info("Entering avoid mode for barrel!")
					self.handle_other_obstacle()
		if self.active is False:
			return
		
		self.detect_horizontal_lines_2d(msg)
		# if self.stopped == True:
		#     self.destroy_node()

		height = msg.height
		width = msg.width

		white_img = np.zeros((height, width, 3), dtype=np.uint8)
		white_ground_points = []

		index = 0
		for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
			x, y, z, rgb = point
			row = index // width
			col = index % width

			# Skip invalid points
			if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
				continue

			try:
				rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
			except:
				continue

			r = (rgb_int >> 16) & 0xFF
			g = (rgb_int >> 8) & 0xFF
			b = rgb_int & 0xFF

			# Improved white detection
			white_threshold = WHITE_THRESHOLD  # Increased threshold
			color_balance_threshold = COLOR_BALANCE_THRESHOLD  # Colors should be similar for true white
			
			# Check if pixel is white (high intensity + balanced RGB)
			avg_color = (r + g + b) / 3
			if (r > white_threshold and g > white_threshold and b > white_threshold and
				abs(r - avg_color) < color_balance_threshold and 
				abs(g - avg_color) < color_balance_threshold and 
				abs(b - avg_color) < color_balance_threshold):

				# Ground level filtering
				if -1.4 < z < -1.3 and 0.0 < x < 4.0:  # Adjusted range
					white_img[row, col] = (255, 255, 255)
					white_ground_points.append([x, y, z])  # Store x,y,z coordinates
			index += 1

		# Only do 3D clustering for lane following if we have enough points
		if len(white_ground_points) < 10:
			self.get_logger().warn("Not enough white points for lane detection")
			self.publish(self.last_cmd)
			return

		points_np = np.array(white_ground_points)
		
		# Use only x,y coordinates for clustering (ignore z)
		points_xy = points_np[:, :2]  # Extract x,y coordinates

		clustering = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy)
		labels = clustering.labels_
		
		# Count clusters and noise points
		unique_labels = set(labels)

		n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
		n_noise = list(labels).count(-1)
		
		# self.get_logger().info(f"Clusters found: {n_clusters}, Noise points: {n_noise}")
		# Process lane clusters
		centers = []
		for label in unique_labels:
			if label == -1:
				continue

			cluster_points = points_xy[labels == label]
			cluster_size = len(cluster_points)
			if cluster_size < 10:
				continue

			center = np.mean(cluster_points, axis=0)
			centers.append((label, center))

		# Lane following logic
		cmd = Twist()
		if len(centers) == 2:
			centers.sort(key=lambda c: c[1][1])  # sort by y coordinate
			# self.get_logger().info(f"{centers}")

			left_lane = centers[1][1]   # leftmost cluster
			right_lane = centers[0][1]  # rightmost cluster
			
			target = (left_lane + right_lane) / 2
			
			cmd = self.calculate_normal_velocity(target, msg, white_img, centers)
			
			self.publish(cmd, target)
			self.last_cmd = cmd
			self.last_cmd.linear.x = 0.0
			return

		if len(centers) >= 3:
			centers.sort(key=lambda c: c[1][1])  # sort by y coordinate

			right_lane = centers[0][1]
			middle_lane = centers[1][1]
			left_lane = centers[2][1]

			target = ((middle_lane + right_lane) / 2) if self.which_lane == 'right' else ((middle_lane + left_lane) / 2)
			cmd = self.calculate_normal_velocity(target, msg, white_img, centers)

			self.latest_lane_centers = {
			"left": np.append(left_lane, np.mean(points_np[:, 2])),
			"right": np.append(right_lane, np.mean(points_np[:, 2])),
			"center": np.append(middle_lane, np.mean(points_np[:, 2]))
			}

			self.publish(cmd, target)
			self.last_cmd = cmd
			self.last_cmd.linear.x = 0.0
			return
		

		elif len(centers) == 1:
			
			# In case only one cluster is detected, this code will fir a parabola to the points in that clsuter and follow the curve.
			# It does this by following the center of the cluster and estimating the slope of the curve at that point. If enough points are not detected, 
			# it will simply move forward without turning.
			
			self.get_logger().info("Only one lane cluster found")
			label, center = centers[0]
			cluster_points = points_xy[labels == label]

			if len(cluster_points) > 7:
				x = cluster_points[:, 0]
				y = cluster_points[:, 1]

				# Fit parabola: y = ax^2 + bx + c
				coeffs = np.polyfit(x, y, deg=2)
				a, b, c = coeffs

				# Estimate local slope at center of lane
				x_center = np.mean(x)
				slope = 2 * a * x_center + b
				angle = math.atan(slope)

				# Control command
				cmd.linear.x = 0.2
				cmd.angular.z = angle

				self.get_logger().info(f"Curve-follow angle: {math.degrees(angle):.2f}°")

				# Visualization marker
				marker = Marker()
				marker.header.frame_id = msg.header.frame_id
				marker.header.stamp = self.get_clock().now().to_msg()
				marker.ns = "fitted_lane"
				marker.id = 0
				marker.type = Marker.LINE_STRIP
				marker.action = Marker.ADD
				marker.scale.x = 0.05  # line width
				marker.color.r = 1.0
				marker.color.g = 0.0
				marker.color.b = 1.0
				marker.color.a = 1.0
				marker.lifetime.sec = 0

				# Plot fitted curve
				x_vals = np.linspace(np.min(x), np.max(x), num=30)
				for x_i in x_vals:
					y_i = a * x_i**2 + b * x_i + c

					# Safely determine z
					if cluster_points.shape[1] >= 3:
						z_i = np.mean(cluster_points[:, 2])
					else:
						z_i = 0.0  # Default height if Z not available

					pt = Point(x=x_i, y=y_i, z=z_i)
					marker.points.append(pt)

				self.marker_pub.publish(marker)

			else:
				self.get_logger().warn("Not enough points to fit curve")
				cmd.linear.x = 0.2
				cmd.angular.z = 0.0

			# Visual debugging and publishing control
			self.debug_time_yo_yo_yo(center[0], center[1], msg, white_img, centers)
			self.publish(cmd)
			self.last_cmd = cmd
			self.last_cmd.linear.x = 0.0

		else:
			self.get_logger().warn("No valid clusters found for lane detection")
			self.publish(self.last_cmd)
		
			
			
	def classify_object_lane(self, odom_pos, current_time):
		if not self.latest_lane_centers:
			self.get_logger().warn("Lane center data unavailable — skipping lane check for object")
			return "right"  # default fallback

		# Transform lane centers to odom
		left_raw = self.latest_lane_centers["left"]
		right_raw = self.latest_lane_centers["right"]

		left = self.transform_to_odom(*left_raw, current_time)
		right = self.transform_to_odom(*right_raw, current_time)

		if left is None or right is None:
			self.get_logger().warn("TF failed — returning default lane")
			return "right"

		left = np.array(left[:2])
		right = np.array(right[:2])
		obj = np.array(odom_pos[:2])

		# Compute center line and perpendicular vector
		lane_vec = right - left
		if np.linalg.norm(lane_vec) < 1e-3:
			self.get_logger().warn("Lane vector too small — probably bad data")
			return "right"

		lane_unit = lane_vec / np.linalg.norm(lane_vec)
		perp_unit = np.array([-lane_unit[1], lane_unit[0]])

		center = (left + right) / 2.0
		obj_vec = obj - center
		lateral_offset = np.dot(obj_vec, perp_unit)

		lane = "left" if lateral_offset > 0 else "right"

		# Log visuals for debugging
		self.get_logger().info(f"Classified {odom_pos} as {lane} lane | offset={lateral_offset:.2f}")
		self.get_logger().info(f"Left: {left_raw} → {left}, Right: {right_raw} → {right}, Obj: {odom_pos}")

		return lane


	def object_callback(self, msg):
		'''takes in object data, tracks obstacles based on distance and time'''
		current_time = self.get_clock().now()
		obj_label = msg.label
		obj_pc = msg.pointcloud
		#filter object base, convert to odom
		obj_base_pos = self.filter_pc(obj_pc)
		if obj_base_pos is None:
			self.get_logger().warn(f"Couldn't compute base position for {obj_label}")
			return
		
		odom_pos = self.transform_to_odom(*obj_base_pos, current_time)
		if odom_pos is None:
			return 

		POSITION_THRESHOLD = 0.5 
		existing_obstacle = None

		for obs in self.tracked_obstacles:
			dist = math.sqrt(
				(obs.odom_position[0] - odom_pos[0])**2 +
				(obs.odom_position[1] - odom_pos[1])**2
			)
			self.get_logger().warn(f"Distance to {obs.label}: {dist:.2f}")
			if dist <= POSITION_THRESHOLD:
				existing_obstacle = obs
				break

		if existing_obstacle:
			existing_obstacle.update(current_time)
			existing_obstacle.odom_position = odom_pos  # update position

			# If mannequin, update its position history
			if existing_obstacle.label == "vest":
				existing_obstacle.position_history.append((current_time, odom_pos))
				if len(existing_obstacle.position_history) > 5:
					existing_obstacle.position_history.pop(0)

			self.get_logger().info(f"Updated obstacle: {existing_obstacle}")

		else:
			# Log position key (rounded)
			position_key = tuple(round(c, 1) for c in odom_pos)
			self.get_logger().info(
				f"Ground position of {obj_label}: x={odom_pos[0]:.2f}, y={odom_pos[1]:.2f}, z={odom_pos[2]:.2f}"
			)

			# Classify lane based on base odom position
			lane = self.classify_object_lane(odom_pos, current_time)
			self.get_logger().info(f"{obj_label} is in the {lane} lane based on dynamic lane detection")

			# Track the new obstacle
			new_obs = TrackedObstacle(
				label=obj_label,
				odom_position=odom_pos,
				lane=lane,
				detection_time=current_time
			)
			if obj_label == "vest":
				new_obs.position_history = [(current_time, odom_pos)]
			self.tracked_obstacles.append(new_obs)
			self.get_logger().info(f"New obstacle: {new_obs}")

		# Step 5: Clean up old or passed obstacles
		self.cleanup_obstacles(current_time)

	def handle_mannequin(self):
		mannequin = next((obs for obs in self.tracked_obstacles if obs.label == "vest"), None)
		if not mannequin or not hasattr(mannequin, "position_history"):
			return
		
		if mannequin.is_being_avoided:
			return  # Already handling this mannequin
		
		history = mannequin.position_history
		if len(history) < 2:
			return
		
		start_pos = np.array(history[0][1])
		latest_pos = np.array(history[-1][1])
		movement = np.linalg.norm(latest_pos[:2] - start_pos[:2])
		
		MOVEMENT_THRESHOLD = 0.5
		
		if movement < MOVEMENT_THRESHOLD:
		# Static mannequin → change lanes
			mannequin_lane = self.classify_lane(latest_pos, self.get_clock().now())
			new_lane = "left" if mannequin_lane == "right" else "right"
			self.which_lane = new_lane
			mannequin.is_being_avoided = True
			self.get_logger().info(f"Static mannequin in {mannequin_lane} lane → switching to {new_lane}")
		else:
		# Moving mannequin → stop temporarily
			if not self.stopping:
				self.stopping = True
				mannequin.is_being_avoided = True
				self.get_logger().warn("Moving mannequin → stopping robot temporarily")
			
				if not mannequin.resume_timer_active:
					mannequin.resume_timer_active = True
			
				def check_and_resume():
					mannequin_still_there = any(
					obs.label == "vest" and obs.is_being_avoided for obs in self.tracked_obstacles
					)
					if not mannequin_still_there:
						self.stopping = False
						self.get_logger().info("Mannequin disappeared → resuming motion")
					else:
						self.get_logger().info("Mannequin still present → staying stopped")
			
					mannequin.resume_timer_active = False
					mannequin.is_being_avoided = False  # Reset the flag
			
				self.create_timer(5.0, check_and_resume)	


	def handle_stop_sign(self):
		stop_sign = next((obs for obs in self.tracked_obstacles if obs.label == "stop-sign"), None)
		if not stop_sign:
			return
		
		if stop_sign.is_being_avoided:
			return  # Already stopped for this one
		
		if not self.stopping:
			self.stopping = True
			stop_sign.is_being_avoided = True
			self.get_logger().info("Stop sign detected. Stopping robot for 3 seconds.")
		
			def resume():
				self.stopping = False
				stop_sign.is_being_avoided = False  # Reset after handling
				self.get_logger().info("Resuming movement after stop.")
			
			self.create_timer(3.0, resume)

		
	def handle_other_obstacle(self):

		self.get_logger().info(f"Avoiding {self.avoid_obstacle.label} in {self.avoid_obstacle.lane} lane")
		if self.which_lane == 'left' and self.avoid_obstacle.lane == 'left':
			self.which_lane = 'right'
		elif self.which_lane == 'right' and self.avoid_obstacle.lane == 'right':
			self.which_lane = 'left'
		
		return

	#helper functions
	def transform_to_odom(self, x, y, z, current_time, frame_id='camera_link'):
		try:
			point = PointStamped()
			point.header.frame_id = frame_id
			point.header.stamp = current_time.to_msg()
			point.point.x = x
			point.point.y = y
			point.point.z = z

			# Use timeout to ensure TF is ready
			if self.tf_buffer.can_transform('odom', frame_id, rclpy.time.Time()):
				transform = self.tf_buffer.lookup_transform(
					'odom',
					frame_id,
					rclpy.time.Time(),  # latest available
					timeout=rclpy.duration.Duration(seconds=1.0)
				)
			else:    
				self.get_logger().warn(f"Transform from {frame_id} to odom not yet available")
				return None

			transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
			return (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)

		except Exception as e:
			self.get_logger().warn(f"Transform failed: {e}")
			return None

	def filter_pc(self, pointcloud, z_eps=0.05):
		raw_points = list(pc2.read_points(pointcloud, field_names=("x", "y", "z"), skip_nans=True))

		if len(raw_points) == 0:
			return None

		# Convert to regular float32 array
		points = np.array([[p[0], p[1], p[2]] for p in raw_points], dtype=np.float32)

		if points.ndim != 2 or points.shape[1] != 3:
			self.get_logger().warn("Unexpected point shape in pointcloud")
			return None

		z_min = np.min(points[:, 2])
		mask = np.abs(points[:, 2] - z_min) < z_eps
		base_points = points[mask]

		if base_points.shape[0] == 0:
			return None
		
		return np.mean(base_points, axis=0)

	def cleanup_obstacles(self, current_time):
		"""Remove old or passed obstacles"""
		#TO DO: REDUCE LOGGING AFTER TESTING PROPERLY
		max_age = 5.0  # seconds
		min_distance = -0.2  # if bot is ahead of obstacle by 0.2 meters, remove it
		active_obstacles = []
		
		#while removign set is being avoided as false!!
		for obs in self.tracked_obstacles:
			# Remove if too old
			age = (current_time - obs.last_updated).nanoseconds / 1e9
			if age > max_age:
				if obs.is_being_avoided:
					#to return back to right if bot in left
					self.which_lane = 'right'
					obs.is_being_avoided = False

				self.get_logger().info(f"age of old_obs: {age}")
				self.get_logger().info(f"Removing old obstacle")
				continue
		
			# Remove if passed and not being avoided
			distance = obs.odom_position[0] - self.robot_x
			if distance < min_distance:
				if obs.is_being_avoided:
					self.which_lane = 'right'
					obs.is_being_avoided = False

				self.get_logger().info(f"distance: {distance}, robot_x: {self.robot_x}, obs_x: {obs.odom_position[0]}")
				self.get_logger().info(f"Removing passed obstacle")
				#set avoiding_mode as false (state of bot, not obstacle)
				continue
			
			#if age > max_age or distance < min_distance:
				# Reset avoidance state
				#if obs.is_being_avoided:
				#	self.which_lane = 'right'
				#	obs.is_being_avoided = 'false'
				#continue  # Skip adding to active_obstacles

			active_obstacles.append(obs)
	
		self.tracked_obstacles = active_obstacles
		self.get_logger().info(f"Currently tracking {len(self.tracked_obstacles)} obstacles.")



def main(args=None):
	rclpy.init(args=args)
	node = LaneFollowerNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
