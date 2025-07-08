#!/usr/bin/env python3
"""Description: Node to integrate YOLO object detection with ROS2 and publish 3D positions, labels, annotated image, and clustered object point clouds."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from object_detection.msg import ObjectData, ObjectArray
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN
import os
from ament_index_python.packages import get_package_share_directory
import torch

CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 3

class ObjectDataNode(Node):
	def __init__(self):
		super().__init__('object_data_node')

		self.image_sub = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
		self.depth_sub = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
		self.pc_sub = self.create_subscription(PointCloud2, '/camera/points_downsampled', self.pc_callback, 10)

		self.object_pub = self.create_publisher(ObjectArray, 'object_data', 10)
		self.pc_pub = self.create_publisher(PointCloud2, 'object_pc', 10)
		self.annotated_img_pub = self.create_publisher(Image, '/detected_object_img', 10)

		self.bridge = CvBridge()
		pkg_share = get_package_share_directory('object_detection')
		model_path = os.path.join(pkg_share, 'models', 'best.pt')
		self.model = YOLO(model_path)
		if torch.cuda.is_available():
			self.model.to('cuda')
			self.get_logger().info(f"Model moved to {next(self.model.model.parameters()).device}")
		self.get_logger().info('YOLOv8 model loaded.')

		self.depth_img = None
		self.latest_pc = None

		self.fx = 246.49 #camera intrinsics
		self.fy = 246.49
		self.cx = 300.0
		self.cy = 300.0

		self.frame_count = 0

	def depth_callback(self, msg):
		try:
			self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
		except Exception as e:
			self.get_logger().error(f'Depth image conversion failed: {e}')

	def pc_callback(self, msg):
		self.latest_pc = msg

	def image_callback(self, msg):
		if self.depth_img is None:  #get depth image
			self.get_logger().warn("No depth image yet.")
			return

		self.frame_count += 1
		if self.frame_count % FRAME_SKIP != 0: #skip every 2-3 frames for improved speed
			return

		try:
			image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
		except Exception as e:
			self.get_logger().error(f'Image decoding error: {e}')
			return

		results = self.model.predict(image, device=0)[0]
		detections = results.boxes
		if not detections or len(detections) == 0:
			return

		stamp = self.get_clock().now().to_msg()
		frame_id = self.latest_pc.header.frame_id if self.latest_pc else "camera_link"
		clustered_pointclouds = []
		msg_out = ObjectArray()
		msg_out.objects = [] 

		for box in detections:
			if float(box.conf[0]) < CONFIDENCE_THRESHOLD:  #check if 
				continue

			xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
			cls_id = int(box.cls[0].item())
			label = self.model.names[cls_id]

			# Draw bounding box
			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
			text = f"{label} ({box.conf[0]:.2f})"
			cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

			points_3d = self.get_points_from_depth(xmin, ymin, xmax, ymax)
			if points_3d.shape[0] == 0:
				continue

			 # Initialize the list

			clustering = DBSCAN(eps=0.15, min_samples=10).fit(points_3d)
			for i, cluster_id in enumerate(set(clustering.labels_)):
				if cluster_id == -1:
					continue
				cluster_points = points_3d[clustering.labels_ == cluster_id]
				if cluster_points.shape[0] == 0:
					continue

				clustered_pointclouds.append(cluster_points)

				centroid = np.mean(cluster_points, axis=0)
				x, y, z = centroid

				obj = ObjectData()
				obj.label = label 
				obj.position = Point(x=float(x), y=float(y), z=float(z))

				if self.latest_pc:
					header = Header(stamp=stamp, frame_id=frame_id)
					obj.pointcloud = pc2.create_cloud_xyz32(header, cluster_points.tolist())

				msg_out.objects.append(obj)

		self.object_pub.publish(msg_out)

		if clustered_pointclouds:
			all_points = np.concatenate(clustered_pointclouds, axis=0)
			header = Header(stamp=stamp, frame_id=frame_id)
			combined_pc = pc2.create_cloud_xyz32(header, all_points.tolist())
			self.pc_pub.publish(combined_pc)

		try:
			img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
			img_msg.header.stamp = stamp
			img_msg.header.frame_id = frame_id
			self.annotated_img_pub.publish(img_msg)
		except Exception as e:
			self.get_logger().warn(f"Annotated image publish failed: {e}")


	def get_points_from_depth(self, xmin, ymin, xmax, ymax):
		margin = 5
		xmin += margin
		xmax -= margin
		ymin += margin
		ymax -= margin

		depth = self.depth_img
		if depth is None:
			return np.empty((0, 3), dtype=np.float32)

		height, width = depth.shape
		xmin = np.clip(xmin, 0, width - 1)
		xmax = np.clip(xmax, 0, width - 1)
		ymin = np.clip(ymin, 0, height - 1)
		ymax = np.clip(ymax, 0, height - 1)

		region = depth[ymin:ymax, xmin:xmax]
		ys, xs = np.indices(region.shape)
		zs = region

		mask = np.isfinite(zs) & (zs > 0)
		if not np.any(mask):
			return np.empty((0, 3), dtype=np.float32)

		us = xs[mask] + xmin
		vs = ys[mask] + ymin
		zs = zs[mask]

		x = (us - self.cx) * zs / self.fx
		y = (vs - self.cy) * zs / self.fy
		z = zs

		dist = np.sqrt(x**2 + y**2 + z**2)
		valid = (dist <= 5.0) & (-y > -1.3)

		return np.stack([z[valid], -x[valid], -y[valid]], axis=1)

def main(args=None):
	rclpy.init(args=args)
	node = ObjectDataNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
