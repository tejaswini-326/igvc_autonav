#!/usr/bin/env python3
"""Optimized: Node to integrate YOLO object detection with ROS2 and publish 3D positions, labels, annotated image, and object point clouds."""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from object_detection.msg import ObjectData, ObjectArray
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw, ImageFont
import os
from ament_index_python.packages import get_package_share_directory
import torch

CONFIDENCE_THRESHOLD = 0.5
FRAME_INTERVAL_SEC = 0.15  # Process every 150 ms

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
        self.last_processed_time = self.get_clock().now()

        self.fx = 246.49
        self.fy = 246.49
        self.cx = 300.0
        self.cy = 300.0

    def depth_callback(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth image conversion failed: {e}')

    def pc_callback(self, msg):
        self.latest_pc = msg

    def image_callback(self, msg):
        now = self.get_clock().now()
        if (now - self.last_processed_time).nanoseconds * 1e-9 < FRAME_INTERVAL_SEC:
            return
        self.last_processed_time = now

        if self.depth_img is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image decoding error: {e}')
            return

        results = self.model.predict(image, device=0)[0]
        detections = results.boxes
        if not detections or len(detections) == 0:
            return
            
        stamp = now.to_msg()
        frame_id = self.latest_pc.header.frame_id if self.latest_pc else "camera_link"
        object_pointclouds = []
        msg_out = ObjectArray()
        msg_out.objects = []

        # Convert to PIL for drawing
        image = PILImage.fromarray(cv_image)
        draw = ImageDraw.Draw(image)

        names = self.model.names  # Cache class names once

        for box in detections:
            if float(box.conf[0]) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            label = names[int(box.cls[0].item())]

            # Annotate
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=2)
            draw.text((xmin, max(ymin - 12, 0)), f"{label} ({box.conf[0]:.2f})", fill=(255, 255, 255))

            points_3d = self.get_points_from_depth(xmin, ymin, xmax, ymax)
            if points_3d.shape[0] == 0:
                continue

            centroid = np.mean(points_3d, axis=0)
            obj = ObjectData()
            obj.label = label
            obj.position = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]))

            if self.latest_pc:
                header = Header(stamp=stamp, frame_id=frame_id)
                obj.pointcloud = pc2.create_cloud_xyz32(header, points_3d.astype(np.float32))

            msg_out.objects.append(obj)
            object_pointclouds.append(points_3d)

        self.object_pub.publish(msg_out)

        if object_pointclouds:
            all_points = np.concatenate(object_pointclouds, axis=0)
            header = Header(stamp=stamp, frame_id=frame_id)
            self.pc_pub.publish(pc2.create_cloud_xyz32(header, all_points.astype(np.float32)))

        try:
            # Convert annotated PIL image back to NumPy array then to ROS msg
            annotated_cv_img = np.array(image)
            img_msg = self.bridge.cv2_to_imgmsg(annotated_cv_img, encoding="bgr8")
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = frame_id
            self.annotated_img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Annotated image publish failed: {e}")

    def get_points_from_depth(self, xmin, ymin, xmax, ymax):
        margin = 5
        stride = 2
        depth = self.depth_img

        if depth is None:
            return np.empty((0, 3), dtype=np.float32)

        h, w = depth.shape
        xmin = np.clip(xmin + margin, 0, w - 1)
        xmax = np.clip(xmax - margin, 0, w - 1)
        ymin = np.clip(ymin + margin, 0, h - 1)
        ymax = np.clip(ymax - margin, 0, h - 1)

        region = depth[ymin:ymax:stride, xmin:xmax:stride]
        ys, xs = np.indices(region.shape)
        z = region

        mask = np.isfinite(z) & (z > 0)
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32)

        us = xs[mask] * stride + xmin
        vs = ys[mask] * stride + ymin
        z = z[mask]

        x = (us - self.cx) * z / self.fx
        y = (vs - self.cy) * z / self.fy
        dist = np.sqrt(x**2 + y**2 + z**2)
        valid = (dist <= 5.0) & (-y > -1.3)

        return np.stack((z[valid], -x[valid], -y[valid]), axis=1)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDataNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
