#!/usr/bin/env python3

# Detects object position and category using YOLO and PointCloud2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from object_detection.msg import ObjectData
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO


class ObjectDataNode(Node):
    def __init__(self):
        super().__init__('object_data_node')

        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.pc_callback,
            10)

        # Publisher
        self.object_pub = self.create_publisher(ObjectData, 'object_data', 10)
        self.pc_pub = self.create_publisher(PointCloud2, 'object_pc', 10)

        self.latest_image = None
        self.latest_pc = None

        # Load YOLOv8 model
        self.get_logger().info('Loading YOLOv8 model...')
        self.model = YOLO('/home/tejaswini/Desktop/abhiyaan/best.pt')  # Update path if needed
        self.model.eval()
        self.get_logger().info('YOLOv8 model loaded.')

    def pc_callback(self, msg):
        self.latest_pc = msg

    def image_callback(self, msg):
        if self.latest_pc is None:
            self.get_logger().warn("No point cloud received yet.")
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            return

        self.latest_image = image

        results = self.model(image)[0]
        detections = results.boxes

        if detections is None or len(detections) == 0:
            return

        for box in detections:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id]

            # Debug draw
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            points_3d = self.get_points_in_bbox(xmin, ymin, xmax, ymax, self.latest_pc)
            self.get_logger().info(f"Detected '{label}' in bbox ({xmin},{ymin})→({xmax},{ymax}) with {len(points_3d)} points")

            if len(points_3d) > 0:
                centroid = np.mean(points_3d, axis=0)
                x, y, z = centroid
                self.get_logger().info(f"{label}: Centroid X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

                msg = ObjectData()
                msg.label = label
                msg.position = Point(x=float(x), y=float(y), z=float(z))

                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.latest_pc.header.frame_id

                msg.pointcloud = pc2.create_cloud_xyz32(header, points_3d.tolist())
                self.object_pub.publish(msg)
                self.pc_pub.publish(msg.pointcloud)

        # Show debugging image
        cv2.imshow("YOLO 3D Detection", image)
        cv2.waitKey(1)

    def get_points_in_bbox(self, xmin, ymin, xmax, ymax, cloud_msg):
        self.get_logger().info(f"Filtering point cloud for bbox: ({xmin}, {ymin}) → ({xmax}, {ymax})")

        # Use approximate intrinsics (replace with actual if known)
        fx, fy = 525.0, 525.0
        cx, cy = 319.5, 239.5

        selected_points = []

        for pt in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = pt

            # Check for bad depth or NaNs
            if not np.isfinite(z) or z <= 0.1:
                continue

            try:
                u = int(fx * x / z + cx)
                v = int(fy * y / z + cy)
            except (ZeroDivisionError, OverflowError, ValueError):
                continue

            if xmin <= u <= xmax and ymin <= v <= ymax:
                selected_points.append([x, y, z])

        return np.array(selected_points, dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDataNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
