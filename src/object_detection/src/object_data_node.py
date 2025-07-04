#!/usr/bin/env python3

'''Description: This node integrates the object detection model with ROS, publishing labels, object position, an annotated image with bounding boxes and the point cloud of the object'''

#importing libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from object_detection.msg import ObjectData
import cv2
import numpy as np
import math
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN
import os
from ament_index_python.packages import get_package_share_directory


CONFIDENCE_THRESHOLD = 0.5


class ObjectDataNode(Node): #node constructor
    def __init__(self):
        super().__init__('object_data_node')

        #subscribers to camera image, depth image and point cloud
        self.image_sub = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, '/camera/points_downsampled', self.pc_callback, 10)

        #publishers to output custom message, object point cloud and image with predictions
        self.object_pub = self.create_publisher(ObjectData, 'object_data', 10)
        self.pc_pub = self.create_publisher(PointCloud2, 'object_pc', 10)
        self.annotated_img_pub = self.create_publisher(Image, '/detected_object_img', 10) #changed to raw image for rviz
        self.bridge = CvBridge()
        #model and utils
        pkg_share = get_package_share_directory('object_detection')
        model_path = os.path.join(pkg_share, 'models', 'best.pt')
        self.model = YOLO(model_path)
        self.get_logger().info('YOLOv8 model loaded.')

        #initialising variables
        self.depth_img = None
        self.latest_pc = None
        #camera intrinsics from urdf
        self.fx = 328.6571260705443
        self.fy = 328.6571260705443
        self.cx = 400.0
        self.cy = 400.0
        self.frame_count = 0  #used to skip frames for speed

        # White color thresholds (HSV space)
        self.lower_white = np.array([0, 0, 110])
        self.upper_white = np.array([25, 60, 255])

    def detect_potholes(self, image):
        """Detect white circles using HSV filtering and Hough transform"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        # Apply morphological operations
        kernel_open = np.ones((3, 3), np.uint8)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        # Fill in holes inside the circle
        kernel_close = np.ones((5, 5), np.uint8)
        filled_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel_close)
        blurred = cv2.GaussianBlur(filled_mask, (9, 9), 2)
        #Detect elliptical blobs using contours + fitEllipse
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #debug_image = image.copy()

        ellipses = []
        for cnt in contours:
            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (x, y), (major_axis, minor_axis), angle = ellipse

            aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
            area = cv2.contourArea(cnt)

            if 0.7 < aspect_ratio < 2.5 and 300 < area < 5000:
                ellipses.append(ellipse)
                #cv2.ellipse(debug_image, ellipse, (0, 255, 0), 2)
                #cv2.putText(debug_image, "oval pothole", (int(x) - 20, int(y) - 20),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #cv2.imshow("4. Pothole Detection Overlay", debug_image)

        # Allow OpenCV windows to refresh
        #cv2.waitKey(1)

        return ellipses
    
    def process_potholes(self, ellipses, image):
        """Process detected ellipses and convert to 3D points, publish aggregated point cloud"""
        ellipse_points_3d = []  # Aggregate points for all ellipses

        if ellipses is not None:
            for ellipse in ellipses:
                (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                angle_rad = np.deg2rad(angle)

                # Create a mask for this ellipse
                mask = np.zeros(self.depth_img.shape, dtype=np.uint8)
                cv2.ellipse(mask, (int(center_x), int(center_y)),
                            (int(major_axis / 2), int(minor_axis / 2)),
                            angle, 0, 360, 255, -1)

                # Get all (u, v) pixel coordinates inside the ellipse
                ys, xs = np.where(mask == 255)

                points_for_this_ellipse = []
                for u, v in zip(xs, ys):
                    z = self.depth_img[v, u]
                    if np.isfinite(z) and z > 0.1:
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        points_for_this_ellipse.append([z, -x, -y])  # or your preferred frame

                if points_for_this_ellipse:
                    points_for_this_ellipse = np.array(points_for_this_ellipse, dtype=np.float32)
                    # Filter out any NaN or infinite points
                    points_for_this_ellipse = points_for_this_ellipse[np.all(np.isfinite(points_for_this_ellipse), axis=1)]
                    ellipse_points_3d.append(points_for_this_ellipse)


                    centroid = np.mean(points_for_this_ellipse, axis=0)
                    x, y, z = centroid

                    msg_out = ObjectData()
                    msg_out.label = 'pothole'
                    msg_out.position = Point(x=float(x), y=float(y), z=float(z))

                    header = Header()
                    header.stamp = self.get_clock().now().to_msg()
                    header.frame_id = 'camera_link'
                    msg_out.pointcloud = pc2.create_cloud_xyz32(header, points_for_this_ellipse.tolist())

                    self.object_pub.publish(msg_out)
                    self.get_logger().info(f"Published pothole ObjectData with {len(points_for_this_ellipse)} points.")

            if ellipse_points_3d:
                ellipse_points_3d = np.concatenate(ellipse_points_3d, axis=0)

                # Filter out any NaN or infinite points
                ellipse_points_3d = ellipse_points_3d[np.all(np.isfinite(ellipse_points_3d), axis=1)]

        return ellipse_points_3d
    
    def depth_callback(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1') #load and decode depth image
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def pc_callback(self, msg):
        self.latest_pc = msg #load and store latest point cloud

    def image_callback(self, msg):
        if self.depth_img is None: #check if depth image is available
            self.get_logger().warn("No depth image received yet.")
            return

        self.frame_count += 1
        if self.frame_count % 3 != 0:  #skip every 2 out of 3 frames for faster inference
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #load and store latest camera image
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            return


        stamp = self.get_clock().now().to_msg()
        frame_id = self.latest_pc.header.frame_id if self.latest_pc else "camera_link"
        all_points = []

        # Detect potholes (white circles)
        potholes = self.detect_potholes(image)
        pothole_points_3d = self.process_potholes(potholes, image)

        # Add pothole points to all_points
        if len(pothole_points_3d) > 0:
            all_points.extend(np.asarray(pothole_points_3d).tolist())

        results = self.model(image)[0] #get labels and bounding boxes using model
        detections = results.boxes #get bounding boxes

        if len(detections) == 0:
            if all_points:
                header = Header(stamp=stamp, frame_id=frame_id)
                combined_pc = pc2.create_cloud_xyz32(header, all_points)
                self.pc_pub.publish(combined_pc)
            return

        for box in detections:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist()) #get label, bounding box coords, and confidence of prediction
            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id]
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            #drawing the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{label} ({confidence:.2f})"
            cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            points_3d = self.get_points_from_depth(xmin, ymin, xmax, ymax)  #load pointcloud of object within bounding box

            if len(points_3d) > 0: 
                all_points.extend(points_3d.tolist())  #all points stores combined pointcloud multiple objects
                # Run DBSCAN clustering on the 3D points
                clustering = DBSCAN(eps=0.15, min_samples=10).fit(points_3d)
                labels_db = clustering.labels_

                unique_labels = set(labels_db)
                for cluster_label in unique_labels:
                    if cluster_label == -1:
                        continue  # Skip noise

                    # Extract points in this cluster
                    cluster_points = points_3d[labels_db == cluster_label]
                    if len(cluster_points) == 0:
                        continue

                    # Compute centroid
                    centroid = np.mean(cluster_points, axis=0)
                    x, y, z = centroid
                    self.get_logger().debug(f"{label} Cluster {cluster_label}: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

                    # Fill and publish custom message
                    msg_out = ObjectData()
                    msg_out.label = label  # Optional: add cluster index or size if needed
                    msg_out.position = Point(x=float(x), y=float(y), z=float(z))

                    if self.latest_pc: #stores latest pc
                        header = Header(stamp=stamp, frame_id=frame_id)
                        msg_out.pointcloud = pc2.create_cloud_xyz32(header, cluster_points.tolist())

                    self.object_pub.publish(msg_out) #publishing the pc of a particular object
                    all_points.extend(cluster_points.tolist())#adding the object pc to a combined pc

        #publish combined point cloud
        if all_points:
            header = Header(stamp=stamp, frame_id=frame_id)
            combined_pc = pc2.create_cloud_xyz32(header, all_points)
            self.pc_pub.publish(combined_pc)

        #publish annotated image for rviz (raw format instead of compressed)
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8") #convert to ros image
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = frame_id
            self.annotated_img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Could not publish annotated image: {e}")

    #function to map points within bounding box to point cloud
    def get_points_from_depth(self, xmin, ymin, xmax, ymax):
        #extending bounding box margins to include more points in case bb is too small
        margin = 5
        xmin += margin
        xmax -= margin
        ymin += margin
        ymax -= margin

        if self.depth_img is None: #check if corresponding depth image exists
            return []

        points = []
        depth = self.depth_img
        height, width = depth.shape

        region = depth[ymin:ymax, xmin:xmax]
        ys, xs = np.indices(region.shape)
        zs = region
        mask = np.isfinite(zs) & (zs > 0)

        xs, ys, zs = xs[mask], ys[mask], zs[mask]
        us = xs + xmin
        vs = ys + ymin

        x = (us - self.cx) * zs / self.fx
        y = (vs - self.cy) * zs / self.fy
        z = zs

        dist = np.sqrt(x**2 + y**2 + z**2)
        valid = (dist <= 5.0) & (-y > -1.3)

        points = np.stack([z[valid], -x[valid], -y[valid]], axis=1)
        return points


#instantiation and spinning of the node
def main(args=None):
    rclpy.init(args=args)
    node = ObjectDataNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    #no need to call destroyAllWindows since imshow is not used


if __name__ == '__main__':
    main()
