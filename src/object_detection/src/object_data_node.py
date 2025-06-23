#!/usr/bin/env python3

'''Description: This node integrates the object detection model with ROS, publishing labels, object position, an annotated image with bounding boxes and the point cloud of the object'''

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from object_detection.msg import ObjectData
import cv2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from ultralytics import YOLO
from cv_bridge import CvBridge


class ObjectDataNode(Node): 
    def __init__(self):
        super().__init__('object_data_node')

        #subscribers to camera image, depth image and point cloud
        self.image_sub = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, '/camera/points', self.pc_callback, 10)

        #publishers to output custom message, object point cloud and image with predictions
        self.object_pub = self.create_publisher(ObjectData, 'object_data', 10)
        self.pc_pub = self.create_publisher(PointCloud2, 'object_pc', 10)
        self.annotated_img_pub = self.create_publisher(Image, '/detected_object_img', 10) #changed to raw image for rviz

        #model and utils
        self.bridge = CvBridge()
        self.model = YOLO('/home/tejaswini/Desktop/abhiyaan/best.pt') #add model path here
        self.model.eval()
        self.get_logger().info('YOLOv8 model loaded.')

        #initialising variables
        self.depth_img = None
        self.latest_pc = None
        self.fx = 102.7348185494929  #camera intrinsics from urdf
        self.fy = 102.7348185494929
        self.cx = 160.0
        self.cy = 120.0
        self.frame_count = 0  #used to skip frames for speed

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

        results = self.model(image)[0] #get labels and bounding boxes using model
        detections = results.boxes #get bounding boxes

        if not detections or len(detections) == 0:
            return

        stamp = self.get_clock().now().to_msg()
        frame_id = self.latest_pc.header.frame_id if self.latest_pc else "camera_link"
        all_points = []

        for box in detections:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist()) #get label, bounding box coords, and confidence of prediction
            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id]
            confidence = float(box.conf[0])

            #drawing the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{label} ({confidence:.2f})"
            cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            points_3d = self.get_points_from_depth(xmin, ymin, xmax, ymax)  #load pointcloud of object within bounding box

            if len(points_3d) > 0: 
                all_points.extend(points_3d.tolist())  #all points stores combined pointcloud multiple objects
                centroid = np.mean(points_3d, axis=0) #finding centroid of individual object
                x, y, z = centroid
                self.get_logger().debug(f"{label}: Centroid X={x:.2f}, Y={y:.2f}, Z={z:.2f}") #log data

                msg_out = ObjectData() #create custom message with label, obj position and corresponding point cloud
                msg_out.label = label
                msg_out.position = Point(x=float(x), y=float(y), z=float(z))

                if self.latest_pc:
                    header = Header(stamp=stamp, frame_id=frame_id)
                    msg_out.pointcloud = pc2.create_cloud_xyz32(header, points_3d.tolist())

                self.object_pub.publish(msg_out) #publish the message

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

        for v in range(ymin, ymax): #iterate through each point, check validity of data
            if not (0 <= v < height):
                continue
            for u in range(xmin, xmax):
                if not (0 <= u < width):
                    continue
                z = depth[v, u]
                if z == 0 or np.isnan(z):
                    continue
                x = (u - self.cx) * z / self.fx  #map points from 2d to 3d using camera intrinsics
                y = (v - self.cy) * z / self.fy

                dist = np.sqrt(x**2 + y**2 + z**2)
                if dist <= 6.0:             #only publishing points within a radius to avoid nans
                    points.append([z, -x, -y]) #tranformed to align depth img with camera 

        return np.array(points, dtype=np.float32)

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
