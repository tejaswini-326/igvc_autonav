# === Configurable Parameters (Hardcoded for Tuning) ===
FX = 246.49
FY = 246.49
CX = 300.0
CY = 300.0

LOWER_WHITE = [0, 0, 110]
UPPER_WHITE = [180, 63, 255]

ASPECT_RATIO_MIN = 1.0
ASPECT_RATIO_MAX = 4.3
NORMALIZED_AREA_MIN = 14000
NORMALIZED_AREA_MAX = 60000
SOLIDITY_THRESHOLD = 0.70
SKIP_FRAME_RATIO = 1  # Process every 3rd frame
Y_HEIGHT_THRESHOLD = 1.3

#look into K-means colour clustering (slower)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import numpy as np
from geometry_msgs.msg import Point, PointStamped
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Bool

class PotholeDetectorNode(Node):
    def __init__(self):
        super().__init__('pothole_detector_node')
        self.subscription = self.create_subscription(Image,'/camera/image',self.image_callback,10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.bridge = CvBridge()
        self.get_logger().info("PotholeDetectorNode initialized.")
        self.pc_pub = self.create_publisher(PointCloud2, '/pothole', 10)
        self.pothole_bool_pub = self.create_publisher(Bool, 'pothole_detected', 10)
        self.pothole_pos_pub = self.create_publisher(PointStamped, '/pothole_position', 10)


        #camera intrinsics from urdf
        self.fx = 246.49
        self.fy = 246.49
        self.cx = 300.0
        self.cy = 300.0
        self.frame_count = 0  #used to skip frames for speed

        #initialising variables
        self.depth_img = None

        # White color thresholds (HSV space)
        self.lower_white = np.array(LOWER_WHITE)
        self.upper_white = np.array(UPPER_WHITE)
        self.create_timer(0.05, self.process_latest_image) 
        self.latest_image = None
        self.latest_depth = None
    
    def process_latest_image(self):
        if self.latest_image is not None and self.latest_depth is not None:
            image = self.latest_image.copy()
            depth = self.latest_depth.copy()
            self.depth_image = depth
            potholes = self.detect_potholes(image, depth)
            self.process_potholes(potholes, image, depth)

    def detect_potholes(self, image, depth):  
        """Detect white circles using HSV filtering and Hough transform"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #cv2.imshow("1 - HSV Image", hsv)

        # Step 2: Crop top portion to ignore sky
        h = hsv.shape[0]
        ground_roi = hsv[int(h * 0.55):, :]  # take only bottom 45% of image
        offset_y = int(h * 0.55)  # for contour shift later

        mask = cv2.inRange(ground_roi, self.lower_white, self.upper_white)
        #cv2.imshow("2 - White Mask", mask)
        # Apply morphological operations
        #kernel_open = np.ones((2, 2), np.uint8)
        #opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        #cv2.imshow("3 - Opened Mask", opened_mask)
        # Fill in holes inside the circle
        kernel_close = np.ones((5, 5), np.uint8)
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        #cv2.imshow("4 - Filled Mask", filled_mask)
        
        blurred = cv2.GaussianBlur(filled_mask, (5, 5), 2)
        #cv2.imshow("5 - Blurred Mask", blurred)

        #Detect elliptical blobs using contours + fitEllipse
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_image = image.copy()

        img_h, img_w, _ = debug_image.shape

        ellipses = []
        for cnt in contours:
            if len(cnt) < 20:
                continue

            # FILTER 1: Reject contours touching image border: LANE MARKINGS
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= 2 or y+offset_y <= 2 or (x + w) >= img_w - 2 or (y+offset_y + h) >= img_h - 2:
                #self.get_logger().info("Contour skipped: touches edge")
                continue

            ellipse = cv2.fitEllipse(cnt)
            (x, y), (major_axis, minor_axis), angle = ellipse
            y += offset_y  # shift y back to full image space
            ellipse = ((x, y), (major_axis, minor_axis), angle)
            
            aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            if hull_area == 0:
                continue  # skip invalid contours

            solidity = area / hull_area
            #self.get_logger().info(f"Solidity: {solidity}, hull area: {hull_area}")
            if solidity < SOLIDITY_THRESHOLD:
                continue  # likely not a filled region

            cy, cx = int(y), int(x)
            if cy < 0 or cy >= depth.shape[0] or cx < 0 or cx >= depth.shape[1]:
                #self.get_logger().warn(f"Skipping contour: ({cx},{cy}) out of bounds for depth size {depth.shape}")
                continue
            z_val = depth[cy, cx]
            z = float(z_val) if np.isfinite(z_val) else None


            if z is None or not np.isfinite(z) or z <= 0.1 or z > 10.0:
                #self.get_logger().info(f"Skipping contour: invalid depth at ({cx},{cy})")
                continue

            normalized_area = area * (z ** 2)
            #self.get_logger().info(f"z= {z}, aspect ratio = {aspect_ratio}, Narea = {normalized_area}")
                
            if ASPECT_RATIO_MIN < aspect_ratio < ASPECT_RATIO_MAX and NORMALIZED_AREA_MIN < normalized_area < NORMALIZED_AREA_MAX:
                ellipses.append(ellipse)
                cv2.ellipse(debug_image, ellipse, (0, 255, 0), 2)
                cv2.putText(debug_image, "oval pothole", (int(x) - 20, int(y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #cv2.imshow("4. Pothole Detection Overlay", debug_image)

        # Allow OpenCV windows to refresh
        #cv2.waitKey(1)

        return ellipses
    
    def voxel_grid_filter(self, points, voxel_size=0.05):
        if len(points) == 0:
            return points

        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        voxel_dict = {}
        for idx, voxel in enumerate(voxel_indices):
            key = tuple(voxel)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[idx])

        downsampled_points = []
        for pts in voxel_dict.values():
            pts = np.array(pts)
            centroid = np.mean(pts, axis=0)
            downsampled_points.append(centroid)

        return np.array(downsampled_points)


    def process_potholes(self, ellipses, image, depth):
        """Process detected ellipses and convert to 3D points, publish aggregated point cloud"""
        pothole_found = False
        for ellipse in ellipses:
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            
            # Create a mask for this ellipse
            mask = np.zeros(depth.shape, dtype=np.uint8)
            cv2.ellipse(mask, (int(center_x), int(center_y)),
                        (int(major_axis / 2), int(minor_axis / 2)),
                        angle, 0, 360, 255, -1)

            # Get all (u, v) pixel coordinates inside the ellipse
            ys, xs = np.where(mask == 255)
            sample_rate = 10 
            points = []
            for u, v in zip(xs[::sample_rate], ys[::sample_rate]):
                z_val = depth[v, u]
                if np.isfinite(z_val) and z_val > 0.1:
                    z = float(z_val)
                    x = float((u - self.cx) * z / self.fx)
                    y = float((v - self.cy) * z / self.fy)
                    points.append([z, -x, -y])


            if points:
                points = np.array(points, dtype=np.float32)
                # Filter out any NaN or infinite points
                points = points[np.all(np.isfinite(points), axis=1)]
                
                # Downsample points using voxel grid filter
                points = self.voxel_grid_filter(points, voxel_size=0.05)

                if len(points) == 0:
                    continue

                centroid = np.mean(points, axis=0)
                x, y, z = centroid  # x=forward, y=left, z=up

                if not np.all(np.isfinite([x, y, z])):
                    continue

                if z > Y_HEIGHT_THRESHOLD:  # height filter
                    #self.get_logger().info(f"Rejected ellipse due to height z={z:.2f}")
                    continue

                # Publish point cloud for debug
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'camera_link'
                msg_out = pc2.create_cloud_xyz32(header, points.tolist())
                self.pc_pub.publish(msg_out)

                # Publish centroid position
                pothole_point = Point()
                pothole_point.x = float(x)
                pothole_point.y = float(y)
                pothole_point.z = float(z)
                pothole_msg = PointStamped()
                pothole_msg.header = header  # same header as point cloud
                pothole_msg.point = pothole_point
                self.pothole_pos_pub.publish(pothole_msg)


                pothole_found = True
                self.get_logger().info(
                    f"Published pothole centroid at (x={x:.2f}, y={y:.2f}, z={z:.2f}) with {len(points)} points."
                )


        self.pothole_bool_pub.publish(Bool(data=pothole_found))   

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')


#instantiation and spinning of the node
def main(args=None):
    rclpy.init(args=args)
    node = PotholeDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    #no need to call destroyAllWindows since imshow is not used


if __name__ == '__main__':
    main()