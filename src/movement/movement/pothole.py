import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2

class PotholeDetectorNode(Node):
    def __init__(self):
        super().__init__('pothole_detector_node')
        self.subscription = self.create_subscription(Image,'/camera/image',self.image_callback,10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.bridge = CvBridge()
        self.get_logger().info("PotholeDetectorNode initialized.")
        self.pc_pub = self.create_publisher(PointCloud2, '/pothole', 10)

        #camera intrinsics from urdf
        self.fx = 246.49
        self.fy = 246.49
        self.cx = 300.0
        self.cy = 300.0
        self.frame_count = 0  #used to skip frames for speed

        #initialising variables
        self.depth_img = None
        self.latest_pc = None

        # White color thresholds (HSV space)
        self.lower_white = np.array([0, 0, 110])
        self.upper_white = np.array([25, 60, 255])

    def detect_potholes(self, image):
        """Detect white circles using HSV filtering and Hough transform"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imshow("1 - HSV Image", hsv)

        # Step 2: Crop top portion to ignore sky
        h, w = hsv.shape[:2]
        ground_roi = hsv[int(h * 0.55):, :]  # take only bottom 45% of image
        offset_y = int(h * 0.55)  # for contour shift later

        mask = cv2.inRange(ground_roi, self.lower_white, self.upper_white)
        cv2.imshow("2 - White Mask", mask)
        # Apply morphological operations
        #kernel_open = np.ones((2, 2), np.uint8)
        #opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        #cv2.imshow("3 - Opened Mask", opened_mask)
        # Fill in holes inside the circle
        kernel_close = np.ones((5, 5), np.uint8)
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        cv2.imshow("4 - Filled Mask", filled_mask)
        blurred = cv2.GaussianBlur(filled_mask, (9, 9), 2)
        cv2.imshow("5 - Blurred Mask", blurred)
        #Detect elliptical blobs using contours + fitEllipse
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

            #self.get_logger().info(f"aspect ratio = {aspect_ratio}, area = {area}")
                
            if 1 < aspect_ratio < 4 and 500 < area < 7000:
                ellipses.append(ellipse)
                cv2.ellipse(debug_image, ellipse, (0, 255, 0), 2)
                cv2.putText(debug_image, "oval pothole", (int(x) - 20, int(y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("4. Pothole Detection Overlay", debug_image)

        # Allow OpenCV windows to refresh
        cv2.waitKey(1)

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
                    ellipse_points_3d.extend(points_for_this_ellipse.tolist())

                    centroid = np.mean(points_for_this_ellipse, axis=0)
                    x, y, z = centroid

                    if y > 1.3:  # skip if too far above road (tweak threshold)
                        self.get_logger().info(f"Rejected ellipse due to height y={y:.2f}")
                        continue

                    header = Header()
                    header.stamp = self.get_clock().now().to_msg()
                    header.frame_id = 'camera_link'
                    msg_out = pc2.create_cloud_xyz32(header, points_for_this_ellipse.tolist())

                    self.pc_pub.publish(msg_out)
                    self.get_logger().info(f"Published pothole ObjectData with {len(points_for_this_ellipse)} points.")
    
    def depth_callback(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1') #load and decode depth image

        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def image_callback(self, msg):
        if self.depth_img is None: #check if depth image is available
            self.get_logger().warn("No depth image received yet.")
            return

        self.frame_count += 1
        if self.frame_count % 3 != 0:  #skip every 2 out of 3 frames for faster inference
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #load and store latest camera image
            # Detect potholes (white circles)
            potholes = self.detect_potholes(image)
            self.process_potholes(potholes, image)
            
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            return

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