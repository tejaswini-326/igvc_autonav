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
import time

class YellowParkingDetector(Node):
    def __init__(self):
        super().__init__('yellow_parking_detector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self.pointcloud_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state = "IN_PARKING_SLOT"
        
    def detect_yellow_color(self, r, g, b):
        pale_threshold = 60
        color_balance_threshold = 50

        avg_color = (r + g + b) / 3

        is_bright_enough = (r > pale_threshold and g > pale_threshold)
        has_yellow_tint = (r > b and g > b) 
        is_balanced = (abs(r - avg_color) < color_balance_threshold and abs(g - avg_color) < color_balance_threshold)    

        yellow_factor = (r + g) / (2 * max(b, 1))

        return (is_bright_enough and has_yellow_tint and is_balanced and yellow_factor > 1.1 and avg_color > 70)
    
    def pull_back(self):
        cmd = Twist()
        cmd.linear.x = -0.5
        cmd.angular.z = 0.0
        self.get_logger().info("Pulling out")
        self.cmd_pub.publish(cmd)

    def pullout_arc_cmd(self):
        cmd = Twist()
        cmd.linear.x = -0.5
        cmd.angular.z = 0.2
        
        duration = 30.0  

        end_time = time.time() + duration
        while time.time() < end_time:
            self.get_logger().info("Turning")
            self.cmd_pub.publish(cmd)
            time.sleep(0.1)
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        self.get_logger().info("Turn complete, stopping.")
    
    def still_in_parking_lot(self, msg):
        height = msg.height
        width = msg.width

        if height == 0 and width == 0:
            return None, None
        
        yellow_img = np.zeros((height, width, 3), dtype = np.uint8)
        yellow_ground_points = []

        index = 0
        for point in pc2.read_points(msg, field_names = ("x","y","z","rgb"), skip_nans=False):
            x, y, z, rgb = point
            row = index // width
            col = index % width

            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                index += 1
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            except:
                index += 1
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            if self.detect_yellow_color(r, g, b):
                if -2.0 < z < -1.3 and 1.0 < x < 3.0:
                    if 0 <= row < height and 0 <= col < width:
                        yellow_img[row, col] = (0,255,255)
                    yellow_ground_points.append([x,y,z])

            index += 1
        
        try:
            if yellow_img.size > 0:
                cv2.imshow("Pale Yellow Parking Lines", yellow_img)
                cv2.waitKey(1)
        except:
            pass

        if len(yellow_ground_points) < 20:
            return None, yellow_img, []
        
        points_np = np.array(yellow_ground_points)
        points_xy = points_np[:, :2]

        eps = 0.3
        min_samples = 15

        clustering = DBSCAN(eps = eps, min_samples=min_samples).fit(points_xy)
        labels = clustering.labels_
        unique_labels = set(labels)

        line_centers = []
        cluster_y_spreads = []
        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = points_xy[labels == label]
            if len(cluster_points) < 10:
                continue
                
            center = np.mean(cluster_points, axis = 0)
            line_centers.append(center)

            # Calculate y-spread of this cluster
            y_vals = cluster_points[:, 1]
            y_spread = np.max(y_vals) - np.min(y_vals)
            cluster_y_spreads.append(y_spread)

        return line_centers, yellow_img, cluster_y_spreads

    def detect_horizontal_line(self, msg):
        yellow_y_vals = []
        
        for point in pc2.read_points(msg, field_names=("x","y","z","rgb"),skip_nans = False):
            x, y, z, rgb = point
            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0] 
            except:
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) &  0xFF
            b = rgb_int & 0xFF

            if self.detect_yellow_color(r, g, b):
                if (3.0 < x < 4.0 and -2.0 < z < -1.2):
                    yellow_y_vals.append(y)
        
        if len(yellow_y_vals) < 100:
            return False
        
        hist, bin_edges = np.histogram(yellow_y_vals, bins = 20, range = (-2.00, 2.0))
        dense_threshold = 5
        dense_bins = [bin_edges[i] for i in range(len(hist)) if hist[i] >= dense_threshold]

        if len(dense_bins) < 2:
            return False
        
        y_dense_min = min(dense_bins)
        y_dense_max = max(dense_bins)
        dense_y_range = y_dense_max - y_dense_min

        self.get_logger().info(f"[STOP CHECK] Yellow dense y-span: {dense_y_range:.2f}, total points: {len(yellow_y_vals)}")

        if dense_y_range > 2.0 and len(yellow_y_vals) > 700:
            self.get_logger().warn("DENSE HORIONTAL YELLOW STOP LINE DETECTED")
            return True

        return False

    def pointcloud_callback(self, msg):
        if self.state == "FINISHED":
            stop = Twist()
            self.cmd_pub.publish(stop)
            return         

        if self.state == "IN_PARKING_SLOT":
            if self.detect_horizontal_line(msg):
                self.pull_back()
            else:
                line_centers, _ , cluster_y_spreads = self.still_in_parking_lot(msg)
                if line_centers and len(line_centers) >=1:
                    if all(spread < 0.41 for spread in cluster_y_spreads[:2]):  
                        self.state = "TURNING"
                    else:
                        self.pull_back()

        if self.state == "TURNING":
            self.pullout_arc_cmd()
            self.state = "FINISHED"

def main(args = None):
    rclpy.init(args=args)
    node = YellowParkingDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


        
        