import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
import math
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import Twist

WHITE_THRESHOLD = 100
COLOR_BALANCE_THRESHOLD = 25
MIN_CLUSTERING_DISTANCE = 0.971
MIN_CLUSTERING_POINTS = 20
PARKING_DIRECTION = -1

class ParallelParkingDetector(Node):
    def __init__(self):
        super().__init__('yellow_parking_detector')

        self.subscription1 = self.create_subscription(
            PointCloud2, '/camera/points', self.pointcloud_callback_1, 10)
        self.subscription2 = self.create_subscription(
            PointCloud2, '/bcamera/points', self.pointcloud_callback_2, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.state = "STARTING"
        self.dist = None
        self.y = 0
        self.centers_1 = None
        self.centers_2 = None
        self.msg_1 = None
        self.msg_2 = None
        self.PARKING_DIRECTION = 1
        self.rotation_start_time = None
        self.rotation_min_duration = 3.0    
        self.timer = self.create_timer(0.1, self.control_loop)
        self.sidelane_dist = None
        
    def control_loop(self):
        if self.state == "STARTING":
            if self.msg_1 is None:
                return
            self.pull_back()
            centers, _ = self.see_white_line(self.msg_1)
            if centers and len(centers) >= 1:
                center = centers[0][1]
                self.y = center[1]
                self.dist = np.linalg.norm(center)
                self.state = "DETECTED"
                self.get_logger().info("White line detected, moving to DETECTED state")
                if self.y < 0:
                    self.PARKING_DIRECTION = 1

        elif self.state == "DETECTED":
            if self.dist is not None:
                done = self.push_front_step()
                if done:
                    self.state = "ROTATING_IN_LINE"
                    self.get_logger().info("Moved to ROTATING_IN_LINE state")
       
        elif self.state == "ROTATING_IN_LINE":
            if self.msg_2 is None:
                return

            if self.rotation_start_time is None:
                self.rotation_start_time = self.get_clock().now()

            elapsed = (self.get_clock().now() - self.rotation_start_time).nanoseconds / 1e9

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 1.0 * self.PARKING_DIRECTION
            self.cmd_pub.publish(cmd)

            if elapsed > self.rotation_min_duration:
                centers, _ = self.see_white_line(self.msg_2)
                if centers and len(centers) >= 1:
                    self.state = "APPROACHING_SIDELINE"
                    self.get_logger().info("Side line detected. Moving to APPROACHING_SIDELINE state.")
                    self.rotation_start_time = None  

        elif self.state == "APPROACHING_SIDELINE":
            if self.msg_2 is None:
                return

            centers, _ = self.see_white_line(self.msg_2)
            if not centers or len(centers) < 1:
                self.get_logger().warn("No side line detected while approaching.")
                return

            label, center = centers[0]
            current_dist = np.linalg.norm(center)

            if self.sidelane_dist is None:
                self.sidelane_dist = current_dist - 0.3
                self.get_logger().info(f"Set target approach distance: {self.sidelane_dist:.2f} m")

            if self.sidelane_dist <= 0:
                self.state = "TURN_TO_FINAL"
                self.sidelane_dist = None
                self.get_logger().info("Reached target distance to sideline. Turning to face stop line.")
                return

            cmd = Twist()
            speed = -0.3
            cmd.linear.x = speed
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            dt = 0.1  
            self.sidelane_dist -= abs(speed) * dt

        elif self.state == "TURN_TO_FINAL":
            if self.msg_2 is None:
                return

            if self.rotation_start_time is None:
                self.rotation_start_time = self.get_clock().now()

            elapsed = (self.get_clock().now() - self.rotation_start_time).nanoseconds / 1e9

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = -1.0 * self.PARKING_DIRECTION  
            self.cmd_pub.publish(cmd)

            if elapsed > self.rotation_min_duration:
                centers, _ = self.see_white_line(self.msg_2)
                if centers and len(centers) >= 1:
                    self.state = "FINAL_STRETCH"
                    self.get_logger().info("Turn complete. Looking for stop line.")
                    self.rotation_start_time = None  

        elif self.state == "FINAL_STRETCH":
            if self.msg_2 is None:
                return
            cond, distance = self.detect_horizontal_line(self.msg_2)
            if distance is None:
                distance = 0.5
            pull_back_speed = -0.1
            duration = distance / abs(pull_back_speed)

            if not hasattr(self, 'pull_back_start_time'):
                self.pull_back_start_time = self.get_clock().now()
                self.pull_back_duration = duration

            elapsed = (self.get_clock().now() - self.pull_back_start_time).nanoseconds / 1e9
            if elapsed < self.pull_back_duration:
                cmd = Twist()
                cmd.linear.x = pull_back_speed
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
            else:
                stop_cmd = Twist()
                stop_cmd.linear.x = 0.0
                stop_cmd.angular.z = 0.0
                self.cmd_pub.publish(stop_cmd)
                self.get_logger().info("Completed final pull back. Parking complete.")
                self.state = "PARKING_COMPLETE"
                del self.pull_back_start_time
                del self.pull_back_duration

        elif self.state == "PARKING_COMPLETE":
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.cmd_pub.publish(stop_cmd)

    def pull_back(self):
        cmd = Twist()
        cmd.linear.x = -0.5
        cmd.angular.z = 0.0
        self.get_logger().info("Pulling back to see lines")
        self.cmd_pub.publish(cmd)

    def push_front_step(self):
        if self.dist is None or self.dist <= 0:
            self.cmd_pub.publish(Twist())  # stop
            self.get_logger().info("Finished pushing front")
            return True  
        else:
            cmd = Twist()
            speed = 0.5  
            timer_period = 0.1  
            step_dist = speed * timer_period
            cmd.linear.x = speed
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.dist -= step_dist
            return False  

    def see_white_line(self, msg):
        height = msg.height
        width = msg.width

        if height == 0 and width == 0:
            return None, None

        white_img = np.zeros((height, width, 3), dtype=np.uint8)
        white_ground_points = []
        index = 0

        for point in pc2.read_points(msg, field_names=("x","y","z","rgb"), skip_nans=False):
            x, y, z, rgb = point 
            row = index // width
            col = index % width 

            if rgb is None or math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue

            try:
                rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
            except:
                continue

            r = (rgb_int >> 16) & 0xFF
            g = (rgb_int >> 8) & 0xFF
            b = rgb_int & 0xFF

            avg_color = (r + g + b) /3

            # After collecting white points (before applying z filter)
            z_vals = [z for x, y, z in white_ground_points]

            if len(z_vals) > 10:
                z_median = np.median(z_vals)
                z_std = np.std(z_vals)
                lower_z = z_median - 0.1 * z_std
                upper_z = z_median + 0.1 * z_std
            else:
                lower_z = -1.4  
                upper_z = -1.3

            if (r > WHITE_THRESHOLD and g > WHITE_THRESHOLD and b > WHITE_THRESHOLD and
                abs(r - avg_color) < COLOR_BALANCE_THRESHOLD and
                abs(g - avg_color) < COLOR_BALANCE_THRESHOLD and
                abs(b - avg_color) < COLOR_BALANCE_THRESHOLD):
                if lower_z < z < upper_z and 0.0 < x < 5.0:
                    white_img[row, col] = (255, 255, 255)
                    white_ground_points.append([x, y, z])
            index += 1

        if len(white_ground_points) < 10:
            self.get_logger().warn("Not enough white points for detection")
            return None, None
        
        points_np = np.array(white_ground_points)
        points_xy = points_np[:, :2]
        clustering = DBSCAN(eps=MIN_CLUSTERING_DISTANCE, min_samples=MIN_CLUSTERING_POINTS).fit(points_xy)
        labels = clustering.labels_

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        self.get_logger().info(f"Clusters found: {n_clusters}, Noise points: {n_noise}")

        line_centers = []
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points_xy[labels == label]
            if len(cluster_points) < 10:
                continue
            center = np.mean(cluster_points, axis=0)
            line_centers.append((label, center))
        return line_centers, white_img

    def detect_horizontal_line(self, msg):
        white_y_vals = []
        white_x_vals = []

        for point in pc2.read_points(msg, field_names=("x","y","z","rgb"),skip_nans=False):
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

            avg_color = (r + g + b) / 3
            if (r > WHITE_THRESHOLD and g > WHITE_THRESHOLD and b > WHITE_THRESHOLD and
                abs(r - avg_color) < COLOR_BALANCE_THRESHOLD and
                abs(g - avg_color) < COLOR_BALANCE_THRESHOLD and
                abs(b - avg_color) < COLOR_BALANCE_THRESHOLD):
                if -0.1 < z < 0.1 and 0.0 < x < 5.0:
                    white_x_vals.append(x)
                    white_y_vals.append(y)

        if len(white_x_vals) > 50:
            return True, min(white_x_vals)
        else:
            return False, None

    def pointcloud_callback_1(self, msg):
        self.msg_1 = msg

    def pointcloud_callback_2(self, msg):
        self.msg_2 = msg


def main(args=None):
    rclpy.init(args=args)
    node = ParallelParkingDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
