'''Description: Node that creates and publishes a costmap od bot's surroundings'''

#importing libraries
import rclpy
import cv2
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
# import tf2_geometry_msgs
# from geometry_msgs.msg import PointStamped


class CostmapNode(Node): #constructor for costmap node
    def __init__(self):
        super().__init__('costmap_node')

        #constmap parameters
        self.resolution = 0.05  #meters per cell
        self.width = 500
        self.height = 500
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.frame_id = 'odom'

        #costmap publisher
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap', 10)

        #subbscriptions
        self.object_pc_sub = self.create_subscription(PointCloud2, '/object_pc', self.object_pc_callback, 10)
        self.wlane_pc_sub = self.create_subscription(PointCloud2, '/white_lane_points', self.white_lane_pc_callback, 10)
        self.ylane_pc_sub = self.create_subscription(PointCloud2, '/yellow_lane_points', self.yellow_lane_pc_callback, 10)
        

        #TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for costmap callback
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        #internal maps
        self.white_map = None
        self.yellow_map = None
        self.object_map = None

    #callback functions for point cloud data
    def object_pc_callback(self, msg):
        self.generate_costmap(msg, tag="object")

    def white_lane_pc_callback(self, msg):
        self.generate_costmap(msg, tag="white")

    def yellow_lane_pc_callback(self, msg):
        self.generate_costmap(msg, tag="yellow")


    def timer_callback(self):
        white = self.white_map if self.white_map is not None else np.zeros((self.height, self.width), dtype=np.uint8)
        yellow = self.yellow_map if self.yellow_map is not None else np.zeros((self.height, self.width), dtype=np.uint8)
        objects = self.object_map if self.object_map is not None else np.zeros((self.height, self.width), dtype=np.uint8)

        combined = np.maximum.reduce([white, yellow, objects])
        self.publish_costmap(combined, self.get_clock().now().to_msg())


    #function to generate costmap
    def generate_costmap(self, msg, tag="lane"):
        try:
            transform = self.tf_buffer.lookup_transform('odom', msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF error: {e}")
            return

        try:
            transformed_msg = do_transform_cloud(msg, transform)
        except Exception as e:
            self.get_logger().warn(f"PointCloud transform failed: {e}")
            return

        # Set robot's current pose as center of costmap
        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        self.origin_x = robot_x - (self.width * self.resolution) / 2
        self.origin_y = robot_y - (self.height * self.resolution) / 2

        # initialize costmap
        costmap = np.zeros((self.height, self.width), dtype=np.uint8)

        # Read transformed points
        points = point_cloud2.read_points(transformed_msg, field_names=["x", "y", "z"], skip_nans=True)

        for x, y, z in points:
            if tag in ["lane", "white", "yellow"] and z > 0.1:
                continue

            mx = int((x - self.origin_x) / self.resolution)
            my = int((y - self.origin_y) / self.resolution)

            if 0 <= mx < self.width and 0 <= my < self.height:
                if tag == "object":
                    costmap[my, mx] = 100
                elif tag == "white":
                    costmap[my, mx] = 250
                elif tag == "yellow":
                    costmap[my, mx] = 200

        # Apply OpenCV Gaussian blur (faster than scipy)
        binary = costmap.astype(np.float32)
        gradient = cv2.GaussianBlur(binary, (31, 31), sigmaX=9)

        if gradient.max() > 0: 
            if tag == "object":
                scaled = ((gradient / (gradient.max() ** 2.0)) * 100).astype(np.uint8)
            else:
                scaled = ((gradient / (gradient.max() ** 0.8)) * 100).astype(np.uint8)
        else:
            scaled = np.zeros_like(costmap)

        costmap = np.maximum(costmap, scaled)

        # Store updated layer
        if tag == "white":
            self.white_map = costmap
        elif tag == "yellow":
            self.yellow_map = costmap
        elif tag == "object":
            self.object_map = costmap

        # clears
        '''if tag == "white":
            self.white_map = None
        elif tag == "yellow":
            self.yellow_map = None
        elif tag == "object":
            self.object_map = None'''


    #function to construct and publish costmap message
    def publish_costmap(self, costmap, stamp):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        # Ensure origin values are floats and not None or NaN
        try:
            msg.info.origin.position.x = float(self.origin_x)
            msg.info.origin.position.y = float(self.origin_y)
        except (TypeError, ValueError):
            self.get_logger().error("Invalid origin_x or origin_y — skipping costmap publish")
            return

        msg.info.origin.orientation.w = 1.0

        msg.data = [int(v / 255 * 100) for v in costmap.flatten()]
        self.costmap_pub.publish(msg)

    #function to transform points into odom frame from some other frame
    # def transform_to_odom(self, x, y, z, frame_id='camera_link'):
    #     try:
    #         point = PointStamped()
    #         point.header.frame_id = frame_id
    #         point.header.stamp = self.get_clock().now().to_msg()
    #         point.point.x = x
    #         point.point.y = y
    #         point.point.z = z

    #         #checks for latest transforms
    #         if self.tf_buffer.can_transform('odom', frame_id, rclpy.time.Time()):
    #             transform = self.tf_buffer.lookup_transform(
    #                 'odom',
    #                 frame_id,
    #                 rclpy.time.Time(),
    #             )
    #         else:
    #             self.get_logger().warn(f"Transform from {frame_id} to odom not available")
    #             return None

    #         transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
    #         return (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)

    #     except Exception as e:
    #         self.get_logger().warn(f"Transform failed: {e}")
    #         return None
    


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
