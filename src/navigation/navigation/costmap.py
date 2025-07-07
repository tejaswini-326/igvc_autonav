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
        self.resolution = 0.067  #meters per cell
        self.width = 300
        self.height = 300
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
        self.empty_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        #internal maps
        self.white_map = self.empty_layer
        self.yellow_map = self.empty_layer
        self.object_map = self.empty_layer
        self.latest_white_pc = None
        self.latest_yellow_pc = None
        self.latest_object_pc = None
        self.new_white = False
        self.new_yellow = False
        self.new_object = False


    #callback functions for point cloud data
    def object_pc_callback(self, msg):
        self.latest_object_pc = msg
        self.new_object = True

    def white_lane_pc_callback(self, msg):
        self.latest_white_pc = msg
        self.new_white = True

    def yellow_lane_pc_callback(self, msg):
        self.latest_yellow_pc = msg
        self.new_yellow = True

    def timer_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform('odom', 'camera_link', rclpy.time.Time())
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
            self.origin_x = self.robot_x - (self.width * self.resolution) / 2
            self.origin_y = self.robot_y - (self.height * self.resolution) / 2
            self.cached_tranform = transform
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed in timer: {e}")
            return

        if self.new_white and self.latest_white_pc:
            self.white_map = self.generate_costmap(self.latest_white_pc, "white")
            self.new_white = False

        if self.new_yellow and self.latest_yellow_pc:
            self.yellow_map = self.generate_costmap(self.latest_yellow_pc, "yellow")
            self.new_yellow = False

        if self.new_object and self.latest_object_pc:
            self.object_map = self.generate_costmap(self.latest_object_pc, "object")
            self.new_object = False

        # Combine and publish
        combined = np.maximum.reduce([self.white_map, self.yellow_map, self.object_map])
        self.publish_costmap(combined, self.get_clock().now().to_msg())

    def generate_costmap(self, msg, tag="lane"):
        try:
            transform = self.cached_tranform
        except Exception as e:
            self.get_logger().warn(f"TF error: {e}")
            return self.empty_layer

        try:
            transformed_msg = do_transform_cloud(msg, transform)
        except Exception as e:
            self.get_logger().warn(f"PointCloud transform failed: {e}")
            return self.empty_layer

        costmap = np.zeros((self.height, self.width), dtype=np.uint8)

        gen = point_cloud2.read_points(transformed_msg, field_names=["x", "y"], skip_nans=True)
        points = np.fromiter(gen, dtype=np.dtype([('x', np.float32), ('y', np.float32)]))
        points = np.stack((points['x'], points['y']), axis=-1)

        if points.shape[0] == 0:
            return self.empty_layer

        mx_raw = (points[:, 0] - self.origin_x) / self.resolution
        my_raw = (points[:, 1] - self.origin_y) / self.resolution
        valid = (
            (mx_raw >= 0) & (mx_raw < self.width) &
            (my_raw >= 0) & (my_raw < self.height) &
            (mx_raw < 100)  # skip bottom (back) part of map
        )
        mx = mx_raw[valid].astype(int)
        my = my_raw[valid].astype(int)


        if tag == "object":
            value = 100
        elif tag == "white":
            value = 250
        elif tag == "yellow":
            value = 150
        else:
            value = 0

        mask = costmap[my, mx] < 255
        costmap[my[mask], mx[mask]] = value

        if not np.any(costmap):
            return self.empty_layer

        binary = costmap.astype(np.float32)
        gradient = cv2.GaussianBlur(binary, (15, 15), sigmaX=2.3)

        grad_max = gradient.max()
        if grad_max > 0:
            power = 2.0 if tag == "object" else 0.8
            scaled = ((gradient / (grad_max ** power)) * 100).astype(np.uint8)
        else:
            scaled = np.zeros_like(costmap)

        return np.maximum(costmap, scaled)

    
        # Store updated layer
        '''if tag == "white":
            self.white_map = costmap
        elif tag == "yellow":
            self.yellow_map = costmap
        elif tag == "object":
            self.object_map = costmap'''

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


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()