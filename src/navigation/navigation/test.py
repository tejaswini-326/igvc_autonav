import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np


class CostmapArrayListener(Node):
    def __init__(self):
        super().__init__('costmap_array_listener')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/costmap',
            self.costmap_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

    def costmap_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin.position

        self.get_logger().info(
            f"Received costmap: {width}x{height}, res={resolution}m, origin=({origin.x:.2f}, {origin.y:.2f})"
        )

        # Convert flat list to 2D NumPy array
        costmap_array = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Print the entire costmap array
        for row in costmap_array:
            for cost in row:
                if cost != 0:
                    print(cost, end=' ')


def main(args=None):
    rclpy.init(args=args)
    node = CostmapArrayListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
