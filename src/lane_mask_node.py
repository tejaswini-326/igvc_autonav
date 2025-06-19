#!/usr/bin/env python3
# lane_mask_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from sensor_msgs.msg import PointField
import struct

class LaneMaskNode(Node):
    def __init__(self):
        super().__init__('lane_mask_node')
        # Subscribe to camera/points topic
        self.sub = self.create_subscription(PointCloud2, '/camera/points', self.pc_callback, 10)
        # Publish to the topic that lane_follow_node expects
        self.pub = self.create_publisher(PointCloud2, '/igvc/lane_mask', 10)
        
        # Color thresholds for white and greyish-white detection
        self.white_threshold = 80     # Pure white threshold
        self.grey_white_threshold = 100  # Greyish white threshold
        self.color_difference_threshold = 50  # Max difference between R, G, B for grey/white

    def pc_callback(self, msg):
        points = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        lane_points = []

        for p in points:
            # Extract RGB values from the packed float
            rgb = struct.unpack('I', struct.pack('f', p[3]))[0]
            r = (rgb >> 16) & 0xFF
            g = (rgb >> 8) & 0xFF
            b = rgb & 0xFF

            # Check if point is white or greyish-white
            if self.is_white_or_grey_white(r, g, b):
                lane_points.append([p[0], p[1], p[2]])

        # Only publish if we found lane points
        if lane_points:
            # Create new point cloud with only lane points
            header = Header()
            header.stamp = msg.header.stamp
            header.frame_id = msg.header.frame_id
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            
            new_pc = pc2.create_cloud(header, fields, lane_points)
            self.pub.publish(new_pc)
            
            self.get_logger().info(f'Published {len(lane_points)} lane points')
        else:
            self.get_logger().debug('No lane points detected')

    def is_white_or_grey_white(self, r, g, b):
        """
        Check if a color is white or greyish-white
        """
        # Check for pure white (all values high)
        if r > self.white_threshold and g > self.white_threshold and b > self.white_threshold:
            return True
        
        # Check for greyish-white (all values reasonably high and similar to each other)
        min_val = min(r, g, b)
        max_val = max(r, g, b)
        
        # All components should be above grey threshold
        if min_val > self.grey_white_threshold:
            # Components should be similar (not too much color variation)
            if (max_val - min_val) < self.color_difference_threshold:
                return True
        
        return False

def main():
    rclpy.init()
    node = LaneMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()