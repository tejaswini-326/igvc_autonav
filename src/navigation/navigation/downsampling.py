#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import time

class FastPointCloudDownsamplerNode(Node):
    """
    Point cloud downsampler that preserves exact binary format
    """
    def __init__(self):
        super().__init__('fast_pointcloud_downsampler_node')
        
        # Parameters
        self.declare_parameter('downsample_factor', 8)  # Keep every 8th point
        self.declare_parameter('input_topic', '/camera/points')
        self.declare_parameter('output_topic', '/camera/points_downsampled')
        
        # Get parameters
        self.downsample_factor = self.get_parameter('downsample_factor').get_parameter_value().integer_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # Subscriber and Publisher
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10
        )
        
        self.get_logger().info(f"Fast Point Cloud Downsampler started:")
        self.get_logger().info(f"  Input topic: {self.input_topic}")
        self.get_logger().info(f"  Output topic: {self.output_topic}")
        self.get_logger().info(f"  Systematic downsample factor: {self.downsample_factor}")
        
        # Statistics
        self.msg_count = 0
        self.total_processing_time = 0.0
        
    def pointcloud_callback(self, msg):
        start_time = time.time()
        try:
            # Calculate point size and number of points
            point_step = msg.point_step
            row_step = msg.row_step
            num_points = len(msg.data) // point_step
            
            if num_points == 0:
                self.get_logger().warn("Received empty point cloud")
                return
            
            # Convert data to numpy array for easier manipulation
            data_array = np.frombuffer(msg.data, dtype=np.uint8)
            
            # Reshape to separate points
            points_data = data_array.reshape(num_points, point_step)
            
            # Systematic sampling (every nth point)
            downsampled_points = points_data[::self.downsample_factor]
            
            final_count = len(downsampled_points)
            
            # Create new message with same header and field structure
            downsampled_msg = PointCloud2()
            downsampled_msg.header = msg.header
            downsampled_msg.height = msg.height
            downsampled_msg.width = final_count if msg.height == 1 else final_count // msg.height
            downsampled_msg.fields = msg.fields
            downsampled_msg.is_bigendian = msg.is_bigendian
            downsampled_msg.point_step = msg.point_step
            downsampled_msg.row_step = final_count * msg.point_step if msg.height == 1 else downsampled_msg.width * msg.point_step
            downsampled_msg.data = downsampled_points.tobytes()
            downsampled_msg.is_dense = msg.is_dense
            
            # Publish downsampled cloud
            self.publisher.publish(downsampled_msg)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.msg_count += 1
            
            # Log statistics every 10 messages
            if self.msg_count % 10 == 0:
                avg_time = self.total_processing_time / self.msg_count
                reduction_ratio = final_count / num_points
                self.get_logger().info(
                    f"Processed {self.msg_count} clouds. "
                    f"Avg time: {avg_time:.3f}s. "
                    f"Last: {num_points} -> {final_count} points ({reduction_ratio:.2f} ratio, {processing_time:.3f}s)"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

class PointCloudDownsamplerNode(Node):
    """
    Simple version for comparison
    """
    def __init__(self):
        super().__init__('pointcloud_downsampler_node')
        
        # Parameters
        self.declare_parameter('downsample_factor', 4)
        self.declare_parameter('input_topic', '/camera/points')
        self.declare_parameter('output_topic', '/camera/points_downsampled')
        
        # Get parameters
        self.downsample_factor = self.get_parameter('downsample_factor').get_parameter_value().integer_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # Subscriber and Publisher
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10
        )
        
        self.get_logger().info(f"Point Cloud Downsampler started:")
        self.get_logger().info(f"  Input topic: {self.input_topic}")
        self.get_logger().info(f"  Output topic: {self.output_topic}")
        self.get_logger().info(f"  Downsample factor: {self.downsample_factor}")
        
        # Statistics
        self.msg_count = 0
        self.total_processing_time = 0.0
        
    def pointcloud_callback(self, msg):
        start_time = time.time()
        try:
            # Work directly with binary data
            point_step = msg.point_step
            num_points = len(msg.data) // point_step
            
            if num_points == 0:
                self.get_logger().warn("Received empty point cloud")
                return
            
            # Simple systematic downsampling in binary space
            downsampled_data = bytearray()
            for i in range(0, num_points, self.downsample_factor):
                start_idx = i * point_step
                end_idx = start_idx + point_step
                downsampled_data.extend(msg.data[start_idx:end_idx])
            
            final_count = len(downsampled_data) // point_step
            
            # Create new message preserving all original properties
            downsampled_msg = PointCloud2()
            downsampled_msg.header = msg.header
            downsampled_msg.height = msg.height
            downsampled_msg.width = final_count if msg.height == 1 else final_count // msg.height
            downsampled_msg.fields = msg.fields
            downsampled_msg.is_bigendian = msg.is_bigendian
            downsampled_msg.point_step = msg.point_step
            downsampled_msg.row_step = final_count * msg.point_step if msg.height == 1 else downsampled_msg.width * msg.point_step
            downsampled_msg.data = bytes(downsampled_data)
            downsampled_msg.is_dense = msg.is_dense
            
            self.publisher.publish(downsampled_msg)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.msg_count += 1
            
            if self.msg_count % 10 == 0:
                avg_time = self.total_processing_time / self.msg_count
                self.get_logger().info(
                    f"Processed {self.msg_count} clouds. "
                    f"Avg time: {avg_time:.3f}s. "
                    f"Last: {num_points} -> {final_count} points ({processing_time:.3f}s)"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    # Choose which node to run
    # node = PointCloudDownsamplerNode()  # Simple binary version
    node = FastPointCloudDownsamplerNode()  # Numpy version with systematic sampling
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()