#remove lidar from bot's body maybe at a certain angle behind bot
#optimise code after taking screenshotsof everypublishing maybe rmeove useless ones

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Vector3
import time
from builtin_interfaces.msg import Duration

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        
        # Parameters (tunable)
        self.ground_threshold = -0.2  # Z-axis value for ground removal
        self.voxel_size = 0.07  # Meters for voxel grid
        self.cluster_tolerance = 0.25 # clustering distance
        self.min_cluster_size = 20
        self.max_cluster_size = 5000
        self.sensor_height = 0.3  # Approx LiDAR mounting height
        
        # Publishers for each processing stage
        # SET IT TO BEST_EFFORT? - doesn't run on rviz2
        reliable_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        self.ground_removed_pub = self.create_publisher(PointCloud2, '/ground_removed', reliable_qos)
        self.downsampled_pub = self.create_publisher(PointCloud2, '/downsampled', reliable_qos)
        self.clusters_pub = self.create_publisher(PointCloud2, '/clusters', reliable_qos)
        self.bbox_pub = self.create_publisher(MarkerArray, '/bounding_boxes', reliable_qos)
        
        # Subscriber - use BEST_EFFORT for incoming LiDAR data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.sub = self.create_subscription(
            PointCloud2,
            '/scan/points',
            self.lidar_cb,
            sensor_qos
        )
        self.get_logger().info("Waiting for LiDAR data...")

    def lidar_cb(self, msg):
        start_time = time.time()
        try:
            # Convert to numpy array (only XYZ)
            pc = list(pc2.read_points(
                msg, 
                skip_nans=True, 
                field_names=("x", "y", "z")
            ))
            
            if not pc:
                self.get_logger().warn("Empty point cloud received")
                return
                
            # Convert to numpy array (N,3)
            arr = np.array([[p[0], p[1], p[2]] for p in pc], dtype=np.float32)
            
            # Filter out NaN and infinite values
            valid_mask = np.all(np.isfinite(arr), axis=1)
            arr = arr[valid_mask]
            
            if len(arr) == 0:
                self.get_logger().warn("No valid points after filtering out infinity vals")
                return
                
            self.get_logger().info(f"Valid points: {len(arr)}")

            # 1. Height-based ground removal
            ground_mask = arr[:, 2] < (self.sensor_height + self.ground_threshold)
            obstacles = arr[~ground_mask]
            self.get_logger().info(f"Ground points: {np.sum(ground_mask)}, Obstacles: {len(obstacles)}")
            self.publish_cloud(obstacles, self.ground_removed_pub, msg.header, "scan_link")
            
            # 2. Voxel downsampling
            if obstacles.size == 0:
                self.get_logger().warn("No obstacles after ground removal")
                return
                
            downsampled = self.voxel_downsample(obstacles, self.voxel_size)
            self.get_logger().info(f"Downsampled points: {len(downsampled)}")
            self.publish_cloud(downsampled, self.downsampled_pub, msg.header, "scan_link")
            
            # 3. Euclidean clustering with KDTree acceleration
            if len(downsampled) < self.min_cluster_size:
                self.get_logger().warn(f"Not enough points for clustering: {len(downsampled)} < {self.min_cluster_size}")
                return
                
            cluster_labels = self.euclidean_clustering(downsampled)
            unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
            self.get_logger().info(f"Found {len(unique_labels)} clusters")
            
            colored_clusters = self.colorize_clusters(downsampled, cluster_labels)
            self.publish_colored_cloud(colored_clusters, self.clusters_pub, msg.header, "scan_link")
            
            # 4. Bounding box calculation and visualization
            bboxes = self.calculate_bounding_boxes(downsampled, cluster_labels)
            self.publish_bounding_boxes(bboxes, msg.header, "scan_link")
            
            proc_time = (time.time() - start_time) * 1000
            self.get_logger().info(f"Processing time: {proc_time:.2f}ms | Clusters: {len(bboxes)}")
            
        except Exception as e:
            self.get_logger().error(f"Processing failed: {str(e)}")

    def voxel_downsample(self, points, voxel_size):
        """Efficient voxel downsampling using vectorized operations"""
        try:
            voxel_coords = np.floor(points / voxel_size).astype(np.int32)
            _, inverse, counts = np.unique(
                voxel_coords, axis=0, 
                return_inverse=True, return_counts=True
            )
            sum_points = np.zeros((counts.size, 3), dtype=np.float32)
            np.add.at(sum_points, inverse, points)
            return sum_points / counts[:, None]
        except Exception as e:
            self.get_logger().error(f"Voxel filtering failed: {str(e)}")
            return points  # Fallback to original points

    def euclidean_clustering(self, points):
        """KDTree-accelerated Euclidean clustering"""
        try:
            # Build KDTree for fast radius searches
            tree = KDTree(points)
            
            # DBSCAN with parameters matching Euclidean clustering
            clustering = DBSCAN(
                eps=self.cluster_tolerance,
                min_samples=self.min_cluster_size,
                metric='euclidean',
                algorithm='kd_tree',
                leaf_size=30,
                n_jobs=-1
            ).fit(points)
            
            # Filter clusters by size
            labels = clustering.labels_
            valid_mask = (labels >= 0)
            unique_labels, counts = np.unique(labels[valid_mask], return_counts=True)
            
            # Remove clusters outside size limits
            for label in unique_labels:
                if not (self.min_cluster_size <= counts[label] <= self.max_cluster_size):
                    labels[labels == label] = -1
                    
            return labels
        except Exception as e:
            self.get_logger().error(f"Clustering failed: {str(e)}")
            return np.zeros(len(points), dtype=int) - 1  # Return all as noise

    def colorize_clusters(self, points, labels):
        """Assign random colors to each cluster"""
        # Create array for RGBA colors (4 channels)
        colors = np.zeros((len(points), 4), dtype=np.uint8)
        colors[:, 3] = 255  # Alpha channel
        
        # Generate random colors for each valid cluster
        unique_labels = np.unique(labels[labels >= 0])
        color_map = np.random.randint(0, 255, (len(unique_labels), 3), dtype=np.uint8)
        
        for idx, label in enumerate(unique_labels):
            mask = (labels == label)
            colors[mask, :3] = color_map[idx]
            
        return np.hstack((points, colors))

    def calculate_bounding_boxes(self, points, labels):
        """Compute axis-aligned bounding boxes for each cluster"""
        bboxes = []
        unique_labels = np.unique(labels[labels >= 0])
        
        for label in unique_labels:
            cluster_points = points[labels == label]
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            center = (min_bounds + max_bounds) / 2
            dimensions = max_bounds - min_bounds
            bboxes.append((center, dimensions))
            
        return bboxes

    def publish_cloud(self, points, publisher, header, frame_id):
        """Publish XYZ point cloud"""
        if points.size == 0:
            self.get_logger().warn("Attempted to publish empty point cloud")
            return
            
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Create new header with correct frame ID
        new_header = Header()
        new_header.stamp = self.get_clock().now().to_msg()
        new_header.frame_id = frame_id
        
        cloud_msg = pc2.create_cloud(new_header, fields, points)
        publisher.publish(cloud_msg)
        self.get_logger().info(f"Published to {publisher.topic} with {len(points)} points")

    def publish_colored_cloud(self, points, publisher, header, frame_id):
        """Publish XYZRGB point cloud"""
        if points.size == 0:
            self.get_logger().warn("Attempted to publish empty colored point cloud")
            return
            
        # Extract XYZ and RGBA components
        xyz = points[:, :3]
        rgba = points[:, 3:].astype(np.uint8)
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
            PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
            PointField(name='a', offset=15, datatype=PointField.UINT8, count=1),
        ]
        
        # Create structured array for point cloud
        structured_data = np.zeros(len(xyz), dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('a', np.uint8)
        ])
        
        structured_data['x'] = xyz[:, 0]
        structured_data['y'] = xyz[:, 1]
        structured_data['z'] = xyz[:, 2]
        structured_data['r'] = rgba[:, 0]
        structured_data['g'] = rgba[:, 1]
        structured_data['b'] = rgba[:, 2]
        structured_data['a'] = rgba[:, 3]
        
        # Create new header with correct frame ID
        new_header = Header()
        new_header.stamp = self.get_clock().now().to_msg()
        new_header.frame_id = frame_id
        
        cloud_msg = pc2.create_cloud(new_header, fields, structured_data)
        publisher.publish(cloud_msg)
        self.get_logger().info(f"Published colored cloud to {publisher.topic}")

    def publish_bounding_boxes(self, bboxes, header, frame_id):
        """Publish bounding boxes as MarkerArray"""
        if not bboxes:
            self.get_logger().warn("No bounding boxes to publish")
            return
            
        marker_array = MarkerArray()
        
        for i, (center, dimensions) in enumerate(bboxes):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = frame_id  # Set correct frame ID
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=center[0], y=center[1], z=center[2])
            marker.scale = Vector3(x=dimensions[0], y=dimensions[1], z=dimensions[2])
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)

            marker.lifetime = Duration()
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 100000000  # 100 ms
            marker_array.markers.append(marker)
            
        self.bbox_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(bboxes)} bounding boxes")

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
