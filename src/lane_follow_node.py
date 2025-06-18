# lane_follow_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
import sensor_msgs_py.point_cloud2 as pc2

class LaneFollowNode(Node):
    def __init__(self):
        super().__init__('lane_follow_node')
        self.sub = self.create_subscription(PointCloud2, '/igvc/lane_mask', self.pc_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Simple parameters
        self.robot_width = 0.6  # Robot width in meters (adjust for your robot)
        self.safety_margin = 0.2  # Extra space from lane lines
        
    def pc_callback(self, msg):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not points:
            return

        # Only look at points in front of the robot (x > 0) and within reasonable distance
        front_points = [p for p in points if 0.5 < p[0] < 3.0]
        
        if len(front_points) < 10:
            # Not enough points, move straight slowly
            cmd = Twist()
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        # Get y-coordinates (lateral positions)
        y_coords = [p[1] for p in front_points]
        
        # Find the closest white line on the left and right of the robot
        left_line = self.find_closest_line_on_left(y_coords)
        right_line = self.find_closest_line_on_right(y_coords)
        
        # Calculate steering based on lane boundaries
        cmd = Twist()
        cmd.linear.x = 0.4  # Forward speed
        
        if left_line is not None and right_line is not None:
            # We can see both lane boundaries
            lane_center = (left_line + right_line) / 2.0
            error = lane_center - 0.0  # Robot is at y=0
            cmd.angular.z = -error * 0.8  # Simple proportional control
            
            self.get_logger().info(f'Both lines: L={left_line:.2f}, R={right_line:.2f}, Center={lane_center:.2f}')
            
        elif left_line is not None:
            # Only see left boundary, stay at safe distance from it
            target_position = left_line + (self.robot_width/2 + self.safety_margin)
            error = target_position - 0.0
            cmd.angular.z = -error * 0.6
            
            self.get_logger().info(f'Left line only: {left_line:.2f}, target: {target_position:.2f}')
            
        elif right_line is not None:
            # Only see right boundary, stay at safe distance from it
            target_position = right_line - (self.robot_width/2 + self.safety_margin)
            error = target_position - 0.0
            cmd.angular.z = -error * 0.6
            
            self.get_logger().info(f'Right line only: {right_line:.2f}, target: {target_position:.2f}')
            
        else:
            # Can't see any clear boundaries, go straight
            cmd.angular.z = 0.0
            self.get_logger().warn('No clear lane boundaries detected')
        
        # Limit angular velocity for safety
        cmd.angular.z = max(-0.5, min(0.5, cmd.angular.z))
        
        self.pub.publish(cmd)

    def find_closest_line_on_left(self, y_coords):
        """Find the closest white line to the left of the robot (negative y)"""
        left_points = [y for y in y_coords if y < -0.1]  # Points to the left
        if not left_points:
            return None
        
        # Group nearby points and find the rightmost (closest to robot) cluster
        left_points.sort(reverse=True)  # Sort from closest to farthest
        
        # Simple clustering: find the largest group of points close together
        clusters = []
        current_cluster = [left_points[0]]
        
        for i in range(1, len(left_points)):
            if abs(left_points[i] - left_points[i-1]) < 0.3:  # Points within 30cm
                current_cluster.append(left_points[i])
            else:
                if len(current_cluster) >= 5:  # At least 5 points for a valid line
                    clusters.append(current_cluster)
                current_cluster = [left_points[i]]
        
        if len(current_cluster) >= 5:
            clusters.append(current_cluster)
        
        if not clusters:
            return None
        
        # Return the average position of the closest (rightmost) cluster
        closest_cluster = clusters[0]  # First cluster is closest due to sorting
        return sum(closest_cluster) / len(closest_cluster)
    
    def find_closest_line_on_right(self, y_coords):
        """Find the closest white line to the right of the robot (positive y)"""
        right_points = [y for y in y_coords if y > 0.1]  # Points to the right
        if not right_points:
            return None
        
        # Group nearby points and find the leftmost (closest to robot) cluster
        right_points.sort()  # Sort from closest to farthest
        
        # Simple clustering
        clusters = []
        current_cluster = [right_points[0]]
        
        for i in range(1, len(right_points)):
            if abs(right_points[i] - right_points[i-1]) < 0.3:  # Points within 30cm
                current_cluster.append(right_points[i])
            else:
                if len(current_cluster) >= 5:  # At least 5 points for a valid line
                    clusters.append(current_cluster)
                current_cluster = [right_points[i]]
        
        if len(current_cluster) >= 5:
            clusters.append(current_cluster)
        
        if not clusters:
            return None
        
        # Return the average position of the closest (leftmost) cluster
        closest_cluster = clusters[0]  # First cluster is closest due to sorting
        return sum(closest_cluster) / len(closest_cluster)

def main():
    rclpy.init()
    node = LaneFollowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()