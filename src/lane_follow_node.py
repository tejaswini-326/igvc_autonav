#!/usr/bin/env python3
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
        
    def pc_callback(self, msg):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not points:
            return

        # Only look at points in front of the robot
        front_points = [p for p in points if 0.5 < p[0] < 2.5]
        
        if len(front_points) < 5:
            # Not enough points, stop
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        # Get y-coordinates (lateral positions)
        y_coords = [p[1] for p in front_points]
        
        # Find all distinct line positions
        line_positions = self.find_line_positions(y_coords)
        
        if len(line_positions) < 2:
            # Need at least 2 lines to define a lane
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            self.get_logger().warn(f'Only found {len(line_positions)} lines')
            return
        
        # Find the lane we should be in (the one closest to robot center)
        lane_center = self.find_current_lane_center(line_positions)
        
        if lane_center is not None:
            # Steer towards lane center
            error = lane_center - 0.0  # Robot is at y=0
            
            cmd = Twist()
            cmd.linear.x = 0.4
            cmd.angular.z = -error * 1.0  # Simple proportional control
            
            # Limit turning
            cmd.angular.z = max(-0.6, min(0.6, cmd.angular.z))
            
            self.get_logger().info(f'Lines at: {line_positions}, Lane center: {lane_center:.2f}, Error: {error:.2f}')
        else:
            # Fallback - go straight
            cmd = Twist()
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            self.get_logger().warn('Could not determine lane center')
            
        self.pub.publish(cmd)

    def find_line_positions(self, y_coords):
        """Find positions of white lines by clustering y-coordinates"""
        if len(y_coords) < 3:
            return []
        
        # Sort coordinates
        y_sorted = sorted(y_coords)
        
        # Group points that are close together (within 20cm)
        groups = []
        current_group = [y_sorted[0]]
        
        for i in range(1, len(y_sorted)):
            if abs(y_sorted[i] - y_sorted[i-1]) < 1:  # 20cm threshold
                current_group.append(y_sorted[i])
            else:
                if len(current_group) >= 3:  # Need at least 3 points for a line
                    groups.append(current_group)
                current_group = [y_sorted[i]]
        
        # Don't forget the last group
        if len(current_group) >= 3:
            groups.append(current_group)
        
        # Calculate average position of each group (line)
        line_positions = []
        for group in groups:
            avg_pos = sum(group) / len(group)
            line_positions.append(avg_pos)
        
        return sorted(line_positions)
    
    def find_current_lane_center(self, line_positions):
        """Find which lane the robot should be in and return its center"""
        if len(line_positions) < 2:
            return None
        
        # Find the pair of lines that the robot is currently between
        # or should be between (closest to robot's current position y=0)
        
        best_lane_center = None
        min_distance_to_robot = float('inf')
        
        # Check all possible lane pairs
        for i in range(len(line_positions) - 1):
            left_line = line_positions[i]
            right_line = line_positions[i + 1]
            
            lane_center = (left_line + right_line) / 2.0
            distance_to_robot = abs(lane_center - 0.0)
            
            # Check if robot is between these lines or very close
            if (left_line <= 0.0 <= right_line) or distance_to_robot < min_distance_to_robot:
                min_distance_to_robot = distance_to_robot
                best_lane_center = lane_center
        
        return best_lane_center

def main():
    rclpy.init()
    node = LaneFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()