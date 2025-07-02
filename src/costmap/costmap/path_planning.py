import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import math

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
from std_msgs.msg import Header

import cv2

# Define the size of the grid
ROW = 500
COL = 500
class PathPlanner(Node):
    class Cell:
        def __init__(self):
            self.parent_i = 0  # Parent cell's row index
            self.parent_j = 0  # Parent cell's column index
            self.f = float('inf')  # Total cost of the cell (g + h)
            self.g = float('inf')  # Cost from start to this cell
            self.h = 0  # Heuristic cost from this cell to destination

    def __init__(self):
        super().__init__("path_planner_node")
        self.costmap_sub = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)
        self.line_marker_sub = self.create_subscription(MarkerArray, '/lane_visualization', self.marker_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_point', 10)


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.costmap = None
        self.robot_pose = None
        self.goal_point = None
        self.mx = -1
        self.my = -1
        self.grid_2d = None
        
    def costmap_cb(self, msg):
        self.get_logger().info("received costmap")
        self.costmap = msg
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.resolution = msg.info.resolution
        self.width = msg.info.width
        self.height = msg.info.height
        if(self.my != -1 and self.my != -1):
            # Convert flat costmap to 2D list
            self.grid_2d = np.array(self.costmap.data,dtype = np.int8).reshape((self.height,self.width)).tolist()

            # self.a_star_search(self.grid_2d, [0, 0], [self.mx, self.my])
            # self.a_star_search(self.grid_2d, [250, 250], [300, 350])
            self.a_star_search(self.grid_2d, [250, 250], [self.mx, self.my])


    def odom_cb(self, msg):
        self.robot_pose = msg.pose.pose

    def marker_cb(self, msg):
        if self.costmap is None:
            self.get_logger().warn("Costmap not received yet. Skipping marker callback.")
            return
        self.goal_point = self.estimate_goal_from_markers(msg)


    def estimate_goal_from_markers(self, marker_array):
        lane_markers = [m for m in marker_array.markers if m.ns == "lane_curves" and m.type == Marker.LINE_STRIP]

        if len(lane_markers) < 2:
            self.get_logger().warn("Not enough lane markers to estimate goal")
            return None

        sorted_markers = sorted(lane_markers, key=lambda m: m.points[0].y)
        right_marker = sorted_markers[0]
        left_marker = sorted_markers[1]

        right_points = right_marker.points
        left_points = left_marker.points

        # Use last few points to compute average goal
        N = min(5, len(right_points), len(left_points))
        if N < 2:
            self.get_logger().warn("Too few points in lane markers")
            return None
        
        # Average the last N points
        avg_rx = sum(p.x for p in right_points[-N:]) / N
        avg_ry = sum(p.y for p in right_points[-N:]) / N

        avg_lx = sum(p.x for p in left_points[-N:]) / N
        avg_ly = sum(p.y for p in left_points[-N:]) / N

        # Midpoint
        mid_x = (avg_rx + avg_lx) / 2.0
        mid_y = (avg_ry + avg_ly) / 2.0
        mid_z = 0.0

        # Transform to odom
        result = self.transform_to_odom(mid_x, mid_y, mid_z, frame_id=right_marker.header.frame_id)
        if result is None:
            self.get_logger().warn("Failed to transform midpoint to odom frame")
            return None

        x_odom, y_odom, z_odom = result

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'odom'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x_odom
        goal_pose.pose.position.y = y_odom
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        # Transform to base_footprint frame
        try:
            point_in_odom = PointStamped()
            point_in_odom.header.frame_id = 'odom'
            point_in_odom.header.stamp = self.get_clock().now().to_msg()
            point_in_odom.point.x = x_odom
            point_in_odom.point.y = y_odom
            point_in_odom.point.z = 0.0

            if self.tf_buffer.can_transform('base_footprint', 'odom', rclpy.time.Time()):
                transform = self.tf_buffer.lookup_transform(
                    'base_footprint',
                    'odom',
                    rclpy.time.Time()
                )
                transformed_point = tf2_geometry_msgs.do_transform_point(point_in_odom, transform)

                goal_base = PoseStamped()
                goal_base.header.frame_id = 'base_footprint'
                goal_base.header.stamp = self.get_clock().now().to_msg()
                goal_base.pose.position = transformed_point.point
                goal_base.pose.orientation.w = 1.0

                # Publish transformed point in base_footprint frame
                self.goal_pub.publish(goal_base)
                self.get_logger().info(f"Published goal in base_footprint: x={goal_base.pose.position.x:.2f}, y={goal_base.pose.position.y:.2f}")

            else:
                self.get_logger().warn("Transform from odom to base_footprint not available")

        except Exception as e:
            self.get_logger().warn(f"Transform to base_footprint failed: {e}")


        self.get_logger().info(f"Goal (odom): ({x_odom:.2f}, {y_odom:.2f})")
        map_coords = self.world_to_map(x_odom, y_odom)

        if map_coords is not None:
            self.mx, self.my = map_coords
            print(f"Goal in map frame: cell_x={self.mx}, cell_y={self.my}")
        else:
            print("Goal is outside the map bounds!")
        return goal_pose

    
    def transform_to_odom(self, x, y, z, frame_id='camera_link'):
        try:
            point = PointStamped()
            point.header.frame_id = frame_id
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x = x
            point.point.y = y
            point.point.z = z

            if self.tf_buffer.can_transform('odom', frame_id, rclpy.time.Time()):
                transform = self.tf_buffer.lookup_transform(
                    'odom',
                    frame_id,
                    rclpy.time.Time(),
                )
            else:
                self.get_logger().warn(f"Transform from {frame_id} to odom not available")
                return None

            transformed_point = tf2_geometry_msgs.do_transform_point(point, transform)
            return (transformed_point.point.x, transformed_point.point.y, transformed_point.point.z)

        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None
        
    def world_to_map(self, x, y):
        mx = int((x - self.origin_x) / self.resolution)
        my = int((y - self.origin_y) / self.resolution)
        if 0 <= mx < self.width and 0 <= my < self.height:
            return (mx, my)
        return None

    def map_to_world(self, mx, my):
        x = mx * self.resolution + self.origin_x + self.resolution / 2
        y = my * self.resolution + self.origin_y + self.resolution / 2
        return (x, y)
    
    def is_valid(self, row, col):
        return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)
    
    def is_unblocked(self, grid, row, col):
        return grid[row][col] == 0

    # Check if a cell is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]

    # Calculate the heuristic value of a cell (Euclidean distance to destination)
    def calculate_h_value(self, row, col, dest):
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5
        # return 0
    
    def trace_path(self, cell_details, dest):
        print("The Path is ")
        path = []
        row = dest[0]
        col = dest[1]

        # Trace the path from destination to source using parent cells
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append((row, col))
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row = temp_row
            col = temp_col

        # Add the source cell to the path
        path.append((row, col))
        # Reverse the path to get the path from source to destination
        path.reverse()

        # Print the path
        for i in path:
            print("->", i, end=" ")
            
        print()
        np_grid_2d = np.array(self.grid_2d).reshape((self.height, self.width)) 

        self.visualize_grid_and_path(np_grid_2d, path)
        self.publish_path(path)

    
    def a_star_search(self, grid, src, dest):
        # Check if the source and destination are valid
        if not self.is_valid(src[0], src[1]) or not self.is_valid(dest[0], dest[1]):
            print("Source or destination is invalid")
            return

        # Check if the source and destination are unblocked
        if not self.is_unblocked(grid, src[0], src[1]) or not self.is_unblocked(grid, dest[0], dest[1]):
            print("Source or the destination is blocked")
            return

        # Check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return

        # Initialize the closed list (visited cells)
        closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
        # Initialize the details of each cell
        cell_details = [[self.Cell() for _ in range(COL)] for _ in range(ROW)]

        # Initialize the start cell details
        i = src[0]
        j = src[1]
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j

        # Initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)

            # Mark the cell as visited
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]

                # If the successor is valid, unblocked, and not visited
                if self.is_valid(new_i, new_j) and self.is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                    # If the successor is the destination
                    if self.is_destination(new_i, new_j, dest):
                        # Set the parent of the destination cell
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        # Trace and print the path from source to destination
                        self.trace_path(cell_details, dest)
                        found_dest = True
                        return
                    else:
                        # Calculate the new f, g, and h values
                        #
                        dx = new_i - i
                        dy = new_j - j
                        step_cost = math.hypot(dx, dy)  # 1.0 or √2

                        g_new = cell_details[i][j].g + step_cost
                        #

                        # g_new = cell_details[i][j].g + 1.0
                        h_new = self.calculate_h_value(new_i, new_j, dest)
                        weight = 0.85
                        f_new = g_new + h_new * weight

                        # If the cell is not in the open list or the new f value is smaller
                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            # Update the cell details
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all cells
        if not found_dest:
            print("Failed to find the destination cell")

    def publish_path(self, path_cells):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'  # Target frame

        for (mx, my) in path_cells:
            wx, wy = self.map_to_world(mx, my)

            # Convert world (map) point to odom frame
            world_pt = PointStamped()
            world_pt.header.frame_id = self.costmap.header.frame_id  # usually "map"
            world_pt.header.stamp = self.get_clock().now().to_msg()
            world_pt.point.x = wx
            world_pt.point.y = wy
            world_pt.point.z = 0.0

            try:
                # wait for transform to be available
                if not self.tf_buffer.can_transform('odom', world_pt.header.frame_id, rclpy.time.Time()):
                    self.get_logger().warn("Transform not available, skipping point")
                    continue

                odom_pt = tf2_geometry_msgs.do_transform_point(world_pt,
                    self.tf_buffer.lookup_transform('odom', world_pt.header.frame_id, rclpy.time.Time()))

                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position = odom_pt.point
                pose.pose.orientation.w = 1.0  # No orientation needed

                path_msg.poses.append(pose)

            except Exception as e:
                self.get_logger().warn(f"Transform error: {e}")
                continue

        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(path_msg.poses)} poses.")


    def visualize_grid_and_path(self, grid_2d, path):
        # Create a blank image (black = 0)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the costmap: white for free, gray for unknown, black for occupied
        for i in range(self.height):
            for j in range(self.width):
                value = grid_2d[i][j]
                if value == 0:
                    img[i, j] = (255, 255, 255)  # Free = white
                elif value == 100:
                    img[i, j] = (0, 0, 0)        # Occupied = black
                else:
                    img[i, j] = (127, 127, 127)  # Unknown = gray

        # Draw the path in blue
        for (i, j) in path:
            img[i, j] = (255, 0, 0)  # Blue

        # Flip vertically so (0,0) is bottom-left
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 1)
        # Resize for better visibility
        scale = 1
        img = cv2.resize(img, (self.width * scale, self.height * scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Costmap with Path", img)
        cv2.waitKey(1)
    
def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()