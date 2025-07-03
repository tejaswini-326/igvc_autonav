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
import os 
import yaml

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

<<<<<<< HEAD
		self.costmap = None
		self.robot_pose = None
		self.goal_point = None
		self.costmap_grid = None
		
	def costmap_cb(self, msg):
		self.get_logger().info("received costmap")
		self.costmap = msg
		self.origin_x = msg.info.origin.position.x
		self.origin_y = msg.info.origin.position.y
		self.resolution = msg.info.resolution
		self.width = msg.info.width
		self.height = msg.info.height 
		data = np.array(msg.data, dtype=np.int8).reshape((self.height, self.width))
		self.costmap_grid = (data.astype(np.float32) / 100.0 * 255).astype(np.uint8)
	
=======
    def __init__(self):
        super().__init__("path_planner_node")
        self.costmap_sub = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_cb, 10)
        self.line_marker_sub = self.create_subscription(MarkerArray, '/lane_visualization', self.marker_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.goal_sub = self.create_subscription(Point, '/goal_point', self.goal_cb, 10)
        # self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.sm_path_pub = self.create_publisher(Path, '/sm_planned_path', 10)
        self.lane = 'right'

        self.param_file_path = os.path.expanduser('~/.config/config_igvc_ui/config.yaml')

>>>>>>> da0a8b8d9dff33ad4eb8f1a1ea9aed7af13c431a

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.costmap = None
        self.robot_pose = None
        self.goal_point = None
        self.goal_x = -1
        self.goal_y = -1
        self.grid_2d = None
        
    def costmap_cb(self, msg):
        self.get_logger().info("received costmap")
        self.costmap = msg
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.resolution = msg.info.resolution
        self.width = msg.info.width
        self.height = msg.info.height
        if(self.goal_y != -1 and self.goal_y != -1):
            # Convert flat costmap to 2D list
            self.grid_2d = np.array(self.costmap.data,dtype = np.int8).reshape((self.height,self.width)).tolist()
            self.robot_pose.x, self.robot_pose.y = self.base_link_to_costmap(self.robot_pose.x, self.robot_pose.y)
            self.a_star_search(self.grid_2d, [self.robot_pose.x, self.robot_pose.y], [self.goal_x, self.goal_y])


    def odom_cb(self, msg):
        self.robot_pose = msg.pose.pose.position

    def marker_cb(self, msg):
        if self.costmap is None:
            self.get_logger().warn("Costmap not received yet. Skipping marker callback.")
            return
        self.estimate_goal_from_markers(msg)

    def goal_cb(self, goal):
        if self.costmap is None:
            self.get_logger().warn("Costmap not received yet. Ignoring goal.")
            return

        result = self.base_link_to_costmap(goal.x, goal.y)
        if result is not None:
            self.goal_x, self.goal_y = result
            self.get_logger().info(f"Goal received and converted to grid: ({self.goal_x}, {self.goal_y})")
        else:
            self.get_logger().warn("Goal is outside map bounds")


    def estimate_goal_from_markers(self, marker_array):
        lane_markers = [m for m in marker_array.markers if m.ns == "lane_curves" and m.type == Marker.LINE_STRIP]

        if len(lane_markers) < 2:
            self.get_logger().warn("Not enough lane markers to estimate goal")
            return None

        sorted_markers = sorted(lane_markers, key=lambda m: m.points[0].y)

        if(self.lane == 'right'):
            right_marker = sorted_markers[0]
            right_points = right_marker.points
            mid_marker = sorted_markers[1]
            mid_points = mid_marker.points
            # Use last few points to compute average goal
            N = min(5, len(right_points), len(mid_points))

            # Average the last N points
            avg_rx = sum(p.x for p in right_points[-N:]) / N
            avg_ry = sum(p.y for p in right_points[-N:]) / N
        else:
            left_marker = sorted_markers[2]
            left_points = left_marker.points
            mid_marker = sorted_markers[1]
            mid_points = mid_marker.points
            N = min(5, len(left_points), len(mid_points))
            avg_lx = sum(p.x for p in left_points[-N:]) / N
            avg_ly = sum(p.y for p in left_points[-N:]) / N


        if N < 2:
            self.get_logger().warn("Too few points in lane markers")
            return None

        avg_mx = sum(p.x for p in mid_points[-N:]) / N
        avg_my = sum(p.y for p in mid_points[-N:]) / N

        # Midpoint
        if(self.lane == 'right'):
            mid_x = (avg_rx + avg_mx) / 2.0
            mid_y = (avg_ry + avg_my) / 2.0
            mid_z = 0.0
            # Transform to odom
            result = self.transform_to_odom(mid_x, mid_y, mid_z, frame_id=right_marker.header.frame_id)
        else:
            mid_x = (avg_lx + avg_mx) / 2.0
            mid_y = (avg_ly + avg_my) / 2.0
            mid_z = 0.0
            result = self.transform_to_odom(mid_x, mid_y, mid_z, frame_id=left_marker.header.frame_id)

        if result is None:
            self.get_logger().warn("Failed to transform midpoint to odom frame")
            return None

        x_odom, y_odom, z_odom = result

        # uncomment if goal point needs to be visualized
        # try:
        #     point_in_odom = PointStamped()
        #     point_in_odom.header.frame_id = 'odom'
        #     point_in_odom.header.stamp = self.get_clock().now().to_msg()
        #     point_in_odom.point.x = x_odom
        #     point_in_odom.point.y = y_odom
        #     point_in_odom.point.z = 0.0

        #     if self.tf_buffer.can_transform('base_footprint', 'odom', rclpy.time.Time()):
        #         transform = self.tf_buffer.lookup_transform(
        #             'base_footprint',
        #             'odom',
        #             rclpy.time.Time()
        #         )
        #         transformed_point = tf2_geometry_msgs.do_transform_point(point_in_odom, transform)

        #         goal_base = PoseStamped()
        #         goal_base.header.frame_id = 'base_footprint'
        #         goal_base.header.stamp = self.get_clock().now().to_msg()
        #         goal_base.pose.position = transformed_point.point
        #         goal_base.pose.orientation.w = 1.0

        #         # Publish transformed point in base_footprint frame
        #         self.goal_pub.publish(goal_base)
        #         self.get_logger().info(f"Published goal in base_footprint: x={goal_base.pose.position.x:.2f}, y={goal_base.pose.position.y:.2f}")

        #     else:
        #         self.get_logger().warn("Transform from odom to base_footprint not available")

        # except Exception as e:
        #     self.get_logger().warn(f"Transform to base_footprint failed: {e}")


        self.get_logger().info(f"Goal (odom): ({x_odom:.2f}, {y_odom:.2f})")
        map_coords = self.base_link_to_costmap(x_odom, y_odom)

        if map_coords is not None:
            self.goal_x, self.goal_y = map_coords
            print(f"Goal in map frame: cell_x={self.goal_x}, cell_y={self.goal_y}")
        else:
            print("Goal is outside the map bounds!")

    
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
        
    def base_link_to_costmap(self, x, y):
        mx = int(x / self.resolution)
        my = int(y / self.resolution)
        if 0 <= mx < self.width and 0 <= my < self.height:
            return (mx, my)
        return None

    def costmap_to_base_link(self, mx, my):
        x = mx * self.resolution + self.resolution / 2
        y = my * self.resolution + self.resolution / 2
        return (x, y)
    
    def is_valid(self, row, col):
        return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)
    
    def is_unblocked(self, grid, row, col):
        return grid[row][col] < 50

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
        # self.publish_path(path)
        smoothed = self.gradient_smooth(path)
        self.publish_sm_path(smoothed)

    def gradient_smooth(self, path, w_data=0.6, w_smooth=0.3, tolerance=1e-4):

        # uncomment if u want to read the parameters from yaml file
        # try:
        #     with open(self.param_file_path, 'r') as file:
        #         data = yaml.safe_load(file)
        #         costmap_config = data.get('parameters', {}).get('costmap', {})
        #         w_data = float(costmap_config.get('w_data', 0.6))
        #         w_smooth = float(costmap_config.get('w_smooth', 0.3))
        #         tolerance = float(costmap_config.get('tolerance', 1)) * 0.001
        #         print(f"w_data: {w_data}, w_smooth: {w_smooth}, tolerance: {tolerance}")
        #         if not (0 <= w_data <= 1 and 0 <= w_smooth <= 1):
        #             self.get_logger().warn("Invalid smoothing weights in YAML. Using defaults.")
        #             w_data, w_smooth, tolerance = 0.6, 0.3, 1e-4
        # except Exception as e:
        #     self.get_logger().warn(f"Error reading YAML: {e}")
        #     w_data = 0.6
        #     w_smooth = 0.3
        #     tolerance = 1e-4

        new_path = [list(p) for p in path]
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for j in range(2):  # x and y
                    old = new_path[i][j]
                    new_path[i][j] += w_data * (path[i][j] - new_path[i][j])
                    new_path[i][j] += w_smooth * (new_path[i - 1][j] + new_path[i + 1][j] - 2.0 * new_path[i][j])
                    change += abs(old - new_path[i][j])
        return [tuple(p) for p in new_path]
    
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

    # uncomment if u want to visualize raw path without smoothening
    # def publish_path(self, path_cells):
    #     path_msg = Path()
    #     path_msg.header.stamp = self.get_clock().now().to_msg()
    #     path_msg.header.frame_id = 'odom'  # Target frame

    #     for (mx, my) in path_cells:
    #         wx, wy = self.costmap_to_base_link(mx, my)

    #         # Convert world (map) point to odom frame
    #         world_pt = PointStamped()
    #         world_pt.header.frame_id = self.costmap.header.frame_id  # usually "map"
    #         world_pt.header.stamp = self.get_clock().now().to_msg()
    #         world_pt.point.x = wx
    #         world_pt.point.y = wy
    #         world_pt.point.z = 0.0

    #         try:
    #             # wait for transform to be available
    #             if not self.tf_buffer.can_transform('odom', world_pt.header.frame_id, rclpy.time.Time()):
    #                 self.get_logger().warn("Transform not available, skipping point")
    #                 continue

    #             odom_pt = tf2_geometry_msgs.do_transform_point(world_pt,
    #                 self.tf_buffer.lookup_transform('odom', world_pt.header.frame_id, rclpy.time.Time()))

    #             pose = PoseStamped()
    #             pose.header = path_msg.header
    #             pose.pose.position = odom_pt.point
    #             pose.pose.orientation.w = 1.0  # No orientation needed

    #             path_msg.poses.append(pose)

    #         except Exception as e:
    #             self.get_logger().warn(f"Transform error: {e}")
    #             continue

    #     self.path_pub.publish(path_msg)
    #     self.get_logger().info(f"Published path with {len(path_msg.poses)} poses.")


    def publish_sm_path(self, path_cells):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'base_link'  # Target frame

        for (mx, my) in path_cells:
            wx, wy = self.costmap_to_base_link(mx, my)

            # Convert world (map) point to odom frame
            world_pt = PointStamped()
            world_pt.header.frame_id = self.costmap.header.frame_id  # usually "map"
            world_pt.header.stamp = self.get_clock().now().to_msg()
            world_pt.point.x = wx
            world_pt.point.y = wy
            world_pt.point.z = 0.0

            try:
                # wait for transform to be available
                if not self.tf_buffer.can_transform('base_link', world_pt.header.frame_id, rclpy.time.Time()):
                    continue

                odom_pt = tf2_geometry_msgs.do_transform_point(world_pt,
                    self.tf_buffer.lookup_transform('base_link', world_pt.header.frame_id, rclpy.time.Time()))

                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position = odom_pt.point
                pose.pose.orientation.w = 1.0  # No orientation needed

                path_msg.poses.append(pose)

            except Exception as e:
                continue

        self.sm_path_pub.publish(path_msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()