import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import time
import math

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
import os 
import yaml
import cv2

# Define the size of the grid
WIDTH = 300
HEIGHT = 300

DIRS = ((0, 1),  (0,-1),  (1, 0), (-1, 0),
        (1, 1),  (1,-1), (-1, 1), (-1,-1))

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
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_point', self.goal_cb, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.sm_path_pub = self.create_publisher(Path, '/sm_planned_path', 10)
        self.debug_pub = self.create_publisher(MarkerArray, '/astar_debug', 10)

        # self.param_file_path = os.path.expanduser('~/.config/config_igvc_ui/config.yaml')

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

        self.grid = np.frombuffer(msg.data,
                                dtype=np.uint8,      # 0-255 stays 0-255
                                count=msg.info.width * msg.info.height
                                ).reshape(msg.info.height, msg.info.width)
        # self.robot_pose.x, self.robot_pose.y = self.odom_to_costmap(self.robot_pose.x, self.robot_pose.y)
        # self.a_star_search(self.grid_2d, [self.robot_pose.x, self.robot_pose.y], [self.goal_x, self.goal_y])
        # self.a_star_search(self.grid_2d, [self.robot_pose.x, self.robot_pose.y], [350,250])


    def odom_cb(self, msg):
        self.robot_pose = msg.pose.pose.position

    def goal_cb(self, goal:PoseStamped):

        if self.costmap is None:
            self.get_logger().warn("Costmap not received yet. Ignoring goal.")
            return

        result = self.odom_to_costmap(goal.pose.position.x, goal.pose.position.y)
        if result is not None:
            self.goal_x, self.goal_y = result
            self.get_logger().info(f"Goal received and converted to grid: ({self.goal_x}, {self.goal_y})")
        else:
            self.get_logger().warn("Goal is outside map bounds")

        if(self.goal_x != -1 and self.goal_y != -1):
            # self.robot_pose.x, self.robot_pose.y = self.odom_to_costmap(self.robot_pose.x, self.robot_pose.y)
            # self.a_star_search(self.grid_2d, [self.robot_pose.x, self.robot_pose.y], [self.goal_x, self.goal_y])
            t0 = time.perf_counter()
            self.a_star_search(self.grid_2d, [150, 150], [self.goal_x, self.goal_y])
            self.get_logger().info(f"Astar total took: {(time.perf_counter()-t0)*1000} ms")
            # self.a_star_search(self.grid_2d, [self.robot_pose.x, self.robot_pose.y], [350,250])

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
        
    def odom_to_costmap(self, x_world: float, y_world: float):
        """
        Convert odom/world (x,y) to grid (row, col).  
        Row → y-index,  Col → x-index.
        """
        col = int((x_world - self.origin_x) / self.resolution)   # x ➜ col
        row = int((y_world - self.origin_y) / self.resolution)   # y ➜ row

        if 0 <= row < HEIGHT and 0 <= col < WIDTH:
            return (row, col)
        return None


    def costmap_to_odom(self, row: int, col: int):
        """
        Convert grid (row, col) back to odom/world (x,y).
        """
        x_world = col * self.resolution + self.origin_x + self.resolution / 2
        y_world = row * self.resolution + self.origin_y + self.resolution / 2
        return (x_world, y_world)
    
    def is_valid(self, row, col):
        return 0 <= row < HEIGHT and 0 <= col < WIDTH


    def is_unblocked(self, grid, row, col):
        idx = row * self.width + col
        return grid[idx] < 70
    
    # Check if a cell is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]

    # Calculate the heuristic value of a cell (Euclidean distance to destination)
    def calculate_h_value(self, row, col, dest):
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5
        # return 0


    def _trace_npy(self, pi, pj, dest):
        """Back-trace using the NumPy parent arrays."""
        path = []
        ci, cj = dest
        while True:
            path.append((ci, cj))
            ni, nj = pi[ci, cj], pj[ci, cj]
            if (ni == ci) and (nj == cj):
                break
            ci, cj = int(ni), int(nj)
        path.reverse()
        self.get_logger().info(f"Number of points in the path = {len(path)}")
        self.publish_debug_markers(path, None)         # cost labels skipped
        smoothed = self.gradient_smooth(path)
        self.publish_sm_path(smoothed)


    def trace_path(self, cell_details, dest):
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

        self.get_logger().info(f"Number of points in the path = {len(path)}")

        # # Print the path
        # print("\nPath with Costs (row, col): f, g, h")
        # for i in path:
        #     print("->", i, end=" ")
        #     print(f"Cell {i} cost={self.grid_2d[i[0]*WIDTH + i[1]]}")

        #     index = i[0] * WIDTH + i[1]
        #     if 0 <= index < len(self.grid_2d):
        #         cost = self.grid_2d[index]
        #     else:
        #         cost = -999  # Invalid
        #     print(f"{i}: cost={cost}")
            
        # print()
        # self.publish_path(path)
        self.publish_debug_markers(list(path), cell_details) 

        smoothed = self.gradient_smooth(path)  
        self.publish_sm_path(smoothed)

    def gradient_smooth(self, path, w_data=0.009, w_smooth=0.4, tolerance=1e-4):

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
        #

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
    
    def publish_debug_markers(self, path_cells, cell_details, ns='astar', lifetime=0.0):
        """
        Publish the A* (or smoothed) path and per-cell cost as RViz markers.
        Each call first clears previous markers (DELETEALL) and then adds:
        • one LINE_STRIP for the path
        • one TEXT_VIEW_FACING per waypoint that shows the raw cost value
        """
        if self.costmap is None:
            self.get_logger().warn('Costmap not ready – cannot publish debug markers')
            return

        ma = MarkerArray()

        # ---------- 0. clear old markers ----------
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        # Helper to stamp markers
        def fresh_marker(mtype, mid):
            m = Marker()
            m.header.frame_id = 'odom'               # same frame you used for Path
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.lifetime.sec = int(lifetime)
            m.scale.x = 0.03                         # default thickness / font height
            m.scale.y = 0.03
            m.scale.z = 0.03
            m.color.a = 1.0
            # white (r=g=b=1) – tweak if you like
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            return m

        # ---------- 1. LINE_STRIP for the path ----------
        line = fresh_marker(Marker.LINE_STRIP, 0)
        line.scale.x = 0.02                         # line width
        for mx, my in path_cells:
            wx, wy = self.costmap_to_odom(mx, my)
            pt = Point(x=wx, y=wy, z=0.02)          # slight z-offset
            line.points.append(pt)
        ma.markers.append(line)

        # ---------- 2. cost labels ----------
        if cell_details is not None:
            for idx, (mx, my) in enumerate(path_cells, start=1):
                mx_i, my_i = int(mx), int(my)
                wx, wy = self.costmap_to_odom(mx_i, my_i)

                cell = cell_details[mx_i][my_i]
                txt = fresh_marker(Marker.TEXT_VIEW_FACING, idx)
                txt.pose.position.x = wx
                txt.pose.position.y = wy
                txt.pose.position.z = 0.05

                # Show f, g, h
                txt.text = f"f={cell.f:.1f} g={cell.g:.1f} h={cell.h:.1f}"
                txt.scale.z = 0.05
                txt.color.r = txt.color.g = txt.color.b = 0.0  # black
                ma.markers.append(txt)

        # ---------- 3. publish ----------
        self.debug_pub.publish(ma)

    
    def a_star_search(self, grid, src, dest):
        """
        NumPy-based A*     ≈ 6–10× faster than the object version.
        Keeps the same interface so the rest of the node works unchanged.
        """
        # ---------- quick rejects ----------
        if not (0 <= src[0] < HEIGHT and 0 <= src[1] < WIDTH):
            self.get_logger().warn("Source outside grid")
            return
        if not (0 <= dest[0] < HEIGHT and 0 <= dest[1] < WIDTH):
            self.get_logger().warn("Destination outside grid")
            return
        if src == dest:
            self.get_logger().info("Already at destination")
            return

        t_total = time.perf_counter()

        # ---------- pre-allocate NumPy slabs ----------
        f = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
        g = np.full_like(f, np.inf)
        h = np.zeros_like(f)
        parent_i = np.full(f.shape, -1, dtype=np.int16)
        parent_j = np.full_like(parent_i, -1, dtype=np.int16)
        closed = np.zeros(f.shape, dtype=bool)

        # ---------- initial cell ----------
        si, sj = src
        f[si, sj] = 0.0
        g[si, sj] = 0.0
        parent_i[si, sj] = si
        parent_j[si, sj] = sj

        open_list = []
        heapq.heappush(open_list, (0.0, si, sj))

        cost_sqrt2 = math.sqrt(2.0)
        dest_i, dest_j = dest
        weight = 30                              # heuristic weight

        self.get_logger().info(f"Init part: {(time.perf_counter()-t_total)*1000:.2f} ms")
        t_loop = time.perf_counter()

        # ---------- main loop ----------
        counter_of_points = 0
        while open_list:
            f_curr, i, j = heapq.heappop(open_list)
            counter_of_points += 1
            if closed[i, j]:
                continue
            closed[i, j] = True

            if (i == dest_i) and (j == dest_j):    # reached goal
                self._trace_npy(parent_i, parent_j, dest)
                self.get_logger().info(f"A* core: {(time.perf_counter()-t_loop)*1000:.2f} ms")
                self.get_logger().info(f"A* total: {(time.perf_counter()-t_total)*1000:.2f} ms")
                self.get_logger().info(f"Total number of while loop iterations: {counter_of_points}")
                return

            # vectorised neighbour generation
            for drow, dcol in DIRS:
                di, dj = i + drow, j + dcol
                if not (0 <= di < HEIGHT and 0 <= dj < WIDTH):
                    continue
                if closed[di, dj]:
                    continue
                raw_cost = self.grid[di, dj]
                if raw_cost >= 70:                 # treat as obstacle
                    continue

                move_cost = 1.0 if (di == i or dj == j) else cost_sqrt2
                g_new = g[i, j] + raw_cost*move_cost
                if g_new >= g[di, dj]:             # not a better path
                    continue

                h_new = math.hypot(di - dest_i, dj - dest_j)
                f_new = g_new + weight * h_new
                g[di, dj] = g_new
                f[di, dj] = f_new
                parent_i[di, dj] = i
                parent_j[di, dj] = j
                heapq.heappush(open_list, (f_new, di, dj))

        self.get_logger().warn("Destination unreachable")



    def publish_sm_path(self, path_cells):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'  # Target frame

        for (mx, my) in path_cells:
            wx, wy = self.costmap_to_odom(mx, my)

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
                    continue

                odom_pt = tf2_geometry_msgs.do_transform_point(world_pt,
                    self.tf_buffer.lookup_transform('odom', world_pt.header.frame_id, rclpy.time.Time()))

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