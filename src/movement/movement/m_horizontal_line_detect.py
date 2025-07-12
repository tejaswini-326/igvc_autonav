#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist 
from sensor_msgs.msg import PointCloud2, PointField, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from movement.intersection_funcs import zero_copy_xy_pointcloud_reader_view

import tf2_ros
from geometry_msgs.msg import Point, PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


CLOSE_ENOUGH_TO_DESIRED_POINT_TRESHOLD = 0.2
DISTANCE_TO_PAUSE_IN_FRONT_OF_HORIZONTAL_LINE = -1.5  # m  ⬅ adjust if needed



# ---------- ROI & image parameters ------------------------------------------
RESOLUTION      = 0.05        # metres per pixel
LOOK_AHEAD_MAX  = 5.0         # forward range (0 … +5 m)
LOOK_SIDE_MAX   = 5.0         # lateral half-width (–5 … +5 m)

IMG_WIDTH_PX  = int(LOOK_AHEAD_MAX   / RESOLUTION)
IMG_HEIGHT_PX = int(2 * LOOK_SIDE_MAX / RESOLUTION)
# -----------------------------------------------------------------------------


K_ANG            = 1.2                 # proportional gain
ALIGN_ANGLE_TOL  = np.deg2rad(2.0)     # stop when |err| < 2°


class StopLineHoughFixedROI(Node):
    def __init__(self):
        super().__init__('stop_line_hough_roi')

        # --------------------------- I/O -------------------------------------
        self.cloud_sub = self.create_subscription(PointCloud2, '/igvc/white_points', self.cloud_cb, 10)

        self.pc_pub  = self.create_publisher(PointCloud2, '/stop_line_points', 10)
        self.mk_pub  = self.create_publisher(Marker,      '/stop_line_marker', 10)

        self.img_pub_1 = self.create_publisher(Image, '/hori/h1', 5)
        self.img_pub_2 = self.create_publisher(Image, '/hori/h2', 5)
        self.img_pub_3 = self.create_publisher(Image, '/hori/h3', 5)
        self.img_pub_4 = self.create_publisher(Image, '/hori/h4', 5)
        self.line_pub = self.create_publisher(Marker, '/hori/stop_line', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.intersection_pub = self.create_publisher(String, '/intersection', 10)

        self.bridge = CvBridge()
        self.cloud  = None
        self.has_stopped = False
        self.is_aligning_to_horizontal_line = False
        self.timer  = self.create_timer(1 / 100, self.timer_process)

        self.stop_point_pub = self.create_publisher(PointStamped, '/horizontal_line_stop_point', 10)  # for logic/consumption

        # TF buffer → odom ⇐ base_link
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    # --------------------------------------------------------------------- #
    def cloud_cb(self, msg: PointCloud2):
        self.cloud = msg

    # --------------------------------------------------------------------- #
    def detect_right_angle(self, mask):
        """
        Return:
            corner_xy, horiz_seg, vert_seg,             <-- as before
            all_hough_lines,                            <-- as before
            slope_lines                                 <-- NEW: filtered, parallel lines
        """
        perp_tol       = 10         # |Δθ − 90°| tolerance
        hough_thresh   = 30
        min_len        = 0.2
        max_gap        = 35

        edges = cv2.Canny(mask, 0, 255, apertureSize=3, L2gradient=False)
        self.img_pub_2.publish(self.bridge.cv2_to_imgmsg(
            cv2.flip(cv2.rotate(edges, cv2.ROTATE_90_CLOCKWISE), 0)))

        hough = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi / 180,
            threshold=hough_thresh,
            minLineLength=int(min_len * mask.shape[1]),
            maxLineGap=max_gap)

        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if hough is not None:
            for x1, y1, x2, y2 in hough[:, 0]:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        self.img_pub_3.publish(self.bridge.cv2_to_imgmsg(
            cv2.flip(cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE), 0), 'bgr8'))

        if hough is None:
            return None, None, None, None, []

        # -- compute angle + length² for every segment ----------------------
        segs = []
        for (x1, y1, x2, y2) in hough[:, 0]:
            dx, dy  = x2 - x1, y2 - y1
            angle   = (np.degrees(np.arctan2(dy, dx)) + 180) % 180
            length2 = dx * dx + dy * dy
            segs.append((angle, length2, (x1, y1, x2, y2)))

        # -- find perpendicular pair of max combined length ----------------
        cands = []
        for i in range(len(segs)):
            ai, li, si = segs[i]
            for j in range(i + 1, len(segs)):
                aj, lj, sj = segs[j]
                delta = abs(ai - aj)
                delta = min(delta, 180 - delta)
                if abs(delta - 90) <= perp_tol:
                    cands.append((-(li + lj), si, sj))

        if not cands:
            return None, None, None, None, []

        _, seg_a, seg_b = min(cands)  # lengths negated

        # -- intersection ----------------------------------------------------
        def intersect(a, b):
            x1, y1, x2, y2 = map(float, a)
            x3, y3, x4, y4 = map(float, b)
            den = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if abs(den) < 1e-6:
                return None
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
            return int(round(px)), int(round(py))

        corner = intersect(seg_a, seg_b)
        if corner is None:
            return None, None, None, None, []

        line_a = ((seg_a[0], seg_a[1]), (seg_a[2], seg_a[3]))
        line_b = ((seg_b[0], seg_b[1]), (seg_b[2], seg_b[3]))

        return corner, line_a, line_b, hough, []  # slope_lines will be filled outside

    # --------------------------------------------------------------------- #
    def timer_process(self):
        msg = self.cloud
        if msg is None:
            return

        pts_xy = zero_copy_xy_pointcloud_reader_view(msg)
        if pts_xy.size == 0:
            return

        x, y = pts_xy[:, 0], pts_xy[:, 1]
        in_roi = (x >= 0.0) & (x <= LOOK_AHEAD_MAX) & \
                 (y >= -LOOK_SIDE_MAX) & (y <= LOOK_SIDE_MAX)
        if not np.any(in_roi):
            return
        x, y = x[in_roi], y[in_roi]

        col = (x / RESOLUTION).astype(np.int32)
        row = ((y + LOOK_SIDE_MAX) / RESOLUTION).astype(np.int32)

        img = np.zeros((IMG_HEIGHT_PX, IMG_WIDTH_PX), dtype=np.uint8)
        img[row, col] = 255

        corner, horiz, vert, hough, _ = self.detect_right_angle(img)

        # ---------- slope-filtered view + keep parallel segments -----------
        slope_vis  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        slope_segs = []            # NEW – keep the cyan lines

        if corner is not None and hough is not None:
            def slope_mag(x1, y1, x2, y2):
                dx, dy = x2 - x1, y2 - y1
                return np.inf if dx == 0 else abs(dy / dx)

            cand_a = (*horiz[0], *horiz[1])
            cand_b = (*vert [0], *vert [1])
            ref    = cand_a if slope_mag(*cand_a) >= slope_mag(*cand_b) else cand_b

            rx1, ry1, rx2, ry2 = ref
            ref_ang = (np.degrees(np.arctan2(ry2 - ry1, rx2 - rx1)) + 180) % 180
            ANG_TOL = 5.0

            for x1, y1, x2, y2 in hough[:, 0]:
                ang = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180) % 180
                d   = abs(ang - ref_ang)
                d   = min(d, 180 - d)
                if d <= ANG_TOL:
                    cv2.line(slope_vis, (x1, y1), (x2, y2),
                             (255, 255, 0), 2)        # cyan
                    slope_segs.append((x1, y1, x2, y2))

         # ========== NEW SECTION – translate longest line to closest ==========
        if slope_segs:
            # -- longest segment --------------------------------------------
            # pick the segment that has the largest squared length
            long_seg = max(
                slope_segs,
                key=lambda s: (s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2)
            self.last_long_seg = long_seg

            x1l, y1l, x2l, y2l = long_seg
            dx, dy = x2l - x1l, y2l - y1l
            norm   = float(np.hypot(dx, dy))
            
            # unit normal of the line (points towards positive robot-X)
            nx, ny = -dy / norm, dx / norm
            if nx * (-x1l) + ny * (-y1l) > 0:     # pointing away → flip
                nx, ny = -nx, -ny

            # signed distance of every segment along the normal
            def proj(seg):
                return seg[0] * nx + seg[1] * ny      # use (x1,y1)

            long_s  = proj(long_seg)
            close_s = min(proj(seg) for seg in slope_segs)
            delta_s = close_s - long_s                # (+) ⇒ move towards robot

            # shift vector in pixel space
            shift_x = nx * delta_s
            shift_y = ny * delta_s

            # shifted pixel end-points
            p1_px = (x1l + shift_x, y1l + shift_y)
            p2_px = (x2l + shift_x, y2l + shift_y)

            # convert to metres (base_link frame)
            p1 = Point(x=p1_px[0] * RESOLUTION,
                       y=p1_px[1] * RESOLUTION - LOOK_SIDE_MAX,
                       z=-1.3)
            p2 = Point(x=p2_px[0] * RESOLUTION,
                       y=p2_px[1] * RESOLUTION - LOOK_SIDE_MAX,
                       z=-1.3)

            mk = Marker()
            mk.header = msg.header          # stays in base_link
            mk.ns, mk.id = "aligned_stop", 0
            mk.type, mk.action = Marker.LINE_STRIP, Marker.ADD
            mk.scale.x = 0.07               # 7 cm
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.0, 1.0, 1.0
            mk.points = [p1, p2]
            mk.lifetime = rclpy.duration.Duration(seconds=0.25).to_msg()
            self.line_pub.publish(mk)
        # =====================================================================


        # ---------- basic visualisation -------------------------------------
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if corner is not None:
            cv2.circle(vis, corner, 6, (0, 255, 0), -1)
            cv2.line(vis, *horiz, (0, 0, 255), 2)
            cv2.line(vis, *vert,  (255, 0, 0), 2)

        vis_rot = cv2.flip(cv2.rotate(vis, cv2.ROTATE_90_CLOCKWISE), 0)

        self.img_pub_1.publish(self.bridge.cv2_to_imgmsg(vis_rot, encoding='bgr8'))
        self.img_pub_4.publish(self.bridge.cv2_to_imgmsg(
            cv2.flip(cv2.rotate(slope_vis, cv2.ROTATE_90_CLOCKWISE), 0), 'bgr8'))
        

        # ──────────────────────────────────────────────────────────────────────
        # ALIGNMENT PHASE : rotate until the stop-line is perfectly horizontal
        # ──────────────────────────────────────────────────────────────────────
        if self.is_aligning_to_horizontal_line:
            if self.last_long_seg is None:
                self.get_logger().error("There is no horizontal line detected and we are stuck in the alignment code")
                return

            # slope of the reference segment (pixel space is OK for the sign test)
            dx_a = self.last_long_seg[2] - self.last_long_seg[0]
            dy_a = self.last_long_seg[3] - self.last_long_seg[1]

            # angle (rad) of the line w.r.t +x axis in image coords
            ang = np.arctan2(dy_a, dx_a)            # +ve ⇒ “/”   , –ve ⇒ “\”

            # desired is perfectly horizontal ⇒ target angle = 0
            err = ang

            twist = Twist()
            twist.angular.z =  K_ANG * err     # +err → +z (CCW ≈ left turn)

            # publish cmd_vel
            self.cmd_pub.publish(twist)

            # if almost horizontal, stop rotating and exit alignment mode
            if abs(err) < ALIGN_ANGLE_TOL:
                self.cmd_pub.publish(Twist())       # zero velocities
                self.is_aligning_to_horizontal_line = False

                msg = String()
                msg.data = 'left'
                self.intersection_pub.publish(msg)
            return
        

        if slope_segs:
            # ---------- pause-point (base_link) ---------------------------------
            mid_x = 0.5 * (p1.x + p2.x)
            mid_y = 0.5 * (p1.y + p2.y)

            # n̂ already points toward the robot in pixel space -> same sign in metres
            nx_m = nx                                          # px normal *already* unit
            ny_m = ny

            horizontal_stop_point_x = mid_x + nx_m * DISTANCE_TO_PAUSE_IN_FRONT_OF_HORIZONTAL_LINE
            horizontal_stop_point_y = mid_y + ny_m * DISTANCE_TO_PAUSE_IN_FRONT_OF_HORIZONTAL_LINE


            if self.has_stopped and abs(horizontal_stop_point_x) < CLOSE_ENOUGH_TO_DESIRED_POINT_TRESHOLD and abs(horizontal_stop_point_y) < CLOSE_ENOUGH_TO_DESIRED_POINT_TRESHOLD:
                intersection_msg = String()
                intersection_msg.data = 'aligning_to_stop_line'
                self.get_logger().info(f"📢 Publishing On Intersection Topic: Msg Data: {intersection_msg.data}")
                self.intersection_pub.publish(intersection_msg)
                self.is_aligning_to_horizontal_line = True



            pause_bl = Point()
            pause_bl.x = horizontal_stop_point_x
            pause_bl.y = horizontal_stop_point_y
            pause_bl.z = 0.0

            # ---------- transform to odom ---------------------------------------
            try:
                tf = self.tf_buffer.lookup_transform(
                    'odom',                # target frame
                    'base_link',           # source frame
                    rclpy.time.Time())     # latest available

                pause_stamped = PointStamped()
                pause_stamped.header.frame_id = 'base_link'
                pause_stamped.header.stamp    = msg.header.stamp
                pause_stamped.point = pause_bl

                tf = self.tf_buffer.lookup_transform(
                    'odom', 'base_link', rclpy.time.Time())

                pause_odom_stamped = tf2_geometry_msgs.do_transform_point(pause_stamped, tf)
                self.stop_point_pub.publish(pause_odom_stamped)


            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                # transform not yet available – skip this frame
                pass

    
    def odom_cb(self, msg: Odometry):
        self.has_stopped = (msg.twist.twist.linear.x == 0 and msg.twist.twist.linear.y == 0 and msg.twist.twist.angular.z == 0)


# --------------------------------------------------------------------------- #
def main(args=None):
    rclpy.init(args=args)
    node = StopLineHoughFixedROI()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
