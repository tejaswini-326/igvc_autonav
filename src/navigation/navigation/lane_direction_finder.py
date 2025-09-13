#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import numpy as np
import math
from math import pi
from builtin_interfaces.msg import Time

ARROW_LEN = 2.0          # meters
ARROW_SHAFT_DIAM = 0.07  # meters
ARROW_HEAD_DIAM  = 0.15  # meters
ARROW_HEAD_LEN   = 0.25  # meters

def normalise_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


class LaneDirectionFinder(Node):
    def __init__(self):
        super().__init__("lane_direction_finder")
        self.sub = self.create_subscription(
            MarkerArray, "/lane_fitted_white", self.cb_markers, 10
        )
        self.pub_arrows = self.create_publisher(MarkerArray, "/lane_directions", 10)
        self.pub_angles = self.create_publisher(Float64MultiArray, "/lane_directions_angles", 10) # This is in bot's frame of reference

        self.ns = "lane_directions"
        # we’ll always use ids {0,1}, but keep this in case older runs left different ids
        self.prev_ids = set()

    # ---------------- helpers ----------------
    @staticmethod
    def _unit_vec(dx, dy):
        n = math.hypot(dx, dy)
        return None if n == 0.0 else (dx/n, dy/n)

    @staticmethod
    def _angle(dx, dy):
        return math.atan2(dy, dx)  # [-pi, pi]

    @staticmethod
    def _angle_mod_pi(theta):
        return theta % math.pi      # [0, pi)

    @staticmethod
    def _double_angle_mean(thetas_mod_pi):
        if len(thetas_mod_pi) == 0:
            return None
        arr = np.array(thetas_mod_pi)
        s = float(np.mean(np.sin(2.0 * arr)))
        c = float(np.mean(np.cos(2.0 * arr)))
        mean2 = math.atan2(s, c)                # (-pi, pi]
        theta = 0.5 * (mean2 if mean2 >= 0 else (mean2 + 2.0*math.pi))
        return theta % math.pi                  # [0, pi)

    def _make_arrow(self, frame_id, start, end, color=(0.0, 1.0, 0.0, 1.0), mid_z=-1.35, mid_id=0):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = Time(sec=0, nanosec=0)  # stable if TF lags
        m.ns = self.ns
        m.id = mid_id
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = ARROW_SHAFT_DIAM
        m.scale.y = ARROW_HEAD_DIAM
        m.scale.z = ARROW_HEAD_LEN
        m.color.r, m.color.g, m.color.b, m.color.a = color

        p0 = Point(); p1 = Point()
        p0.x, p0.y, p0.z = start
        p1.x, p1.y, p1.z = end
        p0.z = mid_z; p1.z = mid_z
        m.points = [p0, p1]
        return m

    def _delete_marker(self, frame_id, mid_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = Time(sec=0, nanosec=0)
        m.ns = self.ns
        m.id = mid_id
        m.action = Marker.DELETE
        return m

    def _delete_all(self, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = Time(sec=0, nanosec=0)
        m.action = Marker.DELETEALL
        ar = MarkerArray()
        ar.markers.append(m)
        self.pub_arrows.publish(ar)
        self.prev_ids.clear()

    # ---------------- callback ----------------
    def cb_markers(self, msg: MarkerArray):
        frame_id = msg.markers[0].header.frame_id if msg.markers else "map"

        # Parse lane line-strips
        segments = []
        angles = []
        for mk in msg.markers:
            if mk.type != Marker.LINE_STRIP or len(mk.points) < 2:
                continue
            p0 = mk.points[0]; p1 = mk.points[-1]
            dx, dy = (p1.x - p0.x, p1.y - p0.y)
            uv = self._unit_vec(dx, dy)
            if uv is None:
                continue
            segments.append(((p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z)))
            angles.append(self._angle(uv[0], uv[1]))

        # Nothing to publish: wipe all
        if len(segments) == 0:
            self._delete_all(frame_id)
            self.pub_angles.publish(Float64MultiArray())
            return

        # Compute orientation to use
        if len(segments) == 1:
            # single-lane mode → use that line's angle
            (s, e) = segments[0]
            uv = self._unit_vec(e[0]-s[0], e[1]-s[1])
            if uv is None:
                self._delete_all(frame_id)
                self.pub_angles.publish(Float64MultiArray())
                return
            theta = self._angle(uv[0], uv[1])
            color = (1.0, 0.0, 1.0, 1.0)   # MAGENTA = single-lane inference
        else:
            # multi-lane mode → dominant orientation (signless)
            thetas_mod_pi = [self._angle_mod_pi(th) for th in angles]
            theta = self._double_angle_mean(thetas_mod_pi)
            if theta is None:
                self._delete_all(frame_id)
                self.pub_angles.publish(Float64MultiArray())
                return
            color = (0.0, 1.0, 1.0, 1.0)   # CYAN = multi-lane orientation

        # Two opposite directions from the robot (origin)
        theta1 = theta
        theta2 = (theta + math.pi) % (2.0 * math.pi)
        o  = (0.0, 0.0, 0.0)
        e1 = (ARROW_LEN * math.cos(theta1), ARROW_LEN * math.sin(theta1), 0.0)
        e2 = (ARROW_LEN * math.cos(theta2), ARROW_LEN * math.sin(theta2), 0.0)

        out_ar = MarkerArray()
        out_ar.markers.append(self._make_arrow(frame_id, o, e1, color=color, mid_id=0))
        out_ar.markers.append(self._make_arrow(frame_id, o, e2, color=color, mid_id=1))

        # Precise delete of any stale IDs (from older runs / different IDs)
        current_ids = {m.id for m in out_ar.markers}
        if self.prev_ids:
            stale = self.prev_ids - current_ids
            if stale:
                del_ar = MarkerArray()
                for sid in stale:
                    del_ar.markers.append(self._delete_marker(frame_id, sid))
                self.pub_arrows.publish(del_ar)

        # Publish fresh arrows + angles
        self.pub_arrows.publish(out_ar)
        angles_msg = Float64MultiArray()
        angles_msg.data = [normalise_angle(theta1), normalise_angle(theta2)]
        self.pub_angles.publish(angles_msg)

        self.prev_ids = current_ids

        # # optional log
        # self.get_logger().info(
        #     f"Lane directions (rad): forward={theta1:+.3f}, reverse={theta2:+.3f} "
        #     f"mode={'SINGLE' if len(segments)==1 else 'MULTI'}"
        # )

def main(args=None):
    rclpy.init(args=args)
    node = LaneDirectionFinder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
