#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Float64MultiArray
from tf_transformations import euler_from_quaternion
import math
from math import pi
from visualization_msgs.msg import Marker       

THRESHOLD_DISTANCE_FOR_DECLARING_THAT_WE_REACHED_A_WAYPOINT = 3

def normalise_angle(angle: float) -> float:
    """Wrap `angle` to the interval [-π, π]."""
    return (angle + pi) % (2 * pi) - pi


def calculate_distance_and_bearing(latitude, longitude, waypt_lat, waypt_lon):
    lat0 = math.radians(latitude)
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0)

    dlat = waypt_lat - latitude
    dlon = waypt_lon - longitude
    northing = dlat * m_per_deg_lat
    easting  = dlon * m_per_deg_lon

    distance = math.hypot(easting, northing)
    # ENU: angle from +X (East) CCW to +Y (North)
    bearing  = math.atan2(northing, easting)
    return distance, bearing



class GPSNextWaypointPublisherNode(Node):
    def __init__(self):
        super().__init__('gps_next_waypoint_publisher_node')

        self.latitude = 0.0
        self.longitude = 0.0
        self.yaw = 0.0
        self.waypoint_index = 0

        self.x = None
        self.y = None

        self.create_subscription(NavSatFix, '/navsat', self.navsat_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.waypoint_marker_pub = self.create_publisher(Marker, '/igvc/next_waypoint_rviz_marker', 1)
        self.next_waypoint_publisher = self.create_publisher(Float64MultiArray, '/igvc/next_waypoint', 10) # msg.data = [distance, heading_error, waypoint_index]

        # Static list of waypoints: [latitude, longitude]
        self.waypoints = [
            [47.47927341782438, 19.057658340347178], # little ahead
            [47.479265885071094, 19.057819240025868], # lot ahead
            [47.47914137816071, 19.05807806011922], # lot ahead to the right in the open
        ]

        self.create_timer(0.05, self.publish_next_waypoint)

    def navsat_callback(self, msg: NavSatFix):
        self.latitude = msg.latitude
        self.longitude = msg.longitude

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
        self.yaw = yaw


    def publish_next_waypoint(self):
        idx = self.waypoint_index

        # If we have reached our last waypoint
        if idx >= len(self.waypoints):
            self.get_logger().info('No more waypoints. Stopping.')
            msg = Float64MultiArray()
            msg.data = [0, 0, idx]
            self.next_waypoint_publisher.publish(msg)
            return

        distance, bearing = calculate_distance_and_bearing(self.latitude, self.longitude, self.waypoints[idx][0], self.waypoints[idx][1])

        # This bearing re-adjustment converts the angle of the navsat systemt to our gazebo system
        # Might need to be changed for the real bot
        #bearing = normalise_angle(-bearing + pi)
        heading_error = normalise_angle(bearing - self.yaw)

        msg = Float64MultiArray()
        msg.data = [float(distance), float(heading_error), float(idx)]
        self.next_waypoint_publisher.publish(msg)


        # ── ③  publish a visual marker in odom/map ────────────────────────
        # convert polar → cartesian in robot frame, then to odom
        if self.x is not None:
            # ------------------------------------------------------------------
            # Convert polar (distance, heading_error) → Cartesian in *robot frame*
            # ------------------------------------------------------------------
            global_x = self.x + distance * math.cos(self.yaw + heading_error)
            global_y = self.y + distance * math.sin(self.yaw + heading_error)
            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'odom'

            m.ns   = 'next_wp'
            m.id   = idx
            m.type = Marker.SPHERE                    # or ARROW, CUBE …
            m.action = Marker.ADD

            m.pose.position.x = global_x
            m.pose.position.y = global_y
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0               # no rotation needed

            m.scale.x = m.scale.y = m.scale.z = 0.3   # 30 cm sphere
            m.color.r = 1.0;  m.color.g = 0.3;  m.color.b = 0.0;  m.color.a = 1.0

            self.waypoint_marker_pub.publish(m)                # ➋ NEW
        ## ---------------------------------------------------------------------

        if distance < THRESHOLD_DISTANCE_FOR_DECLARING_THAT_WE_REACHED_A_WAYPOINT:
            self.get_logger().info(f'Reached Waypoint {idx}. Beginning to publish next waypoint.')
            self.waypoint_index += 1



def main(args=None):
	rclpy.init(args=args)
	node = GPSNextWaypointPublisherNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
    main()
