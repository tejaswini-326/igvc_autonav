#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Float64MultiArray
from tf_transformations import euler_from_quaternion
import math
from math import pi

THRESHOLD_DISTANCE_FOR_DECLARING_THAT_WE_REACHED_A_WAYPOINT = 1

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
    easting = dlon * m_per_deg_lon

    distance = math.hypot(easting, northing)
    bearing = math.atan2(easting, northing)
    return distance, bearing


class GPSNextWaypointPublisherNode(Node):
    def __init__(self):
        super().__init__('gps_next_waypoint_publisher_node')

        self.latitude = 0.0
        self.longitude = 0.0
        self.yaw = 0.0
        self.waypoint_index = 0

        self.create_subscription(NavSatFix, '/navsat', self.navsat_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.next_waypoint_publisher = self.create_publisher(Float64MultiArray, '/igvc/next_waypoint', 10) # msg.data = [distance, heading_error, waypoint_index]

        # Static list of waypoints: [latitude, longitude]
        self.waypoints = [
            [47.47894028321059, 19.057691238180013],
            [47.478878, 19.058149],
            [47.479075, 19.058055],
            [47.478950, 19.057785]
        ]

        self.create_timer(0.1, self.publish_next_waypoint)

    def navsat_callback(self, msg: NavSatFix):
        self.latitude = msg.latitude
        self.longitude = msg.longitude

    def odom_cb(self, msg: Odometry):
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
        bearing = normalise_angle(-bearing + pi)
        heading_error = normalise_angle(bearing - self.yaw)

        msg = Float64MultiArray()
        msg.data = [float(distance), float(heading_error), float(idx)]
        self.next_waypoint_publisher.publish(msg)

        if distance < THRESHOLD_DISTANCE_FOR_DECLARING_THAT_WE_REACHED_A_WAYPOINT:
            self.get_logger().info(f'Reached Waypoint {idx}. Beginning to publish next waypoint.')
            idx += 1




def main(args=None):
    rclpy.init(args=args)
    node = GPSNextWaypointPublisherNode()
    try:
        node.waypoint_follower()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
