#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp" 
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "object_detection/msg/object_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <deque>
#include <unordered_map>
#include <limits>

using std::placeholders::_1;
using namespace std;
typedef geometry_msgs::msg::Point pt;

inline std::pair<double, double> operator*(const std::pair<double, double> &p, double scalar) { return {p.first * scalar, p.second * scalar}; }
inline std::pair<double, double> operator+(const std::pair<double, double> &a, const std::pair<double, double> &b) { return {a.first + b.first, a.second + b.second}; }
inline std::pair<double, double> operator-(const std::pair<double, double> &a, const std::pair<double, double> &b) { return {a.first - b.first, a.second - b.second}; }

class GoalPublisher : public rclcpp::Node {
public:
    GoalPublisher() : Node("goal_publisher") {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_point", 10);
        debug_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/debug_points", 10);
        vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        marker_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lane_visualization", 10, std::bind(&GoalPublisher::marker_callback, this, _1));
        override_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/intersection", 10, std::bind(&GoalPublisher::override_callback_, this, _1));
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&GoalPublisher::odom_callback, this, _1));
        object_data_sub_ = this->create_subscription<object_detection::msg::ObjectArray>(
            "/object_data", 10, std::bind(&GoalPublisher::object_data_callback, this, _1));
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/igvc/white_points", 10, std::bind(&GoalPublisher::pointcloud_callback, this, _1));
        tangent_points_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/tangent_points", 10);

        lanes_pub_ = this->create_publisher<std_msgs::msg::Int32>("/lane_count", 10);


        override_ = "none";
        target_lane_ = "right";
        current_lane_ = "right";

        last_lane_count_pub_ = -1;

        buffer_size_ = 10;
        RCLCPP_INFO(this->get_logger(), "GoalPublisher node initialized");
    }

private:
    float TUNABLE_LOOKAHEAD_FACTOR = 1.25; // Adjust this factor to change the lookahead distance
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr override_sub_;
    rclcpp::Subscription<object_detection::msg::ObjectArray>::SharedPtr object_data_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr vel_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    sensor_msgs::msg::PointCloud2::SharedPtr last_pointcloud_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tangent_points_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    // NEW: lane count publisher
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr lanes_pub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::map<std::string, geometry_msgs::msg::Point> detected_objects_;

    std::string target_lane_;
    std::string current_lane_;
    size_t buffer_size_;
    std::string override_;
    std::pair<double, double> olp_, omp_, orp_;
    std::pair<double, double> robot_pose_;
    tf2::Quaternion current_orientation_;

    // NEW: track last published count (avoid spam)
    int last_lane_count_pub_ = -1;

    struct tracked_points {
        std::pair<double, double> left;
        std::pair<double, double> mid;
        std::pair<double, double> right;
    };

    // === SIMPLE 2D POINT STRUCT FOR CLUSTERING ===
    struct Pt2 { double x; double y; };

    // === RADIUS OUTLIER FILTER (keeps points having >= min_neighbors within radius) ===
    std::vector<Pt2> radius_filter(const std::vector<Pt2>& pts, double radius, int min_neighbors) {
        double r2 = radius * radius;
        std::vector<Pt2> kept;
        kept.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            int cnt = 0;
            for (size_t j = 0; j < pts.size(); ++j) {
                if (i == j) continue;
                double dx = pts[i].x - pts[j].x;
                double dy = pts[i].y - pts[j].y;
                if (dx*dx + dy*dy <= r2) {
                    if (++cnt >= min_neighbors) {
                        kept.push_back(pts[i]);
                        break;
                    }
                }
            }
        }
        return kept;
    }

    // === MINIMAL DBSCAN FOR 2D (Optional but keeps clusters) ===
    class DBSCAN2D {
    public:
        DBSCAN2D(double eps, int minPts) : eps_(eps), eps2_(eps*eps), minPts_(minPts) {}
        std::vector<int> fit(const std::vector<Pt2>& pts) {
            int n = (int)pts.size();
            std::vector<int> labels(n, UNVISITED);
            int cluster_id = 0;
            std::vector<int> neigh; neigh.reserve(n);
            for (int i = 0; i < n; ++i) {
                if (labels[i] != UNVISITED) continue;
                regionQuery(pts, i, neigh);
                if ((int)neigh.size() < minPts_) { labels[i] = NOISE; continue; }
                labels[i] = cluster_id;
                std::deque<int> seeds(neigh.begin(), neigh.end());
                while (!seeds.empty()) {
                    int q = seeds.front(); seeds.pop_front();
                    if (labels[q] == NOISE) labels[q] = cluster_id;
                    if (labels[q] != UNVISITED) continue;
                    labels[q] = cluster_id;
                    std::vector<int> neigh2; regionQuery(pts, q, neigh2);
                    if ((int)neigh2.size() >= minPts_) {
                        for (int idx : neigh2) if (labels[idx] == UNVISITED) seeds.push_back(idx);
                    }
                }
                ++cluster_id;
            }
            return labels;
        }
        enum { UNVISITED = -2, NOISE = -1 };

    private:
        double eps_, eps2_; int minPts_;
        void regionQuery(const std::vector<Pt2>& pts, int i, std::vector<int>& out) {
            out.clear();
            const Pt2 &p = pts[i];
            for (int j = 0; j < (int)pts.size(); ++j) {
                double dx = p.x - pts[j].x; double dy = p.y - pts[j].y;
                if (dx*dx + dy*dy <= eps2_) out.push_back(j);
            }
        }
    };

    // Select the largest cluster (by number of points); simple heuristic
    std::vector<Pt2> largest_cluster(const std::vector<Pt2>& pts, const std::vector<int>& labels) {
        std::unordered_map<int, std::vector<Pt2>> buckets;
        for (size_t i = 0; i < pts.size(); ++i) {
            int lbl = labels[i];
            if (lbl < 0) continue; // skip noise/unvisited
            buckets[lbl].push_back(pts[i]);
        }
        size_t best_size = 0; int best_id = -1;
        for (auto &kv : buckets) {
            if (kv.second.size() > best_size) { best_size = kv.second.size(); best_id = kv.first; }
        }
        if (best_id == -1) return {};
        return buckets[best_id];
    }

    double get_yaw_from_quaternion(const tf2::Quaternion& q) {
        tf2::Matrix3x3 m(q); double roll, pitch, yaw; m.getRPY(roll, pitch, yaw); return yaw; }

    void override_callback_(const std_msgs::msg::String::SharedPtr msg) {
        override_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "Overriding goal publisher\n");
    }
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        robot_pose_.first = msg->pose.pose.position.x;
        robot_pose_.second = msg->pose.pose.position.y;
        tf2::fromMsg(msg->pose.pose.orientation, current_orientation_);
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) { last_pointcloud_ = msg; }

    void publish_tangent_points(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "odom";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "tangent_points";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.05;  // Line width
        marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 1.0;
        marker.points.push_back(p1); marker.points.push_back(p2);
        tangent_points_pub_->publish(marker);
    }

    std::pair<double, double> get_last_point(const std::vector<geometry_msgs::msg::Point> &points, double max_distance = 10.0, double min_distance = 4.0) {
        double max_distance_squared = 0.0; pt ans; ans.x = 0.0; ans.y = 0.0;
        for (const pt &p : points) {
            double dx = p.x - robot_pose_.first; double dy = p.y - robot_pose_.second; double current_distance_squared = dx * dx + dy * dy;
            if (current_distance_squared >= min_distance * min_distance &&
                current_distance_squared <= max_distance * max_distance &&
                current_distance_squared > max_distance_squared) {
                ans = p; max_distance_squared = current_distance_squared;
            }
        }
        return {ans.x, ans.y};
    }

    void publish_goal(const geometry_msgs::msg::PointStamped &goal_point) {
        geometry_msgs::msg::PoseStamped goal_pose;
        goal_pose.header.stamp = this->get_clock()->now();
        goal_pose.header.frame_id = "odom";
        goal_pose.pose.position.x = goal_point.point.x;
        goal_pose.pose.position.y = goal_point.point.y;
        goal_pose.pose.position.z = 0.0;
        goal_pose.pose.orientation.x = 0.0;
        goal_pose.pose.orientation.y = 0.0;
        goal_pose.pose.orientation.z = 0.0;
        goal_pose.pose.orientation.w = 1.0;
        if (override_ == "none") { goal_pub_->publish(goal_pose); }
    }

    void object_data_callback(const object_detection::msg::ObjectArray::SharedPtr msg) {
        for (const auto &obj : msg->objects) {
            geometry_msgs::msg::PointStamped in_pt, out_pt;
            in_pt.point = obj.position;
            // 🔧 Hardcode the frame ID
            in_pt.header.frame_id = "camera_link";
            in_pt.header.stamp = this->get_clock()->now();
            try {
                geometry_msgs::msg::TransformStamped transformStamped = tf_buffer_->lookupTransform(
                    "odom", in_pt.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.5));
                tf2::doTransform(in_pt, out_pt, transformStamped);
                detected_objects_[obj.label] = out_pt.point;
                RCLCPP_INFO(this->get_logger(), "Detected %s at (%.2f, %.2f, %.2f) in odom",
                            obj.label.c_str(), out_pt.point.x, out_pt.point.y, out_pt.point.z);
            } catch (tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Transform failed for %s: %s", obj.label.c_str(), ex.what());
            }
        }
    }

    std::optional<std::pair<double, double>> compute_tangent(
        const std::vector<geometry_msgs::msg::Point> &points,
        const std::pair<double, double> &robot_pose) {
        RCLCPP_INFO(rclcpp::get_logger("GoalPublisher"), "⭐ Tangent is being computed");
        const size_t num_points = points.size();
        if (num_points < 2) { RCLCPP_WARN(rclcpp::get_logger("GoalPublisher"), "Not enough points to compute tangent."); return std::nullopt; }
        const size_t lookahead_index = std::min<size_t>(
            num_points - 1, 
            static_cast<size_t>(std::round(num_points / TUNABLE_LOOKAHEAD_FACTOR))
        );
        const auto &p1 = points.front();
        const auto &p2 = points[lookahead_index];
        publish_tangent_points(p1, p2);
        double dx = p2.x - p1.x; double dy = p2.y - p1.y; double norm = std::sqrt(dx * dx + dy * dy);
        if (norm < 1e-3) { RCLCPP_WARN(rclcpp::get_logger("GoalPublisher"), "Tangent vector too small to normalize."); return std::nullopt; }
        dx /= norm; dy /= norm;
        double to_p2_x = p2.x - robot_pose.first; double to_p2_y = p2.y - robot_pose.second; double dot = dx * to_p2_x + dy * to_p2_y;
        if (dot < 0) { dx = -dx; dy = -dy; RCLCPP_INFO(rclcpp::get_logger("GoalPublisher"), "Tangent direction flipped to face forward."); }
        return std::make_pair(dx, dy);
    }

    void marker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
        int toggle[] = {0, 0, 0};
        for (const auto &marker : msg->markers) {
            geometry_msgs::msg::TransformStamped transformStamped;
            try {
                transformStamped = tf_buffer_->lookupTransform(
                    "odom", "camera_link", tf2::TimePointZero, tf2::durationFromSec(0.5));
                visualization_msgs::msg::Marker transformed_marker = marker;
                transformed_marker.header.frame_id = "odom";
                for (auto &pt : transformed_marker.points) {
                    geometry_msgs::msg::PointStamped in_pt, out_pt; in_pt.header = marker.header; in_pt.point = pt; tf2::doTransform(in_pt, out_pt, transformStamped); pt = out_pt.point;
                }
                if (marker.id == 0 && transformed_marker.points.size() >= 5) {
                    toggle[0] = 1;
                    olp_ = get_last_point(transformed_marker.points);
                } else if (marker.id == 1 && transformed_marker.points.size() >= 5) {
                    toggle[1] = 1;
                    omp_ = get_last_point(transformed_marker.points);
                } else if (marker.id == 2 && transformed_marker.points.size() >= 5) {
                    toggle[2] = 1;
                    orp_ = get_last_point(transformed_marker.points);
                }
            } catch (tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
                continue;
            }
        }

        int count_detected = toggle[0] + toggle[1] + toggle[2];
        
        // --- NEW: publish lane count only when it changes (avoid spam) ---
        if (count_detected != last_lane_count_pub_) {
            std_msgs::msg::Int32 lc_msg;
            lc_msg.data = count_detected;
            lanes_pub_->publish(lc_msg);
            last_lane_count_pub_ = count_detected;
            RCLCPP_INFO(this->get_logger(), "Lane count changed: %d", count_detected);
        }
        // ------------------------------------------------------------------

        // === SINGLE LANE (OR ONLY ONE MARKER) CASE WITH SIMPLE RADIUS FILTER + DBSCAN CLUSTERING ===
        if (count_detected == 1 && override_ == "none") {
            RCLCPP_INFO(this->get_logger(), "Single lane detected. Using point cloud with clustering.");
            if (!last_pointcloud_) { RCLCPP_WARN(this->get_logger(), "No point cloud data received yet."); return; }
            std::vector<geometry_msgs::msg::Point> lane_pts;
            try {
                geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                    "odom", last_pointcloud_->header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.5));
                sensor_msgs::PointCloud2Iterator<float> iter_x(*last_pointcloud_, "x");
                sensor_msgs::PointCloud2Iterator<float> iter_y(*last_pointcloud_, "y");
                sensor_msgs::PointCloud2Iterator<float> iter_z(*last_pointcloud_, "z");
                for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                    geometry_msgs::msg::PointStamped in_pt, out_pt; in_pt.header = last_pointcloud_->header;
                    in_pt.point.x = *iter_x; in_pt.point.y = *iter_y; in_pt.point.z = *iter_z;
                    tf2::doTransform(in_pt, out_pt, transform);
                    double dx = out_pt.point.x - robot_pose_.first; double dy = out_pt.point.y - robot_pose_.second;
                    float TUNABLE_LATERAL_ROI = 5.0; // Adjust this lateral ROI as needed
                    // Basic ROI: only forward, within 10m, lateral ±Tunable_lateral_roi meters
                    if (dx > 0.0 && dx*dx + dy*dy < 100.0 && std::fabs(dy) < TUNABLE_LATERAL_ROI) {
                        lane_pts.push_back(out_pt.point);
                    }
                }
                std::sort(lane_pts.begin(), lane_pts.end(), [&](const auto &a, const auto &b) {
                    double da = std::hypot(a.x - robot_pose_.first, a.y - robot_pose_.second);
                    double db = std::hypot(b.x - robot_pose_.first, b.y - robot_pose_.second);
                    return da < db;
                });
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
                return;
            }
            if (lane_pts.empty()) { RCLCPP_WARN(this->get_logger(), "Point cloud empty after ROI filter."); return; }

            // Convert to 2D for filtering
            std::vector<Pt2> raw2d; raw2d.reserve(lane_pts.size());
            for (auto &p : lane_pts) raw2d.push_back({p.x, p.y});

            // Radius speckle removal
            float TUNABLE_RADIUS = 0.40; // Adjust this radius as needed
            int TUNABLE_MIN_NEIGHBORS = 2; // Minimum neighbors to keep a point
            raw2d = radius_filter(raw2d, TUNABLE_RADIUS, TUNABLE_MIN_NEIGHBORS); // radius 0.30m, at least 3 neighbors
            if (raw2d.size() < 10) { RCLCPP_WARN(this->get_logger(), "Too few points after radius filter (%zu).", raw2d.size()); return; }

            // Estimate eps (simple): average nearest neighbor * 2.5 (bounded)
            double avg_nn = 0.0; if (raw2d.size() > 6) {
                double sum=0; int cnt=0; for (size_t i=0;i<raw2d.size();++i){ double best=1e9; for (size_t j=0;j<raw2d.size();++j){ if(i==j) continue; double dx=raw2d[i].x-raw2d[j].x; double dy=raw2d[i].y-raw2d[j].y; double d=std::hypot(dx,dy); if(d<best) best=d; } if(best<1e9){ sum+=best; ++cnt; } } if(cnt>0) avg_nn = sum/cnt; }
            double eps = std::clamp(avg_nn * 2.5, 0.12, 0.35);
            int minPts = 8;
            DBSCAN2D db(eps, minPts);
            auto labels = db.fit(raw2d);
            auto cluster_pts = largest_cluster(raw2d, labels);
            if (cluster_pts.empty()) { RCLCPP_WARN(this->get_logger(), "No cluster selected. Using all filtered points."); cluster_pts = raw2d; }

            // Rebuild geometry_msgs points from chosen cluster and sort by distance (like original logic)
            std::vector<geometry_msgs::msg::Point> ordered;
            ordered.reserve(cluster_pts.size());
            for (auto &p : cluster_pts) { geometry_msgs::msg::Point gp; gp.x=p.x; gp.y=p.y; gp.z=0.0; ordered.push_back(gp); }
            std::sort(ordered.begin(), ordered.end(), [&](const auto &a, const auto &b) {
                double da = std::hypot(a.x - robot_pose_.first, a.y - robot_pose_.second);
                double db = std::hypot(b.x - robot_pose_.first, b.y - robot_pose_.second);
                return da < db;
            });

            auto tangent_opt = compute_tangent(ordered, robot_pose_);
            if (tangent_opt) {
                auto [dx, dy] = tangent_opt.value();
                double current_yaw = get_yaw_from_quaternion(current_orientation_);
                double dx_robot =  cos(current_yaw) * dx + sin(current_yaw) * dy;
                double dy_robot = -sin(current_yaw) * dx + cos(current_yaw) * dy;
                double error_yaw = atan2(dy_robot, dx_robot);
                double k_angular = 1.5; double max_angular = 1.0;
                geometry_msgs::msg::Twist cmd_vel;
                cmd_vel.angular.z = std::clamp(k_angular * error_yaw, -max_angular, max_angular);
                cmd_vel.linear.x = 0.3 * std::max(0.0, 1.0 - std::abs(cmd_vel.angular.z));
                RCLCPP_INFO(this->get_logger(), "Yaw: %.2f | Tangent(robot): (%.2f, %.2f) | Err: %.2f | ang.z: %.2f | eps=%.2f",
                            current_yaw, dx_robot, dy_robot, error_yaw, cmd_vel.angular.z, eps);
                vel_pub_->publish(cmd_vel);
            } else {
                RCLCPP_WARN(this->get_logger(), "Tangent fitting failed.");
            }
            return; // end single-lane branch
        }

        if (override_ != "none") { return; }

        std::pair<double, double> goal;
        if (target_lane_ == "right") { goal.first = (orp_.first + omp_.first) / 2; goal.second = (orp_.second + omp_.second) / 2; }
        if (target_lane_ == "left")  { goal.first = (olp_.first + omp_.first) / 2; goal.second = (olp_.second + omp_.second) / 2; }

        geometry_msgs::msg::PointStamped goal_point; goal_point.header.stamp = this->get_clock()->now(); goal_point.header.frame_id = "odom";
        goal_point.point.x = goal.first; goal_point.point.y = goal.second; goal_point.point.z = 0.0;
        publish_goal(goal_point);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GoalPublisher>());
    rclcpp::shutdown();
    return 0;
}
/*
ROI filtering: Keeps only points within a forward 10m range and ±3m lateral.

Radius-based speckle removal: Removes points that have fewer than 3 neighbors within 0.30m.

DBSCAN clustering: Groups remaining points and selects the largest cluster for tangent computation.
*/
