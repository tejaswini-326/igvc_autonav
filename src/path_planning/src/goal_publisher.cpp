// #define LANE_DEBUG
// #define RVIZ_DISTANCE_DEBUG

constexpr double MINIMUM_TIME_BEFORE_SWITCHING_LANES_AGAIN = 5.0;

constexpr int LANE_HISTORY_BUFFER_SIZE = 10;

constexpr double MAX_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE = 7.0;
constexpr double MIN_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE = 4.0;

constexpr double REDUCED_MAX_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE = 5.0;
constexpr double REDUCED_MIN_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE = 0.0;

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "object_detection/msg/object_array.hpp"
#include <sstream>
#include <iomanip>

#include <map> 
#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <deque>
#include <unordered_set>


using std::placeholders::_1;
using namespace std;
typedef geometry_msgs::msg::Point pt;

inline std::pair<double, double> operator+(const std::pair<double, double> &a, const std::pair<double, double> &b)
{
    return {a.first + b.first, a.second + b.second};
}

inline std::pair<double, double> operator-(const std::pair<double, double> &a, const std::pair<double, double> &b)
{
    return {a.first - b.first, a.second - b.second};
}

inline std::pair<double, double> operator*(const std::pair<double, double> &p, double scalar)
{
    return {p.first * scalar, p.second * scalar};
}

inline std::pair<double, double> operator/(const std::pair<double, double> &p, double scalar)
{
    if(scalar == 0.0) return{0.0, 0.0};
    return {p.first / scalar, p.second / scalar};
}

class GoalPublisher : public rclcpp::Node
{
public:
    GoalPublisher() : Node("goal_publisher")
    {
        last_lane_switch_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        override_ = "none";
        target_lane_ = "right";
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock(), tf2::durationFromSec(2.0));
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_point", 10);
        debug_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/debug_points", 10);
        marker_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lane_visualization", 10, std::bind(&GoalPublisher::marker_callback, this, _1));
        override_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/intersection", 10, std::bind(&GoalPublisher::override_callback_, this, _1));
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&GoalPublisher::odom_callback, this, _1));
        object_data_sub_ = this->create_subscription<object_detection::msg::ObjectArray>(
            "/object_data", 10, std::bind(&GoalPublisher::object_data_callback, this, _1));
        pothole_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "/pothole_position", 10,
            std::bind(&GoalPublisher::pothole_callback, this, _1));

#ifdef RVIZ_DISTANCE_DEBUG
        distance_viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/a_goalpub_debug_distances", 10);
#endif
        pothole_detected_ = false;
        pothole_pos_.x = pothole_pos_.y = pothole_pos_.z = 0.0;
        RCLCPP_INFO(this->get_logger(), "GoalPublisher node initialized");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr override_sub_;
    rclcpp::Subscription<object_detection::msg::ObjectArray>::SharedPtr object_data_sub_;

#ifdef RVIZ_DISTANCE_DEBUG
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr distance_viz_pub_;
#endif 

    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr pothole_sub_;
    rclcpp::Time pothole_stamp_;                 // when last pothole msg received
    double pothole_ttl_sec_ = 1.0;               // consider param; 1s default

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::map<std::string, geometry_msgs::msg::Point> detected_objects_;

    std::string target_lane_;
    std::string override_;
    std::pair<double, double> olp_, omp_, orp_;
    std::pair<double, double> robot_pose_;
    std::optional<geometry_msgs::msg::PoseStamped> last_goal_;

    rclcpp::Time last_lane_switch_;

    // Pothole tracking
    bool pothole_detected_;
    geometry_msgs::msg::Point pothole_pos_;

    std::deque<visualization_msgs::msg::Marker> left_lane_history_;
    std::deque<visualization_msgs::msg::Marker> right_lane_history_;
    std::deque<visualization_msgs::msg::Marker> middle_lane_history_;

    void pothole_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
    {
        geometry_msgs::msg::PointStamped in_pt = *msg, out_pt;

        try {
            // Transform from the incoming frame (msg->header.frame_id) to odom
            auto tf = tf_buffer_->lookupTransform(
                "odom", msg->header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.2));

            tf2::doTransform(in_pt, out_pt, tf);

            pothole_pos_ = out_pt.point;          // now in odom coords
            pothole_stamp_ = this->get_clock()->now();
            pothole_detected_ = true;

            RCLCPP_INFO(this->get_logger(),
                        "Pothole @ odom (%.2f, %.2f, %.2f) from frame '%s'",
                        pothole_pos_.x, pothole_pos_.y, pothole_pos_.z,
                        msg->header.frame_id.c_str());
        }
        catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Pothole TF failed: %s", ex.what());
            // leave pothole_detected_ as-is (or clear?)
        }
    }

    void override_callback_(const std_msgs::msg::String::SharedPtr msg)
    {
        override_ = msg->data;
        target_lane_ = "right";
        RCLCPP_INFO(this->get_logger(), "Received from /intersection: '%s'", override_.c_str());
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        robot_pose_.first = msg->pose.pose.position.x;
        robot_pose_.second = msg->pose.pose.position.y;
    }

    std::pair<double, double>
    get_last_point(const std::vector<geometry_msgs::msg::Point> &points,
                   double max_distance = MAX_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE,
                   double min_distance = MIN_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE)
    {
        double max_distance_squared = 0.0;
        pt ans;
        ans.x = 0.0;
        ans.y = 0.0;
        const double min_sq = min_distance * min_distance;
        const double max_sq = max_distance * max_distance;

#ifdef RVIZ_DISTANCE_DEBUG
        using geometry_msgs::msg::Point;
        using std_msgs::msg::ColorRGBA;
        using visualization_msgs::msg::Marker;
        using visualization_msgs::msg::MarkerArray;

        const rclcpp::Time stamp = now();

        /* -------- Marker: candidate points (sphere list) -------- */
        Marker pts;
        pts.header.frame_id = "odom"; // <-- change if you use another fixed frame
        pts.header.stamp = stamp;
        pts.ns = "distance_debug";
        pts.id = 0;
        pts.type = Marker::SPHERE_LIST;
        pts.action = Marker::ADD;
        pts.scale.x = pts.scale.y = pts.scale.z = 0.12;
        pts.lifetime = rclcpp::Duration(0, 0); // forever until replaced

        /* -------- We'll collect optional TEXT markers in a vector -------- */
        MarkerArray markers;
        std::vector<Marker> text_markers;
        text_markers.reserve(points.size());

        /* -------- Iterate over points -------- */
        uint32_t text_id = 100; // start text IDs well above 0/1/2 we use elsewhere
        size_t in_window_count = 0;
#endif

        for (const pt &p : points)
        {
            double dx = p.x - robot_pose_.first;
            double dy = p.y - robot_pose_.second;
            double current_distance_squared = dx * dx + dy * dy;
            if (current_distance_squared >= min_sq && 
                current_distance_squared <= max_sq && 
                current_distance_squared > max_distance_squared)
            {
                ans = p;
                max_distance_squared = current_distance_squared;
            }

#ifdef RVIZ_DISTANCE_DEBUG
            // Colour bucket
            ColorRGBA c;
            if (current_distance_squared < min_sq)
            {
                c.r = 1.0;
                c.g = 0.0;
                c.b = 0.0;
                c.a = 1.0;
            } // RED  too close
            else if (current_distance_squared > max_sq)
            {
                c.r = 0.5;
                c.g = 0.5;
                c.b = 0.5;
                c.a = 0.4;
            } // GREY too far
            else
            {
                c.r = 0.0;
                c.g = 1.0;
                c.b = 0.0;
                c.a = 1.0;
                in_window_count++;
            } // GREEN in range

            pts.points.push_back(p);
            pts.colors.push_back(c);

            // Optional per-point label (only for points in window, else unreadable)
            if (current_distance_squared >= min_sq && current_distance_squared <= max_sq)
            {
                Marker txt;
                txt.header = pts.header;
                txt.ns = "distance_debug_txt";
                txt.id = text_id++; // unique per publish cycle
                txt.type = Marker::TEXT_VIEW_FACING;
                txt.action = Marker::ADD;
                txt.pose.position = p;
                txt.pose.position.z += 0.15; // float above the sphere
                txt.scale.z = 0.30;          // text height in meters
                txt.color.r = 1.0;
                txt.color.g = 1.0;
                txt.color.b = 1.0;
                txt.color.a = 1.0;
                double dist = std::sqrt(current_distance_squared);
                char buf[16];
                std::snprintf(buf, sizeof(buf), "%.2f", dist);
                txt.text = buf;
                txt.lifetime = rclcpp::Duration(0, 0); // persistent until overwritten
                text_markers.push_back(std::move(txt));
            }
#endif
        }

#ifdef RVIZ_DISTANCE_DEBUG
        /* -------- Marker: chosen point -------- */
        Marker chosen;
        chosen.header = pts.header;
        chosen.ns = "distance_debug";
        chosen.id = 1;
        chosen.type = Marker::SPHERE;
        chosen.action = Marker::ADD;
        chosen.scale.x = chosen.scale.y = chosen.scale.z = 0.18;
        chosen.color.r = 0.0;
        chosen.color.g = 1.0;
        chosen.color.b = 1.0;
        chosen.color.a = 1.0;
        chosen.pose.position = ans;
        chosen.lifetime = rclcpp::Duration(0, 0);

        /* -------- Marker: line robot → chosen -------- */
        Marker sel_line;
        sel_line.header = pts.header;
        sel_line.ns = "distance_debug";
        sel_line.id = 2;
        sel_line.type = Marker::LINE_STRIP;
        sel_line.action = Marker::ADD;
        sel_line.scale.x = 0.05;
        sel_line.color.r = 1.0;
        sel_line.color.g = 1.0;
        sel_line.color.b = 0.0;
        sel_line.color.a = 1.0;
        {
            Point robot_pt;
            robot_pt.x = robot_pose_.first;
            robot_pt.y = robot_pose_.second;
            robot_pt.z = 0.0;
            sel_line.points = {robot_pt, ans};
        }
        sel_line.lifetime = rclcpp::Duration(0, 0);

        /* -------- Marker: summary text at robot -------- */
        Marker summary;
        summary.header = pts.header;
        summary.ns = "distance_debug_summary";
        summary.id = 3;
        summary.type = Marker::TEXT_VIEW_FACING;
        summary.action = Marker::ADD;
        summary.pose.position.x = robot_pose_.first;
        summary.pose.position.y = robot_pose_.second;
        summary.pose.position.z = 0.6; // above robot
        summary.scale.z = 0.45;
        summary.color.r = 1.0;
        summary.color.g = 1.0;
        summary.color.b = 0.0;
        summary.color.a = 1.0;
        {
            double chosen_dist = (max_distance_squared > 0.0) ? std::sqrt(max_distance_squared) : 0.0;
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(2);
            oss << "min=" << min_distance << "  max=" << max_distance
                << "  chosen=" << chosen_dist << "  N=" << in_window_count;
            summary.text = oss.str();
        }
        summary.lifetime = rclcpp::Duration(0, 0);

        /* -------- Assemble & publish -------- */
        markers.markers.reserve(3 + 1 + text_markers.size());
        markers.markers.push_back(pts);
        markers.markers.push_back(chosen);
        markers.markers.push_back(sel_line);
        markers.markers.push_back(summary);
        markers.markers.insert(markers.markers.end(),
                               std::make_move_iterator(text_markers.begin()),
                               std::make_move_iterator(text_markers.end()));

        distance_viz_pub_->publish(markers);
#endif

        return {ans.x, ans.y};
    }

 

    void publish_goal(const geometry_msgs::msg::PointStamped &goal_point)
    {
        geometry_msgs::msg::PoseStamped goal_pose;
        goal_pose.header.stamp = this->get_clock()->now();
        goal_pose.header.frame_id = "odom";
        goal_pose.pose.position.x = goal_point.point.x;
        goal_pose.pose.position.y = goal_point.point.y;
        goal_pose.pose.position.z = 0.0;
        goal_pose.pose.orientation.w = 1.0;

        if (override_ == "none")
        {
            goal_pub_->publish(goal_pose);
            last_goal_ = goal_pose;
        }
    }

    void debug_markers()
    {
        visualization_msgs::msg::MarkerArray MarkerArray;
        if(!right_lane_history_.empty()) MarkerArray.markers.push_back(right_lane_history_[0]);  // red
        if(!middle_lane_history_.empty()) MarkerArray.markers.push_back(middle_lane_history_[0]); // green
        if(!left_lane_history_.empty()) MarkerArray.markers.push_back(left_lane_history_[0]);   // blue

        debug_pub_->publish(MarkerArray);
    }

    void marker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        try
        {
            transformStamped = tf_buffer_->lookupTransform("odom", "camera_link", tf2::TimePointZero, tf2::durationFromSec(0.5));
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed in marker_callback: %s", ex.what());
            return;
        }
        for (const auto &marker : msg->markers)
        {
            try
            {
                visualization_msgs::msg::Marker transformed_marker = marker;
                transformed_marker.header.frame_id = "odom";

                for (auto &pt : transformed_marker.points)
                {
                    geometry_msgs::msg::PointStamped in_pt, out_pt;
                    in_pt.header = marker.header;
                    in_pt.point = pt;

                    tf2::doTransform(in_pt, out_pt, transformStamped);
                    pt = out_pt.point;
                }

                if (marker.id == 0)
                {
#ifdef LANE_DEBUG
                    cout << "FOUND LEFT LANE\n";
#endif
                    std::pair<double, double> pair = get_last_point(transformed_marker.points);
                    if (pair.first == 0.0 && pair.second == 0.0)
                    {
                        if (!left_lane_history_.empty())
                        {
#ifdef LANE_DEBUG
                            cout << "LEFT LANE HAS NO GOOD POINT. SO POINT TAKEN FROM HISTORY\n";
#endif
                            olp_ = get_last_point(left_lane_history_[0].points, 
                                REDUCED_MAX_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE, 
                                REDUCED_MIN_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE);
                        }
                    }
                    else
                    {
#ifdef LANE_DEBUG
                        cout<<"GOT LEFT LANE POINT FROM CURRENT DATA\n";
#endif
                        olp_ = pair;
                        left_lane_history_.push_front(transformed_marker);
                        while (left_lane_history_.size() > LANE_HISTORY_BUFFER_SIZE)
                            left_lane_history_.pop_back();
                    }
                }
                else if (marker.id == 1)
                {
#ifdef LANE_DEBUG
                    cout << "FOUND MID LANE\n";
#endif
                    std::pair<double, double> pair = get_last_point(transformed_marker.points);
                    if (pair.first == 0.0 && pair.second == 0.0)
                    {
                        if (!middle_lane_history_.empty())
                        {
#ifdef LANE_DEBUG
                            cout << "MID LANE HAS NO GOOD POINT. SO POINT TAKEN FROM HISTORY\n";
#endif
                            omp_ = get_last_point(middle_lane_history_[0].points, 
                                REDUCED_MAX_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE, 
                                REDUCED_MIN_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE);
                        }
                    }
                    else
                    {
#ifdef LANE_DEBUG
                        cout << "GOT MID LANE POINT FROM CURRENT DATA\n";
#endif
                        omp_ = pair;
                        middle_lane_history_.push_front(transformed_marker);
                        while (middle_lane_history_.size() > LANE_HISTORY_BUFFER_SIZE)
                            middle_lane_history_.pop_back();
                    }
                }
                else if (marker.id == 2)
                {
#ifdef LANE_DEBUG
                    cout << "FOUND RIGHT LANE\n";
#endif
                    std::pair<double, double> pair = get_last_point(transformed_marker.points);
                    if (pair.first == 0.0 && pair.second == 0.0)
                    {
                        if (!right_lane_history_.empty())
                        {
#ifdef LANE_DEBUG
                            cout << "RIGHT LANE HAS NO GOOD POINT. SO POINT TAKEN FROM HISTORY\n";
#endif
                            orp_ = get_last_point(right_lane_history_[0].points, 
                                REDUCED_MAX_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE, 
                                REDUCED_MIN_DISTANCE_TO_LOOK_FOR_POINTS_IN_LANE);
                        }
                    }
                    else
                    {
#ifdef LANE_DEBUG
                        cout << "GOT RIGHT LANE POINT FROM CURRENT DATA\n";
#endif
                        orp_ = pair;
                        right_lane_history_.push_front(transformed_marker);
                        while (right_lane_history_.size() > LANE_HISTORY_BUFFER_SIZE)
                            right_lane_history_.pop_back();
                    }
                }
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
                continue;
            }
        }

        //---------------------------------------------------------------------
        // Decide if an obstacle blocks the corridor we are following
        //---------------------------------------------------------------------
        bool corridor_blocked = false;
        std::string blocking_label;

        auto now = this->get_clock()->now();
        bool pothole_fresh = pothole_detected_ &&
                            ((now - pothole_stamp_) < rclcpp::Duration::from_seconds(pothole_ttl_sec_));

        if (pothole_fresh) {

            // ---- Pothole logic ----
            std::pair<double, double> pothole_xy{pothole_pos_.x, pothole_pos_.y};
            bool inside = (target_lane_ == "left")
                            ? is_between_lanes(olp_, omp_, pothole_xy)  // between left & mid
                            : is_between_lanes(orp_, omp_, pothole_xy); // between right & mid

            if (inside)
            {
                corridor_blocked = true;
                blocking_label = "pothole";
            }
        } else {
            // ---- Regular object detection logic ----
            static const std::unordered_set<std::string> kObstacles{
                "traffic barrel", "cone", "tire"};

            for (const auto &[label, p] : detected_objects_) {
                if (kObstacles.find(label) == kObstacles.end())
                    continue;

                std::pair<double, double> obj{p.x, p.y};
                bool inside = (target_lane_ == "left")
                                ? is_between_lanes(olp_, omp_, obj)
                                : is_between_lanes(orp_, omp_, obj);

                if (inside) {
                    corridor_blocked = true;
                    blocking_label = label;
                    break;
                }
            }
        }

        if (!pothole_fresh) {
            pothole_detected_ = false;   // optional: clear state when stale
        }
        

        if (corridor_blocked &&
            (now - last_lane_switch_) >
                rclcpp::Duration::from_seconds(MINIMUM_TIME_BEFORE_SWITCHING_LANES_AGAIN))
        {
            const std::string prev_lane = target_lane_;
            target_lane_ = (target_lane_ == "left" ? "right" : "left");
            last_lane_switch_ = now;

            RCLCPP_WARN(this->get_logger(),
                "Obstacle '%s' detected between %s & middle — switching from %s to %s lane",
                blocking_label.c_str(),
                prev_lane.c_str(),
                prev_lane.c_str(),
                target_lane_.c_str()
            );
        }
        //---------------------------------------------------------------------
        std::pair<double, double> goal;
        if (target_lane_ == "right")
        {
            goal = (orp_ + omp_) / 2;
        }
        if (target_lane_ == "left")
        {
            goal = (olp_ + omp_) / 2;
        }
#ifdef LANE_DEBUG
        cout << "GOAL IS: " << goal.first << ", " << goal.second << '\n';
#endif
        // Create PointStamped to use your existing `publish_goal()` function
        geometry_msgs::msg::PointStamped goal_point;
        goal_point.header.stamp = this->get_clock()->now();
        goal_point.header.frame_id = "odom";
        goal_point.point.x = goal.first;
        goal_point.point.y = goal.second;
        goal_point.point.z = 0.0;
        publish_goal(goal_point);

        debug_markers();
    }

    void object_data_callback(const object_detection::msg::ObjectArray::SharedPtr msg)
    {
        detected_objects_.clear();

        for (const auto &obj : msg->objects)
        {
            geometry_msgs::msg::PointStamped in_pt, out_pt;
            in_pt.point = obj.position;

            // 🔧 Hardcode the frame ID
            in_pt.header.frame_id = "camera_link";
            in_pt.header.stamp = this->get_clock()->now(); // Optional: keep timestamp current

            try
            {
                geometry_msgs::msg::TransformStamped transformStamped = tf_buffer_->lookupTransform(
                    "odom", in_pt.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.5));

                tf2::doTransform(in_pt, out_pt, transformStamped);

                detected_objects_[obj.label] = out_pt.point;

                // RCLCPP_INFO(this->get_logger(), "Detected %s at (%.2f, %.2f, %.2f) in odom",
                //             obj.label.c_str(),
                //             out_pt.point.x, out_pt.point.y, out_pt.point.z);
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Transform failed for %s: %s",
                            obj.label.c_str(), ex.what());
            }
        }
    }

    bool is_between_lanes(const std::pair<double, double> &lane_a,
                          const std::pair<double, double> &lane_b,
                          const std::pair<double, double> &obj) const
    {
        auto cross = [](const std::pair<double, double> &u,
                        const std::pair<double, double> &v)
        {
            return u.first * v.second - u.second * v.first;
        };
        auto dot = [](const std::pair<double, double> &u,
                      const std::pair<double, double> &v)
        {
            return u.first * v.first + u.second * v.second;
        };

        std::pair<double, double> a = lane_a - robot_pose_;
        std::pair<double, double> b = lane_b - robot_pose_;
        std::pair<double, double> p = obj - robot_pose_;

        // ahead of the robot?
        if (dot(a, p) <= 0.0 || dot(b, p) <= 0.0)
            return false;

        double cross_ab = cross(a, b);
        double cross_ap = cross(a, p);
        double cross_pb = cross(p, b);

        if (cross_ab > 0) // a → b is a left turn
            return cross_ap >= 0 && cross_pb >= 0;
        else // a → b is a right turn
            return cross_ap <= 0 && cross_pb <= 0;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GoalPublisher>());
    rclcpp::shutdown();
    return 0;
}