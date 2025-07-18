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

#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <deque>
#include <unordered_set>
// horizontal_line_stop_point -> pointstamped object, listens to it

// intersection -> "none"
using std::placeholders::_1;
using namespace std;
typedef geometry_msgs::msg::Point pt;

inline std::pair<double, double> operator*(const std::pair<double, double> &p, double scalar)
{
    return {p.first * scalar, p.second * scalar};
}

inline std::pair<double, double> operator+(const std::pair<double, double> &a, const std::pair<double, double> &b)
{
    return {a.first + b.first, a.second + b.second};
}

inline std::pair<double, double> operator-(const std::pair<double, double> &a, const std::pair<double, double> &b)
{
    return {a.first - b.first, a.second - b.second};
}

class GoalPublisher : public rclcpp::Node
{
public:
    GoalPublisher() : Node("goal_publisher")
    {
        last_lane_switch_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
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

        override_ = "none";
        target_lane_ = "right";
        current_lane_ = "right";

        lane_history_memory_buffer_size_ = 10;
        RCLCPP_INFO(this->get_logger(), "GoalPublisher node initialized");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr override_sub_;
    rclcpp::Subscription<object_detection::msg::ObjectArray>::SharedPtr object_data_sub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::map<std::string, geometry_msgs::msg::Point> detected_objects_;

    std::string target_lane_;
    std::string current_lane_;
    size_t lane_history_memory_buffer_size_;
    std::string override_;
    std::pair<double, double> olp_, omp_, orp_;
    std::pair<double, double> robot_pose_;
    std::optional<geometry_msgs::msg::PoseStamped> last_goal_;

    rclcpp::Time last_lane_switch_;

    struct tracked_points
    {
        std::pair<double, double> left;
        std::pair<double, double> mid;
        std::pair<double, double> right;
    };

    std::deque<visualization_msgs::msg::Marker> left_lane_history_;
    std::deque<visualization_msgs::msg::Marker> right_lane_history_;
    std::deque<visualization_msgs::msg::Marker> middle_lane_history_;

    void override_callback_(const std_msgs::msg::String::SharedPtr msg)
    {
        override_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "Overriding goal publisher\n");
    }
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        robot_pose_.first = msg->pose.pose.position.x;
        robot_pose_.second = msg->pose.pose.position.y;
    }

    std::pair<double, double> get_last_point(const std::vector<geometry_msgs::msg::Point> &points, double max_distance = 7.0, double min_distance = 4.0)
    {
        double max_distance_squared = 0.0;
        pt ans;
        ans.x = 0.0;
        ans.y = 0.0;
        for (const pt &p : points)
        {
            // cout << "POINT : (" << p.x << ", " << p.y << ")\n";
            double dx = p.x - robot_pose_.first;
            double dy = p.y - robot_pose_.second;
            double current_distance_squared = dx * dx + dy * dy;
            // cout << "CURRENT DISTANCE SQUARED: " << current_distance_squared << "\n";

            if (current_distance_squared >= min_distance * min_distance &&
                current_distance_squared <= max_distance * max_distance &&
                current_distance_squared > max_distance_squared)
            {
                ans = p;
                max_distance_squared = current_distance_squared;
            }
        }
        // cout << "DISTANCE SQUARED: " << max_distance_squared << " CALCULATED: (" << ans.x << ", " << ans.y << ")\n";
        // cout << "ROBOT POSITION CALCULATED: (" << robot_pose_.first << ", " << robot_pose_.second << ")\n";

        return {ans.x, ans.y};
    }

    void publish_goal(const geometry_msgs::msg::PointStamped &goal_point)
    {
        geometry_msgs::msg::PoseStamped goal_pose;
        goal_pose.header.stamp         = this->get_clock()->now();
        goal_pose.header.frame_id      = "odom";
        goal_pose.pose.position.x      = goal_point.point.x;
        goal_pose.pose.position.y      = goal_point.point.y;
        goal_pose.pose.position.z      = 0.0;
        goal_pose.pose.orientation.w   = 1.0;

        if (override_ == "none") {
            goal_pub_->publish(goal_pose);
            last_goal_ = goal_pose;
        }
    }

    void debug_markers()
    {
        visualization_msgs::msg::MarkerArray MarkerArray;
        MarkerArray.markers.push_back(right_lane_history_[0]);  // red
        MarkerArray.markers.push_back(middle_lane_history_[0]); // green
        MarkerArray.markers.push_back(left_lane_history_[0]);   // blue

        debug_pub_->publish(MarkerArray);
    }

    void marker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
    {
        int toggle[] = {0, 0, 0};
        for (const auto &marker : msg->markers)
        {
            geometry_msgs::msg::TransformStamped transformStamped;
            try
            {
                transformStamped = tf_buffer_->lookupTransform(
                    "odom", "camera_link", tf2::TimePointZero, tf2::durationFromSec(0.5));

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

                // Now store `transformed_marker` instead of `marker`
                if (marker.id == 0)
                {
                    toggle[0] = 1;
                    // cout << "RIGHT MARKER SIZE: " << transformed_marker.points.size() << "\n";
                    std::pair<double, double> pair = get_last_point(transformed_marker.points);
                    // cout << "RIGHT POINT CALCULATED: (" << pair.first << ", " << pair.second << ")\n";
                    if (pair.first == 0.0 && pair.second == 0.0)
                    {
                        if (!left_lane_history_.empty())
                        {
                            olp_ = get_last_point(left_lane_history_[0].points, 9.0, 0.0);
                        }
                        // cout<<"LEFT POINT TAKEN FROM HISTORY\n";
                    }
                    else
                    {
                        olp_ = pair;
                        left_lane_history_.push_front(transformed_marker);
                        while (left_lane_history_.size() > lane_history_memory_buffer_size_)
                            left_lane_history_.pop_back();
                    }
                }
                else if (marker.id == 1)
                {
                    toggle[1] = 1;
                    std::pair<double, double> pair = get_last_point(transformed_marker.points);
                    if (pair.first == 0.0 && pair.second == 0.0)
                    {
                        if (!middle_lane_history_.empty())
                        {
                            omp_ = get_last_point(middle_lane_history_[0].points, 9.0, 0.0);
                        }
                        // cout << "MID POINT TAKEN HISTORY: (" << omp_.first << ", " << omp_.second << ")\n";
                        //  cout<<"MID POINT TAKEN FROM HISTORY\n";
                    }
                    else
                    {
                        omp_ = pair;
                        middle_lane_history_.push_front(transformed_marker);
                        // cout << "MID POINT ADDED: (" << omp_.first << ", " << omp_.second << ")\n";
                        while (middle_lane_history_.size() > lane_history_memory_buffer_size_)
                            middle_lane_history_.pop_back();
                    }
                }
                else if (marker.id == 2)
                {
                    toggle[2] = 1;
                    std::pair<double, double> pair = get_last_point(transformed_marker.points);
                    if (pair.first == 0.0 && pair.second == 0.0)
                    {
                        if (!right_lane_history_.empty())
                        {
                            orp_ = get_last_point(right_lane_history_[0].points, 9.0, 0.0);
                        }
                        // cout<<"RIGHT POINT TAKEN FROM HISTORY\n";
                    }
                    else
                    {
                        orp_ = pair;
                        right_lane_history_.push_front(transformed_marker);
                        while (right_lane_history_.size() > lane_history_memory_buffer_size_)
                            right_lane_history_.pop_back();
                    }
                }
                else
                {
                    if (toggle[0] == 0)
                    {
                        if (!left_lane_history_.empty())
                        {
                            olp_ = get_last_point(left_lane_history_[0].points, 9.0, 0.0);
                        }
                    }
                    if (toggle[1] == 0)
                    {
                        if (!middle_lane_history_.empty())
                        {
                            omp_ = get_last_point(middle_lane_history_[0].points, 9.0, 0.0);
                        }
                    }
                    if (toggle[2] == 0)
                    {
                        if (!right_lane_history_.empty())
                        {
                            orp_ = get_last_point(right_lane_history_[0].points, 9.0, 0.0);
                        }
                    }
                }
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
                continue;
                ;
            }
        }

        if (right_lane_history_.size() < 3 || middle_lane_history_.size() < 3 || left_lane_history_.size() < 3 || override_ != "none")
        {
            if (last_goal_) goal_pub_->publish(*last_goal_);
            return;
        }








        //---------------------------------------------------------------------
        // Decide if an obstacle blocks the corridor we are following
        //---------------------------------------------------------------------
        static const std::unordered_set<std::string> kObstacles{
            "traffic barrel", "cone", "tire"};

        bool corridor_blocked = false;
        for (const auto &[label, p] : detected_objects_)
        {
            if (kObstacles.find(label) == kObstacles.end()) continue;

            std::pair<double,double> obj{p.x, p.y};
            bool inside = false;

            if (target_lane_ == "left")
                inside = is_between_lanes(olp_, omp_, obj);   // between left & mid
            else
                inside = is_between_lanes(orp_, omp_, obj);   // between right & mid

            if (inside) { corridor_blocked = true; break; }
        }

        auto now = this->get_clock()->now();
        if (corridor_blocked && (now - last_lane_switch_) > rclcpp::Duration::from_seconds(2.0))
        {
            target_lane_ = (target_lane_ == "left" ? "right" : "left");
            last_lane_switch_ = now;
            RCLCPP_WARN(this->get_logger(),
                        "Obstacle detected between %s & middle ‑> switching to %s lane",
                        (target_lane_ == "left" ? "right" : "left"),
                        target_lane_.c_str());
        }
        //---------------------------------------------------------------------













        std::pair<double, double> goal;
        if (target_lane_ == "right")
        {
            goal.first = (orp_.first + omp_.first) / 2;
            goal.second = (orp_.second + omp_.second) / 2;
        }
        if (target_lane_ == "left")
        {
            goal.first = (olp_.first + omp_.first) / 2;
            goal.second = (olp_.second + omp_.second) / 2;
        }

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

                RCLCPP_INFO(this->get_logger(), "Detected %s at (%.2f, %.2f, %.2f) in odom",
                            obj.label.c_str(),
                            out_pt.point.x, out_pt.point.y, out_pt.point.z);
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Transform failed for %s: %s",
                            obj.label.c_str(), ex.what());
            }
        }
    }


    bool is_between_lanes(const std::pair<double,double>& lane_a,
                        const std::pair<double,double>& lane_b,
                        const std::pair<double,double>& obj) const
    {
        auto vec = [](const std::pair<double,double>& s,
                    const std::pair<double,double>& t){
            return std::pair<double,double>{t.first - s.first, t.second - s.second};
        };
        auto cross = [](const std::pair<double,double>& u,
                        const std::pair<double,double>& v){
            return u.first * v.second - u.second * v.first;
        };
        auto dot = [](const std::pair<double,double>& u,
                    const std::pair<double,double>& v){
            return u.first * v.first + u.second * v.second;
        };

        auto a = vec(robot_pose_, lane_a);
        auto b = vec(robot_pose_, lane_b);
        auto p = vec(robot_pose_, obj);

        // ahead of the robot?
        if (dot(a,p) <= 0.0 || dot(b,p) <= 0.0) return false;

        double cross_ab = cross(a,b);
        double cross_ap = cross(a,p);
        double cross_pb = cross(p,b);

        if (cross_ab > 0)          // a → b is a left turn
            return cross_ap >= 0 && cross_pb >= 0;
        else                       // a → b is a right turn
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