// goal_publisher.cpp
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "object_detection/msg/object_array.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <tf2/utils.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <deque>
#include <map>

using std::placeholders::_1;
using namespace std;
typedef geometry_msgs::msg::Point pt;

/* ─────────── helper operators ─────────── */
inline std::pair<double, double>
operator*(const std::pair<double, double> &p, double scalar)
{ return {p.first * scalar, p.second * scalar}; }

inline std::pair<double, double>
operator+(const std::pair<double, double> &a, const std::pair<double, double> &b)
{ return {a.first + b.first, a.second + b.second}; }

inline std::pair<double, double>
operator-(const std::pair<double, double> &a, const std::pair<double, double> &b)
{ return {a.first - b.first, a.second - b.second}; }

inline double wrap_to_pi(double a)
{
    while (a >  M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

/* ─────────── node ─────────── */
class GoalPublisher : public rclcpp::Node
{
public:
    GoalPublisher() : Node("goal_publisher")
    {
        RCLCPP_INFO(this->get_logger(),"🛣️   goal publisher running");
        /* TF & pubs/subs */
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_point", 10);
        vel_pub_  = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        debug_pub_= this->create_publisher<visualization_msgs::msg::MarkerArray>("/debug_points", 10);

        marker_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lane_visualization", 10, std::bind(&GoalPublisher::marker_callback, this, _1));

        override_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/intersection", 10, std::bind(&GoalPublisher::override_callback_, this, _1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&GoalPublisher::odom_callback, this, _1));

        object_data_sub_ = this->create_subscription<object_detection::msg::ObjectArray>(
            "/object_data", 10, std::bind(&GoalPublisher::object_data_callback, this, _1));

        override_      = "none";
        target_lane_   = "right";
        current_lane_  = "right";
        buffer_size_   = 10;

        RCLCPP_INFO(this->get_logger(), "GoalPublisher node initialized");
    }

private:
    /* publishers & subscribers */
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr       vel_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;

    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr override_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<object_detection::msg::ObjectArray>::SharedPtr object_data_sub_;

    /* tf */
    std::shared_ptr<tf2_ros::Buffer>           tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    /* config / state */
    std::map<std::string, geometry_msgs::msg::Point> detected_objects_;
    std::string target_lane_;
    std::string current_lane_;
    size_t      buffer_size_;
    std::string override_;

    /* robot pose (odom) */
    std::pair<double,double> robot_pose_{0.0, 0.0};
    double robot_yaw_{0.0};

    /* last chosen lane points */
    std::pair<double,double> olp_, omp_, orp_;

    struct tracked_points
    {
        std::pair<double,double> left, mid, right;
    };

    std::deque<visualization_msgs::msg::Marker> left_lane_history_;
    std::deque<visualization_msgs::msg::Marker> middle_lane_history_;
    std::deque<visualization_msgs::msg::Marker> right_lane_history_;

    /* speed & gain */
    const double LIN_SPEED_ = 0.7;  // m/s
    const double KP_ANG_    = 1.5;  // rad/s per rad

    /* ─────────── callbacks ─────────── */
    void override_callback_(const std_msgs::msg::String::SharedPtr msg)
    {
        override_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "Override set to: %s", override_.c_str());
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        robot_pose_.first  = msg->pose.pose.position.x;
        robot_pose_.second = msg->pose.pose.position.y;
        robot_yaw_         = tf2::getYaw(msg->pose.pose.orientation);
    }

    std::pair<double,double>
    get_last_point(const std::vector<geometry_msgs::msg::Point> &points,
                   double max_distance = 10.0, double min_distance = 4.0)
    {
        double best_dist2 = 0.0;
        pt ans{};
        for (const pt &p : points)
        {
            double dx = p.x - robot_pose_.first;
            double dy = p.y - robot_pose_.second;
            double d2 = dx*dx + dy*dy;

            if (d2 >= min_distance*min_distance &&
                d2 <= max_distance*max_distance &&
                d2 > best_dist2)
            {
                ans = p;
                best_dist2 = d2;
            }
        }
        return {ans.x, ans.y};  // (0,0) if none found
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
            goal_pub_->publish(goal_pose);
    }

    void debug_markers()
    {
        if (left_lane_history_.empty() ||
            middle_lane_history_.empty() ||
            right_lane_history_.empty())
            return;

        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(right_lane_history_.front());  // red
        arr.markers.push_back(middle_lane_history_.front()); // green
        arr.markers.push_back(left_lane_history_.front());   // blue
        debug_pub_->publish(arr);
    }

    void object_data_callback(const object_detection::msg::ObjectArray::SharedPtr msg)
    {
        for (const auto &obj : msg->objects)
        {
            geometry_msgs::msg::PointStamped in_pt, out_pt;
            in_pt.point         = obj.position;
            in_pt.header.frame_id = "camera_link";
            in_pt.header.stamp  = this->get_clock()->now();

            try
            {
                auto tf = tf_buffer_->lookupTransform(
                    "odom", in_pt.header.frame_id, tf2::TimePointZero,
                    tf2::durationFromSec(0.5));

                tf2::doTransform(in_pt, out_pt, tf);
                detected_objects_[obj.label] = out_pt.point;
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Object TF failed: %s", ex.what());
            }
        }
    }

    /* ── velocity control when single‑lane available ── */
    void drive_along_tangent(const visualization_msgs::msg::Marker &lane_marker)
    {
        RCLCPP_INFO(this->get_logger(),"🛣️  Tangent Following Activated");
        const auto &pts = lane_marker.points;
        if (pts.size() < 2) return;

        const auto &p1 = pts[pts.size()-2]; 
        const auto &p2 = pts[pts.size()-1];

        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double desired_yaw = std::atan2(dy, dx);
        double heading_err = wrap_to_pi(desired_yaw - robot_yaw_);

        geometry_msgs::msg::Twist cmd;
        cmd.linear.x  = LIN_SPEED_;
        cmd.angular.z = KP_ANG_ * heading_err;
        vel_pub_->publish(cmd);
    }

    /* ───────── lane marker main callback ───────── */
    void marker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
    {
        int lane_seen[3] = {0,0,0};

        /* transform each lane marker to odom and save to history */
        for (const auto &marker : msg->markers)
        {
            try
            {
                auto tf = tf_buffer_->lookupTransform(
                    "odom", "camera_link", tf2::TimePointZero,
                    tf2::durationFromSec(0.5));

                visualization_msgs::msg::Marker m = marker;
                m.header.frame_id = "odom";

                for (auto &pt : m.points)
                {
                    geometry_msgs::msg::PointStamped inp, outp;
                    inp.header = marker.header;
                    inp.point  = pt;
                    tf2::doTransform(inp, outp, tf);
                    pt = outp.point;
                }

                switch (marker.id)
                {
                    case 0:
                        lane_seen[0]=1;
                        orp_ = get_last_point(m.points);
                        right_lane_history_.push_front(m);
                        while (right_lane_history_.size() > buffer_size_)
                            right_lane_history_.pop_back();
                        break;
                    case 1:
                        lane_seen[1]=1;
                        omp_ = get_last_point(m.points);
                        middle_lane_history_.push_front(m);
                        while (middle_lane_history_.size() > buffer_size_)
                            middle_lane_history_.pop_back();
                        break;
                    case 2:
                        lane_seen[2]=1;
                        olp_ = get_last_point(m.points);
                        left_lane_history_.push_front(m);
                        while (left_lane_history_.size() > buffer_size_)
                            left_lane_history_.pop_back();
                        break;
                }
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_WARN(this->get_logger(), "Lane TF failed: %s", ex.what());
            }
        }

        bool have_left   = !left_lane_history_.empty();
        bool have_mid    = !middle_lane_history_.empty();
        bool have_right  = !right_lane_history_.empty();

        if (override_ != "none") return;

        /* three lanes → original mid‑line goal logic */
        if (have_left && have_mid && have_right)
        {
            std::pair<double,double> goal;
            if (target_lane_ == "right") {
                goal.first  = (orp_.first  + omp_.first ) / 2;
                goal.second = (orp_.second + omp_.second) / 2;
            } else {
                goal.first  = (olp_.first  + omp_.first ) / 2;
                goal.second = (olp_.second + omp_.second) / 2;
            }

            geometry_msgs::msg::PointStamped gp;
            gp.header.stamp = this->get_clock()->now();
            gp.header.frame_id = "odom";
            gp.point.x = goal.first;
            gp.point.y = goal.second;
            gp.point.z = 0.0;
            publish_goal(gp);
            debug_markers();
            return;
        }

        /* single lane → velocity control */
        if (have_right && !have_mid && !have_left)
            drive_along_tangent(right_lane_history_.front());
        else if (have_left && !have_mid && !have_right)
            drive_along_tangent(left_lane_history_.front());
        else if (have_mid && !have_left && !have_right)
            drive_along_tangent(middle_lane_history_.front());

        debug_markers();
    }
};

/* ─────────── entry point ─────────── */
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GoalPublisher>());
    rclcpp::shutdown();
    return 0;
}
