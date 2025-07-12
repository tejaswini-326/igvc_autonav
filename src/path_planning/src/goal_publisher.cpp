#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "std_msgs/msg/string.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <deque>
// horizontal_line_stop_point -> pointstamped object, listens to it

// intersection -> "none" 
using std::placeholders::_1;
using namespace std;
typedef geometry_msgs::msg::Point pt;

inline std::pair<double, double> operator*(const std::pair<double, double> &p, double scalar){
    return {p.first * scalar, p.second * scalar};
}

inline std::pair<double, double> operator+(const std::pair<double, double> &a, const std::pair<double, double> &b){
    return {a.first + b.first, a.second + b.second};
}

class GoalPublisher : public rclcpp::Node
{
public:
    GoalPublisher() : Node("goal_publisher")
    {    
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_point", 10);
        debug_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/debug_points", 10);
        marker_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/lane_visualization", 10, std::bind(&GoalPublisher::marker_callback, this, _1));
        override_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/intersection", 10, std::bind(&GoalPublisher::override_callback_, this, _1));

        override_ = "none";
        target_lane_ = "right";
        current_lane_ = "right";

        buffer_size_ = 2;
        RCLCPP_INFO(this->get_logger(), "GoalPublisher node initialized");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr override_sub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::string target_lane_;
    std::string current_lane_;
    size_t buffer_size_;
    std::string override_;

    struct tracked_points {
        std::pair<double, double> left;
        std::pair<double, double> mid;
        std::pair<double, double> right;
    };
    std::deque<tracked_points> history_;

    void override_callback_(const std_msgs::msg::String::SharedPtr msg)
    {
        override_ = msg->data;
        RCLCPP_INFO(this->get_logger(), "Overriding goal publisher\n");
    }

    std::pair<double, double> get_last_point(const std::vector<geometry_msgs::msg::Point> &points, double max_distance = 6.5)
    {
        double distance_squared = 0.0;
        pt ans;
        ans.x = 0.0;
        ans.y = 0.0;
        for (const pt &p : points)
        {
            if (p.x * p.x + p.y * p.y <= max_distance * max_distance && p.x * p.x + p.y * p.y > distance_squared)
            {
                ans = p;
                distance_squared = p.x * p.x + p.y * p.y;
            }
        }
        return {ans.x, ans.y};
    }

    std::optional<geometry_msgs::msg::PointStamped> transform_to_odom(double goal_x, double goal_y)
    {
        try
        {
            geometry_msgs::msg::PointStamped stamped_point;
            stamped_point.header.stamp = this->get_clock()->now();
            stamped_point.header.frame_id = "camera_link";
            stamped_point.point.x = goal_x;
            stamped_point.point.y = goal_y;
            stamped_point.point.z = 0.0;

            geometry_msgs::msg::TransformStamped transform =
                tf_buffer_->lookupTransform("odom", "camera_link", tf2::TimePointZero, tf2::durationFromSec(0.5));
            geometry_msgs::msg::PointStamped transformed_point;
            tf2::doTransform(stamped_point, transformed_point, transform);

            std::cout << transformed_point.point.x << " , " << transformed_point.point.y << " is the goal\n";
            return transformed_point;
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Transform failed from camera_link to odom: %s", ex.what());
            return std::nullopt;
        }
    }

    void publish_goal(const geometry_msgs::msg::PointStamped& goal_point){
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
        if(override_ == "none"){
            goal_pub_->publish(goal_pose);
        }
    }

    void debug_markers(){
        visualization_msgs::msg::MarkerArray MarkerArray;
        int marker_id = 0;
        rclcpp::Time timestamp = this->get_clock()->now();

        auto make_marker = [&marker_id, &timestamp](std::pair<double, double> point, const std::array<float, 3>& color, const std::string& label){
            visualization_msgs::msg::Marker m;
            m.header.stamp = timestamp;
            m.header.frame_id = "odom";
            m.ns = label;
            m.id = marker_id;
            marker_id += 1;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = point.first;
            m.pose.position.y = point.second;
            m.pose.position.z = 0.1;
            m.pose.orientation.w = 1.0;
            m.scale.x = 0.3;
            m.scale.y = 0.3;
            m.scale.z = 0.3;
            m.color.r = color[0];
            m.color.g = color[1];
            m.color.b = color[2];
            m.color.a = 1.0;
            return m;
        };
        MarkerArray.markers.push_back(make_marker(history_[0].right, {1.0f,  0.0f, 0.0f}, "right_point"));
        MarkerArray.markers.push_back(make_marker(history_[0].mid, {0.0f,  1.0f, 0.0f}, "mid_point"));
        MarkerArray.markers.push_back(make_marker(history_[0].left, {0.0f,  0.0f, 1.0f}, "left_point"));

        debug_pub_->publish(MarkerArray);
    }

    void apply_ema(std::pair<double, double>& rp, std::pair<double, double>& mp, std::pair<double, double>& lp){

        double alpha = 0.1;

        cout<< " rp****: "<<rp.first<<", "<<rp.second;
        cout<< " mp****: "<<mp.first<<", "<<mp.second;
        cout<< " lp****: "<<lp.first<<", "<<lp.second;
        if(history_.size() < 2) return;

        rp = history_[0].right * alpha + history_[1].right * (1 - alpha);
        mp = history_[0].mid   * alpha + history_[1].mid   * (1 - alpha);
        lp = history_[0].left  * alpha + history_[1].left  * (1 - alpha);
    }

    void marker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
    {
        std::vector<std::pair<double, double>> end_points; // vector of markers with marker and its end point
        std::pair<double, double> mp;
        for (const auto &marker : msg->markers)
        {
            std::pair<double, double> pair = get_last_point(marker.points);
            if(pair.first == 0.0 && pair.second == 0.0){continue;}
            if(marker.id == 10){
                mp = pair;
            }
            else{
                end_points.emplace_back(pair);
            }
        }

        std::sort(end_points.begin(), end_points.end(), [](const auto &a, const auto &b)
                  { return a.second < b.second; }); // second second is for sorting by y. Also it sorts in increasing order of y

        std::pair<double, double> rp, lp;
        double goal_x, goal_y;

        cout << "size of points array: "<<end_points.size() << '\n';

        cout<<'\n';
        for(auto i : end_points){
            cout<<"( "<<i.first<<", "<<i.second<<" )";
        }
        cout<<'\n';

        if (end_points.size() == 2)
        {
            rp = end_points[0];
            lp = end_points[1];

            if (target_lane_ == "right")
            {
                goal_x = (rp.first + mp.first) / 2.0;
                goal_y = (rp.second + mp.second) / 2.0;
                current_lane_ = "right";
            }
            else
            {
                goal_x = (lp.first + mp.first) / 2.0;
                goal_y = (lp.second + mp.second) / 2.0;
                current_lane_ = "left";
            }
            cout<< " rp: "<<rp.first<<", "<<rp.second;
            cout<< " mp: "<<mp.first<<", "<<mp.second;
            cout<< " lp: "<<lp.first<<", "<<lp.second;

            geometry_msgs::msg::PointStamped olp = *transform_to_odom(lp.first, lp.second);
            geometry_msgs::msg::PointStamped omp = *transform_to_odom(mp.first, mp.second);
            geometry_msgs::msg::PointStamped orp = *transform_to_odom(rp.first, rp.second);

            history_.push_front(tracked_points{.left = {olp.point.x, olp.point.y}, .mid = {omp.point.x, omp.point.y}, .right = {orp.point.x, orp.point.y}});

            while(history_.size() > buffer_size_){
                history_.pop_back();
            }
            cout << "\nGoal: " << goal_x << ", " << goal_y << '\n';

            if (auto transformed = transform_to_odom(goal_x, goal_y)) {
                debug_markers();
                publish_goal(*transformed);
            }
        }
        else if(end_points.size() < 2){
            apply_ema(rp, mp, lp);

            if (target_lane_ == "right")
            {
                goal_x = (rp.first + mp.first) / 2.0;
                goal_y = (rp.second + mp.second) / 2.0;
                current_lane_ = "right";
            }
            else
            {
                goal_x = (lp.first + mp.first) / 2.0;
                goal_y = (lp.second + mp.second) / 2.0;
                current_lane_ = "left";
            }
            cout<< " rp: "<<rp.first<<", "<<rp.second;
            cout<< " mp: "<<mp.first<<", "<<mp.second;
            cout<< " lp: "<<lp.first<<", "<<lp.second;

            history_.push_front(tracked_points{.left = {lp.first, lp.second}, .mid = {mp.first, mp.second}, .right = {rp.first, rp.second}});

            while(history_.size() > buffer_size_){
                history_.pop_back();
            }
            geometry_msgs::msg::PoseStamped goal_pose;
            goal_pose.header.stamp = this->get_clock()->now();
            goal_pose.header.frame_id = "odom";
            goal_pose.pose.position.x = goal_x;
            goal_pose.pose.position.y = goal_y;
            goal_pose.pose.position.z = 0.0;
            goal_pose.pose.orientation.x = 0.0;
            goal_pose.pose.orientation.y = 0.0;
            goal_pose.pose.orientation.z = 0.0;
            goal_pose.pose.orientation.w = 1.0;

            if(override_ == "none"){
                goal_pub_->publish(goal_pose);
            }
        }
        else{
            return;
        }

    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GoalPublisher>());
    rclcpp::shutdown();
    return 0;
}
