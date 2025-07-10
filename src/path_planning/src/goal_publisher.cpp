#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>

using std::placeholders::_1;
using namespace std;
typedef geometry_msgs::msg::Point pt;
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

        target_lane_ = "right";
        current_lane_ = "right";
        RCLCPP_INFO(this->get_logger(), "GoalPublisher node initialized");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr marker_sub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::string target_lane_;
    std::string current_lane_;
    std::pair<double, double> last_rp_, last_lp_, last_mp_;

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

        goal_pub_->publish(goal_pose);
    }

    void debug_markers(){
        visualization_msgs::msg::MarkerArray MarkerArray;
        int marker_id = 0;
        rclcpp::Time timestamp = this->get_clock()->now();

        auto make_marker = [&marker_id, &timestamp](std::pair<double, double> point, const std::array<float, 3>& color, const std::string& label){
            visualization_msgs::msg::Marker m;
            m.header.stamp = timestamp;
            m.header.frame_id = "camera_link";
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
        MarkerArray.markers.push_back(make_marker(this->last_rp_, {1.0f,  0.0f, 0.0f}, "right_point"));
        MarkerArray.markers.push_back(make_marker(this->last_mp_, {0.0f,  1.0f, 0.0f}, "mid_point"));
        MarkerArray.markers.push_back(make_marker(this->last_lp_, {0.0f,  0.0f, 1.0f}, "left_point"));

        debug_pub_->publish(MarkerArray);
    }

    void marker_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
    {
        std::vector<visualization_msgs::msg::Marker> lane_markers;
        for (const auto &marker : msg->markers)
        {
            lane_markers.push_back(marker);
        }

        if (lane_markers.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No lane markers found");
        }

        std::vector<std::pair<visualization_msgs::msg::Marker, std::pair<double, double>>> end_points; // vector of markers with marker and its end point
        for (const auto &marker : lane_markers)
        {
            std::pair<double, double> pair = get_last_point(marker.points);
            if(pair.first == 0.0 && pair.second == 0.0){continue;}
            end_points.emplace_back(marker, get_last_point(marker.points));
        }

        std::sort(end_points.begin(), end_points.end(), [](const auto &a, const auto &b)
                  { return a.second.second < b.second.second; }); // second second is for sorting by y. Also it sorts in increasing order of y

        std::pair<double, double> rp, lp, mp;
        double goal_x, goal_y;

        cout << "size of points array: "<<end_points.size() << '\n';

        if (end_points.size() >= 3)
        {
            rp = (end_points[0].second.first == 0.0 && end_points[0].second.second == 0.0) ? last_rp_ : end_points[0].second;
            mp = (end_points[1].second.first == 0.0 && end_points[1].second.second == 0.0) ? last_mp_ : end_points[1].second;
            lp = (end_points[2].second.first == 0.0 && end_points[2].second.second == 0.0) ? last_lp_ : end_points[2].second;

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
            last_rp_ = rp;
            last_lp_ = lp;
            last_mp_ = mp;
        }
        else if(end_points.size() == 2){
            if(current_lane_ == "right"){
                rp = (end_points[0].second.first == 0.0 && end_points[0].second.second == 0.0) ? last_rp_ : end_points[0].second;
                mp = (end_points[1].second.first == 0.0 && end_points[1].second.second == 0.0) ? last_mp_ : end_points[1].second;
                double dx = rp.first - mp. first;
                double dy = rp.second - mp.second;
                lp = {mp.first - dx, mp.second - dy};
            }
            else{
                mp = (end_points[0].second.first == 0.0 && end_points[0].second.second == 0.0) ? last_mp_ : end_points[0].second;
                lp = (end_points[1].second.first == 0.0 && end_points[1].second.second == 0.0) ? last_lp_ : end_points[1].second;
                double dx = lp.first - mp.first;
                double dy = lp.second - mp.second;
                rp = {mp.first - dx, mp.second - dy};
            }

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
            last_rp_ = rp;
            last_lp_ = lp;
            last_mp_ = mp;
        }
        else{
            return;
        }

        cout << "\n***************\n";
        cout << goal_x << goal_y << '\n';

        if (auto transformed = transform_to_odom(goal_x, goal_y)) {
            debug_markers();
            publish_goal(*transformed);
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
