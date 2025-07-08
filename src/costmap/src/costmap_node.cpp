// Description: Node that creates and publishes a costmap of the bot's surroundings in C++

#include <memory>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

using std::placeholders::_1;
using namespace std::chrono_literals;

class CostmapNode : public rclcpp::Node {
public:
    CostmapNode() : Node("costmap_node"),
                    tf_buffer_(this->get_clock()),
                    tf_listener_(tf_buffer_) {

        resolution_ = 0.067;
        width_ = 300;
        height_ = 300;
        frame_id_ = "odom";

        origin_x_ = 0.0;
        origin_y_ = 0.0;

        empty_layer_ = cv::Mat::zeros(height_, width_, CV_8UC1);
        white_map_ = empty_layer_.clone();
        yellow_map_ = empty_layer_.clone();
        object_map_ = empty_layer_.clone();

        costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/costmap", 10);

        white_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/white_lane_points", 10, std::bind(&CostmapNode::whiteCallback, this, _1));

        yellow_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/yellow_lane_points", 10, std::bind(&CostmapNode::yellowCallback, this, _1));

        object_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/object_pc", 10, std::bind(&CostmapNode::objectCallback, this, _1));

        timer_ = this->create_wall_timer(150ms, std::bind(&CostmapNode::timerCallback, this));
    }

private:
    double resolution_;
    int width_, height_;
    double origin_x_, origin_y_;
    std::string frame_id_;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr white_sub_, yellow_sub_, object_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    geometry_msgs::msg::TransformStamped cached_transform_;

    cv::Mat empty_layer_, white_map_, yellow_map_, object_map_;
    sensor_msgs::msg::PointCloud2::SharedPtr latest_white_, latest_yellow_, latest_object_;
    bool new_white_ = false, new_yellow_ = false, new_object_ = false;

    void whiteCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        latest_white_ = msg;
        new_white_ = true;
    }

    void yellowCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        latest_yellow_ = msg;
        new_yellow_ = true;
    }

    void objectCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        latest_object_ = msg;
        new_object_ = true;
    }

    void timerCallback() {
        try {
            cached_transform_ = tf_buffer_.lookupTransform("odom", "camera_link", tf2::TimePointZero);
            origin_x_ = cached_transform_.transform.translation.x - (width_ * resolution_) / 2.0;
            origin_y_ = cached_transform_.transform.translation.y - (height_ * resolution_) / 2.0;
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return;
        }

        if (new_white_ && latest_white_) {
            white_map_ = generateCostmap(latest_white_, "white");
            new_white_ = false;
        }
        if (new_yellow_ && latest_yellow_) {
            yellow_map_ = generateCostmap(latest_yellow_, "yellow");
            new_yellow_ = false;
        }
        if (new_object_ && latest_object_) {
            object_map_ = generateCostmap(latest_object_, "object");
            new_object_ = false;
        }

        cv::Mat combined;
        cv::max(white_map_, yellow_map_, combined);
        cv::max(combined, object_map_, combined);

        publishCostmap(combined);
    }

    cv::Mat generateCostmap(const sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg, const std::string & tag) {
        sensor_msgs::msg::PointCloud2 transformed;
        try {
            tf2::doTransform(*cloud_msg, transformed, cached_transform_);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "PointCloud transform failed: %s", ex.what());
            return empty_layer_.clone();
        }

        cv::Mat layer = cv::Mat::zeros(height_, width_, CV_8UC1);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(transformed, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(transformed, "y");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) {
            float x = *iter_x;
            float y = *iter_y;

            int mx = static_cast<int>((x - origin_x_) / resolution_);
            int my = static_cast<int>((y - origin_y_) / resolution_);

            if (mx >= 0 && mx < width_ && my >= 0 && my < height_ && my > 100) {
                uint8_t value = (tag == "object") ? 100 : 250;
                if (layer.at<uchar>(my, mx) < 255) {
                    layer.at<uchar>(my, mx) = value;
                }
            }
        }

        if (cv::countNonZero(layer) == 0)
            return empty_layer_.clone();

        cv::Mat float_layer;
        layer.convertTo(float_layer, CV_32F);
        cv::Mat blurred;
        cv::GaussianBlur(float_layer, blurred, cv::Size(15, 15), 2.3);

        double max_val;
        cv::minMaxLoc(blurred, nullptr, &max_val);

        cv::Mat scaled = cv::Mat::zeros(layer.size(), CV_8UC1);
        if (max_val > 0) {
            float power = (tag == "object") ? 2.0f : 0.8f;
            cv::Mat temp;
            blurred = blurred / std::pow(max_val, power) * 100.0;
            blurred.convertTo(scaled, CV_8UC1);
        }

        cv::Mat result;
        cv::max(layer, scaled, result);
        return result;
    }

    void publishCostmap(const cv::Mat & map) {
        nav_msgs::msg::OccupancyGrid msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = frame_id_;

        msg.info.resolution = resolution_;
        msg.info.width = width_;
        msg.info.height = height_;
        msg.info.origin.position.x = origin_x_;
        msg.info.origin.position.y = origin_y_;
        msg.info.origin.orientation.w = 1.0;

        msg.data.resize(width_ * height_);
        for (int i = 0; i < height_; ++i) {
            for (int j = 0; j < width_; ++j) {
                msg.data[i * width_ + j] = static_cast<int8_t>((map.at<uchar>(i, j) / 255.0) * 100);
            }
        }

        costmap_pub_->publish(msg);
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CostmapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
