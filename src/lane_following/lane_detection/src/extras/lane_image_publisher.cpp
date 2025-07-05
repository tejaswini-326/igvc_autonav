#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"


class PointCloudToIPMImage : public rclcpp::Node {
public:
    PointCloudToIPMImage() : Node("pointcloud_to_ipm_image") {
        // Subscriber to PointCloud2
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lane_detection/largest_cluster", 10, std::bind(&PointCloudToIPMImage::pointcloudCallback, this, std::placeholders::_1));
        
        // Publisher for IPM image
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/ipm_image", 10);
    }

private:
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        // Create a black image with desired size
        int rows = 480; // Change based on your use case
        int cols = 640;
        cv::Mat ipm_image = cv::Mat::zeros(rows, cols, CV_8UC1);


        // Loop through the points in the point cloud
        for (const auto& point : cloud->points) {
            // Skip invalid points
            if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
                continue;

            // Scale and translate to fit within the image
            int px = static_cast<int>(rows - point.x * 100);  // Scale and invert
            int py = static_cast<int>(cols / 2 + point.y * 100); // Center horizontally

            // Check bounds and set pixel in the image
            if (px >= 0 && px < rows && py >= 0 && py < cols) {
                ipm_image.at<uchar>(px, py) = 255;
            }
        }
        // std::cout<<ipm_image<<std::endl;
        // cv::imshow("ipm_image",ipm_image);
        cv::waitKey(10);
        // Convert OpenCV Mat to ROS Image message
        sensor_msgs::msg::Image::SharedPtr ipm_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", ipm_image).toImageMsg();
        // ipm_msg->header.stamp = this->now();
        // ipm_msg->header.frame_id = "ipm_frame";

        // Publish the IPM image
        image_pub_->publish(*ipm_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudToIPMImage>());
    rclcpp::shutdown();
    return 0;
}
