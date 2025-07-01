#include <rclcpp/rclcpp.hpp>
#include <unistd.h>
#include <string>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using namespace pcl;
using namespace std;

#define RING_RADIUS 20


class PointCloudPrinter : public rclcpp::Node
{
public:

    PointCloudPrinter()
        : Node("point_cloud_printer"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/points",
            10,
            std::bind(&PointCloudPrinter::pointCloudCallback, this, std::placeholders::_1));

        publisher_bl = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lane_point_cloud_bl", 10);

		this->declare_parameter<float>("bt_low", 78.0);
		this->declare_parameter<float>("bt_high", 171.0);
		this->declare_parameter<float>("z_thresh", 0.4);
		this->declare_parameter<float>("yellow_r_min", 82.000);
		this->declare_parameter<float>("yellow_r_max", 255.000);
		this->declare_parameter<float>("yellow_g_min", 95.000);
		this->declare_parameter<float>("yellow_g_max", 255.000);
		this->declare_parameter<float>("yellow_b_min", 0.000);
		this->declare_parameter<float>("yellow_b_max", 102.000);

		while(true)
		{
			// We do not start until we get the transform
			try {
            // Wait for the transform to be available
            if (tf_buffer_.canTransform("base_link", "camera_link", tf2::TimePointZero, tf2::durationFromSec(1.0))) {
                transform_stamped = tf_buffer_.lookupTransform("base_link", "camera_link", tf2::TimePointZero);
                RCLCPP_INFO(this->get_logger(), "Transform lookup successful");
                break;
            } else {
                RCLCPP_WARN(this->get_logger(), "Transform not available yet, retrying...");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Transform exception: %s", ex.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
		}
	}

private:
    double bt_low_, bt_high_, z_thresh_, ring_radius_, yellow_r_min_, yellow_r_max_, yellow_b_min_, yellow_b_max_, yellow_g_min_, yellow_g_max_;
    std::mutex param_mutex_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    geometry_msgs::msg::TransformStamped transform_stamped;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_bl;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

#if defined(PERF_TESTING)

	 __inline__ void TIMEIT(std::string msg)
	 {
		 static std::chrono::time_point prev = std::chrono::high_resolution_clock::now(); 			
		 static std::chrono::duration<double> dur(0); 

		 dur = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - prev);
		 prev = std::chrono::high_resolution_clock::now(); 			
		 RCLCPP_INFO(this->get_logger(), "%f: %s\n", dur.count(), msg);
	 }
#else

	 __inline__ void TIMEIT(std::string msg){;}

#endif

    uint8_t convertRGBtoGray(uint8_t r, uint8_t b, uint8_t g) {
		return 0.299*r + 0.587*g + 0.114*b;
		
		//uint8_t tmp_min = min(r, b)
		//uint8_t tmp_max = max(r, b)
		//uint8_t Cmin = min(tmp_min, g)
		//uint8_t Cmax = max(tmp_max, g)		
		//if Cmax=
	}

    pcl::PointXYZI transformToBaseLink(pcl::PointXYZI point, const geometry_msgs::msg::TransformStamped &transform_stamped) 
    {
        geometry_msgs::msg::PointStamped point_in, point_out;
        // point_in.header.frame_id = "base_link";
        point_in.point.x = point.x;
        point_in.point.y = point.y;
        point_in.point.z = point.z;

        tf2::doTransform(point_in, point_out, transform_stamped);

        // Update the point cloud with transformed points
        point.x = point_out.point.x;
        point.y = point_out.point.y;
        point.z = point_out.point.z;
        return point;
    }

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {

        TIMEIT(__FUNCTION__);
        bt_low_ = this->get_parameter("bt_low").as_double();
        bt_high_ = this->get_parameter("bt_high").as_double();
		z_thresh_ = this->get_parameter("z_thresh").as_double();
        yellow_r_min_ = this->get_parameter("yellow_r_min").as_double();
        yellow_r_max_ = this->get_parameter("yellow_r_max").as_double();
		yellow_g_min_ = this->get_parameter("yellow_g_min").as_double();
        yellow_g_max_ = this->get_parameter("yellow_g_max").as_double();
        yellow_b_min_ = this->get_parameter("yellow_b_min").as_double();
		yellow_b_max_ = this->get_parameter("yellow_b_max").as_double();
	
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        pcl::fromROSMsg(*msg, cloud);

        // Create a new point cloud for publishing
        pcl::PointCloud<pcl::PointXYZI> filtered_cloud, base_link_cloud;

         TIMEIT("created pointcloud");

        // Process each point and append to the 
		// filtered cloud if it meets the criteria
        double low, high, zthresh, radius, y_r_min, y_r_max, y_b_min, y_b_max, y_g_min, y_g_max;
        {
            std::lock_guard<std::mutex> lock(param_mutex_);
            low = bt_low_;
            high = bt_high_;
            zthresh = z_thresh_;
            radius = ring_radius_;
            y_r_min = yellow_r_min_;
            y_r_max = yellow_r_max_;
            y_g_min = yellow_g_min_;
            y_g_max = yellow_g_max_;
            y_b_min = yellow_b_min_;
            y_b_max = yellow_b_max_;
        }
        for (auto &point : cloud.points)
        {
            float x = point.x, y = point.y, z = point.z;
            std::uint32_t rgb = *reinterpret_cast<int *>(&point.rgb);
            uint8_t r = (rgb >> 16) & 0xFF;
            uint8_t g = (rgb >> 8) & 0xFF;
            uint8_t b = rgb & 0xFF;

            if ((r >= low) && (g >= low) && (b >= low) &&
                (r <= high) && (g <= high) && (b <= high))
            {
                pcl::PointXYZI filtered_point = {x, y, z};
                pcl::PointXYZI pbl = transformToBaseLink(filtered_point, transform_stamped);
                pbl.intensity = 100.0f; //white points have 100 intensity

                if (fabs(pbl.z) < zthresh && pbl.x * pbl.x + pbl.y * pbl.y < RING_RADIUS * RING_RADIUS)
                {
                    base_link_cloud.push_back(pbl);
                }
            }

            else if ((r >= y_r_min) && (g >= y_g_min) && (b >= y_b_min) &&
                (r <= y_r_max) && (g <= y_g_max) && (b <= y_b_max))
            {
                pcl::PointXYZI filtered_point = {x, y, z};
                pcl::PointXYZI pbl = transformToBaseLink(filtered_point, transform_stamped);
                pbl.intensity = 200.0f; //yellow points have 200 intensity

                if (fabs(pbl.z) < zthresh && pbl.x * pbl.x + pbl.y * pbl.y < RING_RADIUS * RING_RADIUS)
                {
                    base_link_cloud.push_back(pbl);
                }
            }

        }

        TIMEIT("filtering done");


        // Convert the filtered cloud back to a ROS message
        // sensor_msgs::msg::PointCloud2 output;
        sensor_msgs::msg::PointCloud2 output_baselink;
        // pcl::toROSMsg(filtered_cloud, output);
        pcl::toROSMsg(base_link_cloud, output_baselink);


        // output.header.frame_id = msg->header.frame_id;
        output_baselink.header.frame_id = "base_link";

        TIMEIT("created message");
	
        // Publish the filtered point cloud
		//publisher_->publish(output);
        RCLCPP_INFO(this->get_logger(), "output has %zu points", base_link_cloud.size());
        publisher_bl->publish(output_baselink);

        TIMEIT("sent message");
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudPrinter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
