/*
 * This program finds the inverse transform from a specified frame
 * (in the argument) to the base_link of the robot
 *
 * Do what you will with the info.
 *
 */
#include <string>
#include <cmath>
#include <iostream>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

double atan2(double x, double y){
	double ret = atan(x/y);
	if((x < 0 && y < 0) || (x < 0 && y > 0)){
		return ret + M_PI;
	}
	return ret;
}
class InvTransformLookup: public rclcpp::Node
{
  public:
    InvTransformLookup(std::string fromFrameRel)
    : Node("inv_transform_lookup")
{
	this->set_parameter(rclcpp::Parameter("use_sim_time", false));
	tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
	tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
	while(true){
		std::string toFrameRel = "base_link";
		try 
		{
			t = tf_buffer_->lookupTransform(
					fromFrameRel, toFrameRel,
					tf2::TimePointZero);
			RCLCPP_INFO(this->get_logger(), "transform received!");
			break;
		} 
		catch (const tf2::TransformException &ex) 
		{
			RCLCPP_INFO(
				this->get_logger(), "ERROR: %s, transform from %s to %s not available", 
				ex.what(), toFrameRel.c_str(), "zed_camera_link");
		}
	}
	double w = t.transform.rotation.w;
	double x = t.transform.rotation.x;
	double y = t.transform.rotation.y;
	double z = t.transform.rotation.z;

	double sinr_cosp = 2 * (w * x + y * z);
	double cosr_cosp = 1 - 2 * (x * x + y * y);
	double roll = atan2(sinr_cosp, cosr_cosp);

	double sinp = 2 * (w * y - z * x);
	double pitch = asin(sinp);

	double siny_cosp = 2 * (w * z + x * y);
	double cosy_cosp = 1 - 2 * (y * y + z * z);
	double yaw = atan2(siny_cosp,cosy_cosp);
	RCLCPP_INFO( this->get_logger(), "In the order [X, Y, Z, R, P, Y]: \n'%f', '%f', '%f', '%f', '%f', '%f'\n\n",
		t.transform.translation.x, t.transform.translation.y, t.transform.translation.z, roll, pitch, yaw);
	RCLCPP_INFO( this->get_logger(), "In the order [X, Y, Z, R, P, Y]: \n[%f, %f, %f, %f, %f, %f]",
		t.transform.translation.x, t.transform.translation.y, t.transform.translation.z, roll, pitch, yaw);
}
	
  private:
	std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
	std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
	geometry_msgs::msg::TransformStamped t;
};

int main(int argc, char * argv[])
{
	if(argc < 2){
		std::cerr << "Please give the frame FROM which you want to the transform of base link"<< std::endl;
		return 0;
	}
	rclcpp::init(argc, argv);
	std::string from_frame = argv[1];
	rclcpp::spin(std::make_shared<InvTransformLookup>(from_frame));
	rclcpp::shutdown();
	return 0;
}
