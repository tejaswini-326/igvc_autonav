/*
 *@Author: Krutarth 
 *@Date: 20th April 2025
 *@Description : Cluster visualizer
 */

#include "rclcpp/rclcpp.hpp"
#include "vizhi_interfaces/msg/clusters.hpp"
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <queue>
#include <exception>
#include <string>

using namespace std;
using std::placeholders::_1;
typedef vizhi_interfaces::msg::Clusters::SharedPtr vizhi_cl_ptr;
typedef vizhi_interfaces::msg::Clusters vizhi_cl_t;
typedef sensor_msgs::msg::PointCloud2 sensor_pc_t;
typedef rclcpp::Publisher<sensor_pc_t>::SharedPtr lane_pub_ptr;
typedef pcl::PointCloud<pcl::PointXYZ> pc_t;
typedef pcl::PointXYZRGB rgbpt_t;
typedef pcl::PointCloud<rgbpt_t> rgbpc_t;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pc_ptr;

#include <vector>
#include <string>

const std::vector<int> COLOR= {
    0x000000, 0x00FF00, 0x0000FF, 0xFF0000, 0x01FFFE, 0xFFA6FE, 0xFFDB66, 0x006401, 0x010067, 0x95003A,
    0x007DB5, 0xFF00F6, 0xFFEEE8, 0x774D00, 0x90FB92, 0x0076FF, 0xD5FF00, 0xFF937E, 0x6A826C, 0xFF029D,
    0xFE8900, 0x7A4782, 0x7E2DD2, 0x85A900, 0xFF0056, 0xA42400, 0x00AE7E, 0x683D3B, 0xBDC6FF, 0x263400,
    0xBDD393, 0x00B917, 0x9E008E, 0x001544, 0xC28C9F, 0xFF74A3, 0x01D0FF, 0x004754, 0xE56FFE, 0x788231,
    0x0E4CA1, 0x91D0CB, 0xBE9970, 0x968AE8, 0xBB8800, 0x43002C, 0xDEFF74, 0x00FFC6, 0xFFE502, 0x620E00,
    0x008F9C, 0x98FF52, 0x7544B1, 0xB500FF, 0x00FF78, 0xFF6E41, 0x005F39, 0x6B6882, 0x5FAD4E, 0xA75740,
    0xA5FFD2, 0xFFB167, 0x009BFF, 0xE85EBE
};

class ClusterViz: public rclcpp::Node {
public:
    ClusterViz() : Node("cluster_viz") 
    {
        this->declare_parameter<std::string>("topic", "/clusters");
        subscription = this->create_subscription<vizhi_cl_t>(
            "/clusters", 10, 
            std::bind(&ClusterViz::visualize, this, std::placeholders::_1));
        clusters_publisher=this->create_publisher<sensor_pc_t>("/cluster_viz", 10); 
        std::cout << "Initialized!" << std::endl;
    }
private:
    void publish_cloud(lane_pub_ptr publisher, rgbpc_t &pc_msg, std_msgs::msg::Header header)
    {
        sensor_msgs::msg::PointCloud2 cluster;
        pcl::toROSMsg(pc_msg, cluster);
        cluster.header = header;
        publisher->publish(cluster);
    }

    void visualize(const vizhi_cl_t &msg) 
    {
        std::vector<pc_t> clusters;
        std_msgs::msg::Header header;
        for(auto cl: msg.clusters){
            header = cl.header;
            clusters.push_back(pc_t());
            pcl::fromROSMsg(cl, clusters.back()); 
        }
        if(clusters.empty())
        {
            rgbpc_t emp;
            publish_cloud(clusters_publisher, emp, std_msgs::msg::Header());
            return;
        }
        std::cout << "number of clusters: " << clusters.size() << std::endl;
        rgbpc_t cloud;
        int color_ind =0;
        for(auto pcld: clusters){
            for(auto &point: pcld){
                uint8_t r = (COLOR[color_ind] >> 16) & 0xFF;
                uint8_t g = (COLOR[color_ind] >> 8) & 0xFF;
                uint8_t b = (COLOR[color_ind]) & 0xFF;
                rgbpt_t pt(point.x,point.y,point.z,r,g,b);
                cloud.points.push_back(pt);
            }
            color_ind++;
            color_ind %= COLOR.size();
        }
        publish_cloud(clusters_publisher, cloud, header);
    }

    rclcpp::Subscription<vizhi_cl_t>::SharedPtr subscription;
    lane_pub_ptr clusters_publisher;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ClusterViz>());
    rclcpp::shutdown();
    return 0;
}
