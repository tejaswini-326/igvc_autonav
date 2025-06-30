/*
 *@Author: Krutarth 
 *@Date: 20th December 2024
 *@Description : separate lanes from
 *thresholded pointcloud
 *using GPU-DBSCAN
 */

//#include "dbscan-again.cuh"
#include "dbscan-again.cuh"
#include "rclcpp/rclcpp.hpp"
#include "vizhi_interfaces/msg/clusters.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/float64.hpp"
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
#include <chrono>

using namespace std;
using std::placeholders::_1;
typedef vizhi_interfaces::msg::Clusters::SharedPtr vizhi_cl_ptr;
typedef vizhi_interfaces::msg::Clusters vizhi_cl_t;
typedef rclcpp::Publisher<vizhi_cl_t>::SharedPtr lane_pub_ptr;
typedef sensor_msgs::msg::PointCloud2 sensor_pc_t;
typedef pcl::PointCloud<pcl::PointXYZI> pc_t;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr;

#define MIN_SIZE 19    // min points to consider a cluster

class LaneFollower : public rclcpp::Node {
public:
    LaneFollower() : Node("lane_follower") {   
        this->declare_parameter("eps", 1.506);
        this->declare_parameter("min_pts", 1);
        RCLCPP_INFO(this->get_logger(), "cluster node initialized");

        //double eps = atof(getenv("ABH_EPS"));
        //int min_pts = atoi(getenv("ABH_MIN_PTS"));
        double eps = this->get_parameter("eps").as_double();
        int min_pts = this->get_parameter("min_pts").as_int();

        RCLCPP_INFO(this->get_logger(), "eps: %f, min_pts: %d", eps, min_pts);

        if(min_pts==0 || eps==0){
            // most likely the bash variables 
            // are not defined!
            RCLCPP_ERROR(this->get_logger(), "either min_pts or eps is zero, \
            do you know what you are doing? \n make sure ABH_EPS and  \
            ABH_MIN_PTS are defined and exported");
        }

        std::cout << min_pts << std::endl;

        subscription = this->create_subscription<sensor_pc_t>(
            "/lane_point_cloud_bl", 10, 
            std::bind(&LaneFollower::cluster, this, std::placeholders::_1));

        clusters_publisher=this->create_publisher<vizhi_cl_t>("/clusters", 10); 
        //largest_cluster_publisher=this->create_publisher<vizhi_cl_ptr>("/largest_cluster", 10); 
        //second_largest_cluster_publisher=this->create_publisher<vizhi_cl_ptr>("/second_largest_cluster", 10);
        scanner = std::make_unique<DBSCAN<double>>(eps, min_pts);
        cluster_msg = std::make_unique<vizhi_cl_t>();
}
private:
    void publish_cloud(lane_pub_ptr publisher, std::vector<pc_t> &pc_msg, const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        sensor_msgs::msg::PointCloud2 cluster;
        int ind = 0;
        for(auto cls: pc_msg){
            pcl::toROSMsg(cls, cluster);
            cluster.header = msg->header;
            // we reuse the memory in this way
            // should be faster
            if(ind == cluster_msg->clusters.size()){
                cluster_msg->clusters.push_back(cluster);
            }else{
                cluster_msg->clusters[ind] = cluster;
            }
            ind++;
        }
        cluster_msg->clusters.resize(ind);
        publisher->publish(*cluster_msg);
    }

    //void publish_cloud(lane_pub_ptr publisher, pc_ptr pc_msg, const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    //{
        //sensor_msgs::msg::PointCloud2 largest_cluster_msg;
        //pcl::toROSMsg(*pc_msg, largest_cluster_msg);
        //largest_cluster_msg.header = msg->header;
        //publisher->publish(largest_cluster_msg);

    //}

    void cluster(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        pc_ptr cloud(new pc_t());
        pcl::fromROSMsg(*msg, *cloud); 
        if(cloud->empty())
        {
            std::vector<pc_t> clusters = {*cloud};
            publish_cloud(clusters_publisher, clusters, msg);
            return;
        }
        // flattened array
        // NOTE: massive performance
        // increase by just sorting 
        std::vector<double> white_pixel_indices(2 * cloud->points.size());
        //sort(cloud->points.begin(), cloud->points.end(), [](pcl::PointXYZ a, pcl::PointXYZ b){
                        //return a.x < b.x;
                        //});
        for (int i=0;i < cloud->points.size(); ++i)
        {
            auto point = cloud->points[i];
            white_pixel_indices[2*i] = point.x;
            white_pixel_indices[2*i+1] = point.y; 
        }
        auto cpu_start_time = std::chrono::high_resolution_clock::now();
        // there is a hard limit on the number of points,
        // actual size returns the number that DBSCAN used
        // for clustering
        __START_TIMER__
        int num_clusters = 0;
        int num_points = scanner->identify_cluster(white_pixel_indices, num_clusters);

        std::cout << cloud->points.size() << ' ';
        __END_TIMER__
        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_time - cpu_start_time);
        std::cout << "[CLUSTER] PointCloud size: " << cloud->points.size()
              << ", Total clustering time (host + device): " << cpu_duration.count() << " ms" << std::endl;

        //debug
#if defined(DEBUG)
        scanner->show_labels();
#endif
        std::vector<size_t> assignments(num_points);
        for (int i = 0; i < assignments.size(); ++i) {
                assignments[i] = scanner->label(i);
        }
        std::unordered_map<size_t, size_t> clusterSizes;
        for (size_t i = 0; i < assignments.size(); ++i) {
                if (assignments[i] != SIZE_MAX) {
                        clusterSizes[assignments[i]]++;
                }
        }
        std::vector<std::pair<size_t, size_t>> sortedClusters(clusterSizes.begin(), clusterSizes.end());
        auto cmp = [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                      return b.second < a.second; 
              };
        std::sort(sortedClusters.begin(), sortedClusters.end(), cmp);
        if (sortedClusters.size() < 2) {
            RCLCPP_INFO(this->get_logger(), "not enough clusters");
            return;
        }
        //__END_TIMER__
        // get the last clusters which is just not big enough
        auto clust_end = std::upper_bound(sortedClusters.begin(),sortedClusters.end(),
                                          std::pair<size_t,size_t>(0, MIN_SIZE),cmp);
        int nclust = clust_end - sortedClusters.begin();
        std::vector<pc_t> clusters(nclust);
        pc_t yellow_cluster;
        pcl::PointXYZI p;

        for (size_t i = 0; i < assignments.size(); ++i) {
            p.x = white_pixel_indices[2 * i];
            p.y = white_pixel_indices[2 * i + 1];
            p.z = 0;

            // Get original intensity of this point
            float intensity = cloud->points[i].intensity;
            p.intensity = intensity;

            if (intensity == 200.0f) {
                yellow_cluster.points.push_back(p);  // all yellow points grouped
                continue;
            }

            // Add non-yellow point to its cluster as usual
            for (int j = 0; j < nclust; ++j) {
                if (sortedClusters[j].first == assignments[i]) {
                    clusters[j].points.push_back(p);
                    break;
                }
            }
        }

        if (!yellow_cluster.empty()) {
            clusters.push_back(yellow_cluster);  // one cluster for all yellow points
        }
        
        //__END_TIMER__
        std::cout << "number of clusters: " << nclust << std::endl;
        publish_cloud(clusters_publisher, clusters, msg);
    }


    rclcpp::Subscription<sensor_pc_t>::SharedPtr subscription;
    lane_pub_ptr clusters_publisher;
    //lane_pub_ptr largest_cluster_publisher;
    //lane_pub_ptr second_largest_cluster_publisher;
    std::unique_ptr<DBSCAN<double>> scanner;
    std::unique_ptr<vizhi_cl_t> cluster_msg; 
    bool cam_info_received = false;
    int bt_low;
    int bt_high;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaneFollower>());
    rclcpp::shutdown();
    return 0;
}
