// adaptor for common slam module
#include "ros/ros.h"
#include "dynamic_removal/keyScan.h"
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>


using namespace std;
using namespace message_filters;

ros::Publisher pubKeyFrame;
// pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;

void callback(const sensor_msgs::PointCloud2::ConstPtr& cloudMsg, const nav_msgs::Odometry::ConstPtr& odomMsg) {
    // downsample 0.4
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOri(new pcl::PointCloud<pcl::PointXYZI>());
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloudDS(new pcl::PointCloud<pcl::PointXYZI>());
    // pcl::fromROSMsg(*cloudMsg, *cloudOri);
    // downSizeFilter.setInputCloud(cloudOri);
    // downSizeFilter.filter(*cloudDS);
  
    dynamic_removal::keyScan kf;
    // pcl::toROSMsg(*cloudDS, kf.cloud_raw);
    kf.cloud_raw = *cloudMsg;
    kf.odomMsg = *odomMsg;
    pubKeyFrame.publish(kf);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "genKeyFrameNode");
    ros::NodeHandle nh;
    float leafSize;
    // nh.getParam("leafSize", leafSize);
    // downSizeFilter.setLeafSize(leafSize, leafSize, leafSize);
    pubKeyFrame = nh.advertise<dynamic_removal::keyScan>("lio_sam/slam_info", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/cloud_registered_body", 100);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "aft_pgo_odom", 100);
    typedef sync_policies::ExactTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;
  // ExactTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(20), cloud_sub, odom_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
    ros::spin();
}

