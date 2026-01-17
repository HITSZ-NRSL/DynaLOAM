#include "dynamic_removal/utility.h"
#include "common/pcl_utils/pcl_utils.h"
#include <iostream>

using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointXYZI>;
using namespace std;

int main(int argc, char **argv)
{
    if (argc != 2) {
        cout << "invalid argc, please give the bi file path";
    }

    string path = argv[1];
    cout << "file path: " << path;

    ros::init(argc, argv, "pub_lidar_cloud");
    ros::NodeHandle nh;
    ros::Publisher cloudPub;
    cloudPub = nh.advertise<sensor_msgs::PointCloud2>("lidar_points", 1);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    cloud.reset(new PointCloud());
    pcl_utils::readBinFile(cloud, path);
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = "velodyne";
    msg.header.stamp = ros::Time::now();
    msg.is_dense = true;
    ros::Rate r(1);
    while(ros::ok()) {
        cloudPub.publish(msg);
        r.sleep();
    }
    return 0;
}