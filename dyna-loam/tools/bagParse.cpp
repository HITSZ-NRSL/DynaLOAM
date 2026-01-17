// 本文件用于生成kitti格式的实车数据集，将rosbag里面的点云帧保存为bin格式，同时记录其在map系下的坐标。(xyzi格式)
// 保存的文件夹为velodyne，里面存放bin格式的点云。times.txt保存时间戳，pose.txt保存位姿，每行12个数据，为odomImu_incremental

#include <string>
#include <vector>
#include <queue>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#include "common/ros_utils/transform.hpp"
#include "common/pcl_utils/pcl_utils.h"

using namespace std;
using namespace ros_utils;



using Point = pcl::PointXYZI;

int main(int argc, char **argv) {
    ros::init(argc, argv, "bagRecord");
    rosbag::Bag bag;
    ros::NodeHandle nh;

    string bag_path;
    string folder_path;
    string lidar_topic;
    string odom_topic; 
    nh.getParam("/bagParse/bag_path", bag_path);
    nh.getParam("/bagParse/folder_path", folder_path);
    nh.getParam("/bagParse/lidar_topic", lidar_topic);
    nh.getParam("/bagParse/odom_topic", odom_topic);

    std::cout << "ros bag path :" << bag_path << std::endl;
    std::cout << "output folder path :" << folder_path << std::endl; 
    std::cout << "lidar_topic: " << lidar_topic << endl;
    std::cout << "odom_topic: " << odom_topic << endl;

    queue<sensor_msgs::PointCloud2> pcQue;
    queue<nav_msgs::Odometry> odomQue;
    vector<sensor_msgs::PointCloud2> pcVec;
    vector<Eigen::Matrix4f> lidarOdomVec;

    pcVec.reserve(10000);
    lidarOdomVec.reserve(10000);

    bag.open(bag_path, rosbag::bagmode::Read);

    vector<string> topics = {lidar_topic, odom_topic};

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    foreach(rosbag::MessageInstance const m, view) {
        sensor_msgs::PointCloud2::ConstPtr pc_ptr = m.instantiate<sensor_msgs::PointCloud2>();
        if (pc_ptr != nullptr)
            pcQue.push(*pc_ptr);
        nav_msgs::Odometry::ConstPtr odom_ptr = m.instantiate<nav_msgs::Odometry>();
        if (odom_ptr != nullptr) {
            odomQue.push(*odom_ptr);
        }   
    }
    
    std::cout << "pcQue size : " << pcQue.size() << std::endl;

    while(!pcQue.empty()) {
        sensor_msgs::PointCloud2 &pc = pcQue.front(); 
        double curFrameTime = pc.header.stamp.toSec();
        double firstOdomTime = odomQue.front().header.stamp.toSec();
        if(curFrameTime <= firstOdomTime)
            pcQue.pop();
        else
            break;
    }

    std::cout << "pcQue size after filter : " << pcQue.size() << std::endl;

    while(!pcQue.empty()) {
        pcVec.push_back(pcQue.front());
        pcQue.pop();
    }

    for(auto pc : pcVec) {
        double curFrameTime = pc.header.stamp.toSec();
        while(!odomQue.empty() && odomQue.front().header.stamp.toSec() < curFrameTime) {
            odomQue.pop();
        }
        if(odomQue.empty())
            break;
        nav_msgs::Odometry thisOdom = odomQue.front();
        auto trans = OdomToTransform(thisOdom);
        Eigen::Matrix4f transMatrix = TransformToMatrix(trans);
        lidarOdomVec.push_back(transMatrix);
    }

    std::cout << "pcVec size : " << pcVec.size() << std::endl;
    std::cout << "lidaOdomVec size : " << lidarOdomVec.size() << std::endl;
    assert(pcVec.size() == lidarOdomVec.size());    

    // 点云直接保存，为bin格式
    char fmt[8];

    for(int i = 0; i < pcVec.size(); i++) {
        sprintf(fmt, "%06d", i);
        string fileName = string(fmt) + ".bin";
        pcl::PointCloud<pcl::PointXYZI>::Ptr curCloudptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(pcVec[i], *curCloudptr);
        string tmpfilePath = folder_path + "velodyne/" + fileName;
        curCloudptr->height = 1;
        curCloudptr->width = curCloudptr->size();
        pcl_utils::saveBinFile(curCloudptr, tmpfilePath);
    }  

    // 保存lidar odom
    string odomFilePath = folder_path + "pose.txt";
    ofstream outputOdom(odomFilePath);
    for(int i = 0; i < lidarOdomVec.size(); i++) {
        Eigen::Matrix4f curTrans = lidarOdomVec[i];
        outputOdom << curTrans(0, 0) << " " << curTrans(0, 1) << " " << curTrans(0, 2) << " " << curTrans(0, 3) << " " 
                   << curTrans(1, 0) << " " << curTrans(1, 1) << " " << curTrans(1, 2) << " " << curTrans(1, 3) << " " 
                   << curTrans(2, 0) << " " << curTrans(2, 1) << " " << curTrans(2, 2) << " " << curTrans(2, 3) << endl;
    }
    outputOdom.close();


    // 保存time
    string timeFilePath = folder_path + "time.txt";
    ofstream outputTime(timeFilePath);
    double startTime = pcVec[0].header.stamp.toSec();
    for(int i = 0; i < pcVec.size(); i++) {
        double thisTime = pcVec[i].header.stamp.toSec() - startTime;
        outputTime << thisTime << endl;
    }
    outputTime.close();
    std::cout << "finished record" << endl;
    ros::shutdown();
}
