// 本文件用于生成kitti格式的仿真数据集，将rosbag里面的点云帧保存为bin格式，同时记录其在map系下的坐标。(xyzi格式)
// 保存的文件夹为velodyne，里面存放bin格式的点云。times.txt保存时间戳，pose.txt保存位姿，每行12个数据，初始为单位阵

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

#include "common/ros_utils/transform.hpp"
#include "common/pcl_utils/pcl_utils.h"

using namespace std;
using namespace ros_utils;



using Point = pcl::PointXYZI;

void labelCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn, vector<double> &rangeFilter, const Eigen::Matrix4f& transCur) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZI>());
    int cloudSize = cloudIn->size();
        
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < cloudSize; ++i)
    {
        Point pointFrom = cloudIn->points[i];
        Point point = pointFrom;
        point.x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        point.y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        point.z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        if ((point.x > rangeFilter[0]) && (point.x < rangeFilter[1]) 
        && (point.y > rangeFilter[2]) && (point.y < rangeFilter[3]) && point.z > 0.1) {
            point.intensity = 100;
            cloudOut->points.push_back(point);
        }
        else
            cloudOut->points.push_back(point);
    }
    Eigen::Matrix4f transInv = transCur.inverse();
    cloudIn->clear();
    pcl::transformPointCloud(*cloudOut, *cloudIn, transInv);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "bagRecord");
    rosbag::Bag bag;
    ros::NodeHandle nh;

    string bag_path;
    string folder_path;
    nh.getParam("/bagParse/bag_path", bag_path);
    nh.getParam("/bagParse/folder_path", folder_path);

    cout << "ros bag path :" << bag_path << endl;
    cout << "output folder path :" << folder_path << endl; 

    vector<sensor_msgs::PointCloud2> pcVec;
    queue<nav_msgs::Odometry> odomQue;
    vector<Eigen::Matrix4f> lidarOdomVec;

    pcVec.reserve(1000);
    lidarOdomVec.reserve(1000);
    

    bag.open(bag_path, rosbag::bagmode::Read);
    vector<string> topics = {"/ouster/points", "/Odometry"};

    for(rosbag::MessageInstance const m : rosbag::View(bag)) {
        sensor_msgs::PointCloud2::ConstPtr pc_ptr = m.instantiate<sensor_msgs::PointCloud2>();
        if (pc_ptr != nullptr)
            pcVec.push_back(*pc_ptr);
        nav_msgs::Odometry::ConstPtr odom_ptr = m.instantiate<nav_msgs::Odometry>();
        if (odom_ptr != nullptr) {
            odomQue.push(*odom_ptr);
        }   
    }

    for (auto pc : pcVec) {
        double curFrameTime = pc.header.stamp.toSec();
        while(!odomQue.empty() && odomQue.front().header.stamp.toSec() < curFrameTime) {
            odomQue.pop();
        }
        if(odomQue.empty())
            break;
        nav_msgs::Odometry thisOdom = odomQue.front();
        auto trans = OdomToTransform(thisOdom);
        Eigen::Matrix4f transMatrix = TransformToMatrix(trans);
        // Eigen::Matrix4f lidarToBase;
        // lidarToBase << 0, -1, 0, 0.223, 
        //                1, 0, 0, 0, 
        //                0, 0, 1, 1.212,
        //                0, 0, 0, 1; 
        // transMatrix = transMatrix * lidarToBase;
        lidarOdomVec.push_back(transMatrix);
        // 点云转化到原始坐标系
    }

    // 点云直接保存，为bin格式
    char fmt[6];
    // vector<double> rangeFilter = {-13, 20, -23, 30};

    for(int i = 0; i < pcVec.size(); i++) {
        sprintf(fmt, "%06d", i);
        string fileName = string(fmt) + ".bin";
        pcl::PointCloud<pcl::PointXYZI>::Ptr curCloudptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(pcVec[i], *curCloudptr);

        // 标记动态点云，设置强度为100，其中原始点云强度为0；
        // labelCloud(curCloudptr, rangeFilter, lidarOdomVec[i]);

        string tmpfilePath = folder_path + "velodyne/" + fileName;
        pcl_utils::saveBinFile(curCloudptr, tmpfilePath);
    }  

    // 保存lidar odom
    string odomFilePath = folder_path + "pose.txt";
    ofstream outputOdom(odomFilePath);
    outputOdom << "1 0 0 0 0 1 0 0 0 0 1 0" << endl;
    for(int i = 1; i < lidarOdomVec.size(); i++) {
        Eigen::Matrix4f curTrans = (lidarOdomVec[0].inverse()) * (lidarOdomVec[i]);
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
    cout << "finished parse" << endl;
    ros::shutdown();
}
