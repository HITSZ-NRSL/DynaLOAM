// 本文件用来滤除bag点云中的动态点，需要包含的消息为原始点云，odom坐标，imu消息

#include <string>
#include <vector>
#include <queue>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Header.h>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include "common/ros_utils/transform.hpp"

using namespace std;
using namespace ros_utils;


struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

using PointXYZIRT = VelodynePointXYZIRT;

vector<double> rangeFilter = {-13, 20, -25, 30};

pcl::PointCloud<PointXYZIRT>::Ptr filterPointCloud(pcl::PointCloud<PointXYZIRT>::Ptr cloudIn, Eigen::Matrix4f& transCur)
    {
        pcl::PointCloud<PointXYZIRT>::Ptr cloudOut(new pcl::PointCloud<PointXYZIRT>());

        int cloudSize = cloudIn->size();
        
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < cloudSize; ++i)
        {
            PointXYZIRT pointFrom = cloudIn->points[i];
            PointXYZIRT point = pointFrom;
            point.x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            point.y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            point.z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            if ((point.x > rangeFilter[0]) && (point.x < rangeFilter[1]) && (point.y > rangeFilter[2]) && (point.y < rangeFilter[3]) && point.z > 0.2)
                continue;
            else
                cloudOut->points.push_back(point);
        }
        return cloudOut;
    }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "bagRemoval");
    rosbag::Bag bag;
    ros::NodeHandle nh;
    string bag_path;
    nh.getParam("/bagRemoval/bag_path", bag_path);
    string bag_output_path = bag_path.substr(0, bag_path.size() - 4);
    bag_output_path += "_filter.bag";

    vector<sensor_msgs::Imu> imuVec;
    vector<sensor_msgs::PointCloud2> pcVec;
    vector<sensor_msgs::PointCloud2> pcFilterVec;
    queue<nav_msgs::Odometry> odomQue;
    vector<nav_msgs::Odometry> odomVec;
    imuVec.reserve(10000);
    pcVec.reserve(10000);
    odomVec.reserve(10000);
    pcFilterVec.reserve(10000);



    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudOut;
    laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
    laserCloudOut.reset(new pcl::PointCloud<PointXYZIRT>());

    // 读取rosbag的数据
    bag.open(bag_path, rosbag::bagmode::Read);
    vector<string> topics = {"/lidar_points", "/odom_basefootprint", "/imu/data"};

    for(rosbag::MessageInstance const m : rosbag::View(bag)) {
        sensor_msgs::Imu::ConstPtr imu_ptr = m.instantiate<sensor_msgs::Imu>();
        if (imu_ptr != nullptr)
            imuVec.push_back(*imu_ptr);
        sensor_msgs::PointCloud2::ConstPtr pc_ptr = m.instantiate<sensor_msgs::PointCloud2>();
        if (pc_ptr != nullptr)
            pcVec.push_back(*pc_ptr);
        nav_msgs::Odometry::ConstPtr odom_ptr = m.instantiate<nav_msgs::Odometry>();
        if (odom_ptr != nullptr) {
            odomQue.push(*odom_ptr);
            odomVec.push_back(*odom_ptr);
        }
            
    }
    cout << "imuVec size: " << imuVec.size() << endl;
    cout << "pcVec size: " << pcVec.size() << endl;
    cout << "odomVec size: " << odomQue.size() << endl;

    // 将点云转化到世界坐标系下面，然后滤除固定区域的点云，滤除的点云放进pcFilterVec里面；
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
        Eigen::Matrix4f lidarToBase;
        lidarToBase << 0, -1, 0, 0.223, 
                       1, 0, 0, 0, 
                       0, 0, 1, 1.212,
                       0, 0, 0, 1; 
        transMatrix = transMatrix * lidarToBase;
        laserCloudIn->clear();
        laserCloudOut->clear();
        pcl::moveFromROSMsg(pc, *laserCloudIn);
        laserCloudOut = filterPointCloud(laserCloudIn, transMatrix);
        // cout << laserCloudOut->size() << endl;

        // 点云转化到原始坐标系

        pcl::transformPointCloud(*laserCloudOut, *laserCloudOut, transMatrix.inverse());

        sensor_msgs::PointCloud2 cloudMsg;
        pcl::toROSMsg(*laserCloudOut, cloudMsg);
        cloudMsg.header = pc.header;
        pcFilterVec.push_back(cloudMsg);
    }

    cout << "filter pcVec size: " << pcFilterVec.size() << endl;
    bag.close();

    // generate rosbag
    rosbag::Bag outputBag;
    outputBag.open(bag_output_path, rosbag::bagmode::Write);
    for(int i = 0; i < pcFilterVec.size(); i++) {
        outputBag.write("lidar_points", pcFilterVec[i].header.stamp, pcFilterVec[i]);
    }

    for(int i = 0; i < imuVec.size(); i++) {
        outputBag.write("/imu/data", imuVec[i].header.stamp, imuVec[i]);
    }

    for(int i = 0; i < odomVec.size(); i++) {
        outputBag.write("/odom_basefootprint", odomVec[i].header.stamp, odomVec[i]);
    }
    outputBag.close();
    return 0;
}
