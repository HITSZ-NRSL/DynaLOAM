
// std
#include <iostream>
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <signal.h>

// ros
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Point.h>
#include <tf/LinearMath/Quaternion.h> // to Quaternion_to_euler
#include <tf/LinearMath/Matrix3x3.h> // to Quaternion_to_euler
#include <tf/transform_datatypes.h> // createQuaternionFromRPY
#include <tf_conversions/tf_eigen.h> // tf <-> eigen

// other
#include "dynamic_removal/dynamicFilter.h"
#include "dynamic_removal/save_map.h"
#include "dynamic_removal/keyScan.h"

using namespace std;
using PointXYZI = pcl::PointXYZI;
using PointXYZIN = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointXYZI>;

ros::Publisher pubStaticCloud;
ros::Publisher pubdynamicCloud;
ros::Publisher pubScanFilteredbyKNN;
ros::Publisher pubCornerCloudFiltered;
ros::Publisher pubSurfCloudFiltered;
ros::Publisher pubDynamicFeature;
ros::Publisher pubStaticLocalMap;
// ros::Publisher pubOdomOptimized;
// ros::Publisher pubLocalMap;
// ros::Publisher pubRawScanInMap;
// ros::Publisher pubOptScanInMap;

ros::ServiceServer srvSaveMap;
shared_ptr<travel::TravelGroundSeg<PointXYZI>> groundSegSingleScan;

pcl::KdTreeFLANN<PointXYZI>::Ptr kdtreeFilter;
pcl::KdTreeFLANN<PointXYZI>::Ptr kdtreeInitialFilter;
pcl::KdTreeFLANN<PointXYZI>::Ptr kdtreeGround;
pcl::VoxelGrid<PointXYZI> curLocalmapDownSizeFilter;

dynamicFilter* df;
// cloud vector and pose vector
vector<PointCloud::Ptr> scanVec;
vector<Eigen::Matrix4f> poseVec;

// static cloud vector, original lidar scan use knn filter in map frame
queue<PointCloud::Ptr> staticScanBuffer;
queue<PointCloud::Ptr> dynamicScanBuffer;
queue<double> staticScanTimeBuffer;

// cloud buffer and pose buffer
vector<PointCloud::Ptr> scanVecBuffer;
vector<Eigen::Matrix4f> poseVecBuffer;

// save processed submap in map frame
vector<PointCloud::Ptr> staticSubmapVec;
vector<PointCloud::Ptr> dynamicSubmapVec;

// all history raw key frame vec
vector<PointCloud::Ptr> rawScanVecFull;

// all history processed key frame vec
vector<PointCloud::Ptr> dynamicScanVecFull;
vector<PointCloud::Ptr> staticScanVecFull;
// odom msg
vector<nav_msgs::Odometry> scanOdomVecFull;
vector<Eigen::Matrix4f> updatedOdomFull;
vector<Eigen::Matrix4f> originOdomFull;
nav_msgs::Path keyFramePath;

// the map used for scan to map
PointCloud::Ptr curLocalMap;
PointCloud::Ptr curLocalMapDS;
// the pre laser scan
PointCloud::Ptr lastStaticNogroundSubmap;
PointCloud::Ptr lastDynamicSubmap;

//relative pose
Eigen::Matrix4f odomDelta;
Eigen::Matrix4f curPGOPose;

// mutex
mutex deltaOdomMutex;
mutex cornerFeatureMutex;
mutex surfFeatureMutex;
mutex staticSubmapMutex;

// queue
queue<nav_msgs::Odometry> rawOdomQueue;
queue<sensor_msgs::PointCloud2ConstPtr> cornerCloudQueue;
queue<sensor_msgs::PointCloud2ConstPtr> surfCloudQueue;

// downSizeFilter

int maxCapacity{15}; // need to be set
int curCapacity{0};
int id{0};

int frame_count{0};
int interval{1}; // need to be set

double curStaticSubmapTime{0};
double lastStaticSubmapTime{0};

std::mutex mtxFilter;
std::condition_variable conditionV;
bool handleFlag{false};
bool filterScanInDR{false};

void signal_handler(sig_atomic_t s)
{
  std::cout << "You pressed Ctrl + C, exiting" << std::endl;
  exit(1);
}

float pointDistance(PointXYZI p) {
        return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

Eigen::Matrix4f scan2MapOptimization(PointCloud::Ptr sourceCloud, PointCloud::Ptr targetCloud, Eigen::Matrix4f& initTrans) {
    static pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp;
    icp.setMaxCorrespondenceDistance(0.5);
    icp.setMaximumIterations(10);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(sourceCloud);
    icp.setInputTarget(targetCloud);
    PointCloud::Ptr res(new PointCloud());
    icp.align(*res);
    if (icp.hasConverged() == false) {
        ROS_INFO_STREAM("\033[31m icp failed ! \033[0m");
        return initTrans;
    }
    Eigen::Matrix4f icp_res = icp.getFinalTransformation();
    icp_res = initTrans * icp_res.inverse();
    return icp_res;
}

Eigen::Matrix4f odom2Matrix4f(const nav_msgs::Odometry odom_msg) {
  geometry_msgs::Quaternion orientation = odom_msg.pose.pose.orientation;
  geometry_msgs::Point position = odom_msg.pose.pose.position;
  Eigen::Quaternionf quat;
  quat.w() = orientation.w;
  quat.x() = orientation.x;
  quat.y() = orientation.y;
  quat.z() = orientation.z;
  Eigen::Isometry3f isometry = Eigen::Isometry3f::Identity();
  isometry.linear() = quat.toRotationMatrix();
  isometry.translation() = Eigen::Vector3f(position.x, position.y, position.z);
  return isometry.matrix().cast<float>();
} 

Eigen::Matrix4f pose2Matrix4f(const geometry_msgs::Pose pose) {
    geometry_msgs::Point position = pose.position;
    Eigen::Quaternionf quat;
    quat.w() = pose.orientation.w;
    quat.x() = pose.orientation.x;
    quat.y() = pose.orientation.y;
    quat.z() = pose.orientation.z;
    Eigen::Isometry3f isometry = Eigen::Isometry3f::Identity();
    isometry.linear() = quat.toRotationMatrix();
    isometry.translation() = Eigen::Vector3f(position.x, position.y, position.z);
    return isometry.matrix().cast<float>();
}

geometry_msgs::Pose matrix4f2Pose(const Eigen::Matrix4f &pose_eig_in)
{
    geometry_msgs::Pose pose_;
    pose_.position.x = pose_eig_in(0, 3);
    pose_.position.y = pose_eig_in(1, 3);
    pose_.position.z = pose_eig_in(2, 3);
    Eigen::Quaternionf _q;
    _q = pose_eig_in.block<3, 3>(0, 0);
    pose_.orientation.x = _q.x();
    pose_.orientation.y = _q.y();
    pose_.orientation.z = _q.z();
    pose_.orientation.w = _q.w();
	return pose_;
}


void transformPoint(const PointXYZI& point_in, PointXYZI& point_out, const Eigen::Matrix4f& _trans) {
    point_out.x = _trans(0, 0) * point_in.x + _trans(0, 1) * point_in.y + _trans(0, 2) * point_in.z + _trans(0, 3);
    point_out.y = _trans(1, 0) * point_in.x + _trans(1, 1) * point_in.y + _trans(1, 2) * point_in.z + _trans(1, 3);
    point_out.z = _trans(2, 0) * point_in.x + _trans(2, 1) * point_in.y + _trans(2, 2) * point_in.z + _trans(2, 3);
}

void laserKeyFrameHandler(const dynamic_removal::keyScan::ConstPtr& keyFrameMsg) {


    // test
    if (0) {
        common::TicToc t2;
        lock_guard<mutex> lock_corner(cornerFeatureMutex);
        while (!cornerCloudQueue.empty() && cornerCloudQueue.front()->header.stamp.toSec() < keyFrameMsg->cloud_raw.header.stamp.toSec()) {
            cornerCloudQueue.pop();
        }
        if (cornerCloudQueue.empty()) {
            return;
        }
        lock_guard<mutex> lock_surf(surfFeatureMutex);
        while (!surfCloudQueue.empty() && surfCloudQueue.front()->header.stamp.toSec() < keyFrameMsg->cloud_raw.header.stamp.toSec()) {
            surfCloudQueue.pop();
        }
        if (surfCloudQueue.empty()) {
            return;
        }
        double timeLaserCloudCornerLast = cornerCloudQueue.front()->header.stamp.toSec();
        double timeLaserCloudSurfLast = surfCloudQueue.front()->header.stamp.toSec();
        double timeLaserCloudFullRes = keyFrameMsg->cloud_raw.header.stamp.toSec();
        
        if (timeLaserCloudCornerLast != timeLaserCloudFullRes || timeLaserCloudSurfLast != timeLaserCloudFullRes) {
            ROS_INFO_STREAM("no valid surf features and corner features");
            return;
        }

        PointCloud::Ptr laserCloudCornerLast(new PointCloud());
        PointCloud::Ptr laserCloudSurfLast(new PointCloud());
        sensor_msgs::PointCloud2 cornerCloudCur = *(cornerCloudQueue.front());
        sensor_msgs::PointCloud2 surfCloudCur = *(surfCloudQueue.front());
        sensor_msgs::PointCloud2 oriCloud = keyFrameMsg->cloud_raw;
        cornerCloudQueue.pop();
        surfCloudQueue.pop();
        std::this_thread::sleep_for(std::chrono::milliseconds(90));
        pubScanFilteredbyKNN.publish(oriCloud);
        pubSurfCloudFiltered.publish(surfCloudCur);
        pubCornerCloudFiltered.publish(cornerCloudCur);
        cout << "handle single scan cost: " << t2.toc() * 1000 << "ms" << endl;
        return;
    }

    frame_count += 1;
    common::TicToc t1;
    PointCloud::Ptr curCloud(new PointCloud());
    PointCloud::Ptr curCloudDS(new PointCloud());
    pcl::fromROSMsg(keyFrameMsg->cloud_raw, *curCloud);
    curLocalmapDownSizeFilter.setInputCloud(curCloud);
    curLocalmapDownSizeFilter.filter(*curCloudDS);

    Eigen::Matrix4f curPose;
    {
        lock_guard<mutex> lock(deltaOdomMutex);
        rawOdomQueue.push(keyFrameMsg->odomMsg);
        if (!rawOdomQueue.empty()) {
            odomDelta = odom2Matrix4f(rawOdomQueue.front()).inverse() * odom2Matrix4f(rawOdomQueue.back());
        }
        while (rawOdomQueue.size() > 2000) {
            // in case no pgp pose received;
            curPGOPose = odom2Matrix4f(rawOdomQueue.front());
            rawOdomQueue.pop();
        }
        curPose = curPGOPose * odomDelta;
    }

    // gen local dynamic map
    {
        lock_guard<mutex> _lock(staticSubmapMutex);
        static int lastStaticSubmapVecSize = 0;
        if (!staticSubmapVec.empty()) {
            if (staticSubmapVec.size() != lastStaticSubmapVecSize) {
                while (!staticScanBuffer.empty() && staticScanTimeBuffer.front() <= lastStaticSubmapTime) {
                    staticScanBuffer.pop();
                    dynamicScanBuffer.pop();
                    staticScanTimeBuffer.pop();
                }
                queue<PointCloud::Ptr> _tmpStaticScanBuffer = staticScanBuffer;
                curLocalMap->clear();
                *curLocalMap = *lastStaticNogroundSubmap;
                while (!_tmpStaticScanBuffer.empty()) {
                    *curLocalMap += *_tmpStaticScanBuffer.front();
                    _tmpStaticScanBuffer.pop();
                }
                if (!curLocalMap->empty()) {
                    curLocalmapDownSizeFilter.setInputCloud(curLocalMap);
                    curLocalmapDownSizeFilter.filter(*curLocalMapDS);
                    kdtreeInitialFilter->setInputCloud(curLocalMapDS);
                    lastStaticSubmapVecSize = staticSubmapVec.size();
                } 
            } else {
                *curLocalMap += *staticScanBuffer.back();
                if (!curLocalMap->empty()) {
                    curLocalmapDownSizeFilter.setInputCloud(curLocalMap);
                    curLocalmapDownSizeFilter.filter(*curLocalMapDS);
                    kdtreeInitialFilter->setInputCloud(curLocalMapDS);
                } 
            }
        }
    }

    cout << "frame " << frame_count << " curLocalmap size: " << curLocalMapDS->size() << endl;
    cout << "gennerate local map cost: " << t1.toc() * 1000 << "ms" << endl;

    if (curLocalMap->empty()) {
        lock_guard<mutex> lock_corner(cornerFeatureMutex);
            while (!cornerCloudQueue.empty() && cornerCloudQueue.front()->header.stamp.toSec() < keyFrameMsg->cloud_raw.header.stamp.toSec()) {
                cornerCloudQueue.pop();
            }
            if (cornerCloudQueue.empty()) {
                return;
            }
            lock_guard<mutex> lock_surf(surfFeatureMutex);
            while (!surfCloudQueue.empty() && surfCloudQueue.front()->header.stamp.toSec() < keyFrameMsg->cloud_raw.header.stamp.toSec()) {
                surfCloudQueue.pop();
            }
            if (surfCloudQueue.empty()) {
                return;
            }
            double timeLaserCloudCornerLast = cornerCloudQueue.front()->header.stamp.toSec();
            double timeLaserCloudSurfLast = surfCloudQueue.front()->header.stamp.toSec();
            double timeLaserCloudFullRes = keyFrameMsg->cloud_raw.header.stamp.toSec();
            
            if (timeLaserCloudCornerLast != timeLaserCloudFullRes || timeLaserCloudSurfLast != timeLaserCloudFullRes) {
                ROS_INFO_STREAM("no valid surf features and corner features");
                return;
            }

            PointCloud::Ptr laserCloudCornerLast(new PointCloud());
            PointCloud::Ptr laserCloudSurfLast(new PointCloud());
            pcl::fromROSMsg(*cornerCloudQueue.front(), *laserCloudCornerLast);
            pcl::fromROSMsg(*surfCloudQueue.front(), *laserCloudSurfLast);
            sensor_msgs::PointCloud2 oriCornerCloudMsg = *cornerCloudQueue.front();
            sensor_msgs::PointCloud2 oriSurfCloudMsg = *surfCloudQueue.front();
            cornerCloudQueue.pop();
            surfCloudQueue.pop();
            // publish
            pubScanFilteredbyKNN.publish(keyFrameMsg->cloud_raw);
            pubSurfCloudFiltered.publish(oriCornerCloudMsg);
            pubCornerCloudFiltered.publish(oriSurfCloudMsg);

    } else {
        PointCloud::Ptr scanFilteredByKNN(new PointCloud());
        PointCloud::Ptr dynamicFeature(new PointCloud());

        // split cur scan to ground and no grounb
        PointCloud::Ptr curScanNoground(new PointCloud());
        PointCloud::Ptr curScanGround(new PointCloud());
        double _ground_seg_time = 0;

        groundSegSingleScan->estimateGround(*curCloud, *curScanGround, *curScanNoground, _ground_seg_time);
        kdtreeGround->setInputCloud(curScanGround);

        if (!curLocalMapDS->empty()) {
            // resTrans = scan2MapOptimization(curCloudTrans, curLocalMapDS, curPose);
            //filter cloud by knn
            for (auto const& p : *curScanNoground) {
                PointXYZI p_trans;
                transformPoint(p, p_trans, curPose);
                int K = 1;
                vector<int> pointIdxKNNSearch(K);
                vector<float> pointKNNSquaredDistance(K);
                int _res = kdtreeInitialFilter->nearestKSearch(p_trans, K, pointIdxKNNSearch, pointKNNSquaredDistance);
                if (pointDistance(p) > 40 || _res <= 0 || pointKNNSquaredDistance[0] < 0.3) {
                    // dynamicFeature->push_back(p);
                    scanFilteredByKNN->push_back(p);
                } else {
                    // scanFilteredByKNN->push_back(p);
                    dynamicFeature->push_back(p);
                }
            }
        } 

        // add cur static points
        PointCloud::Ptr curStaticScan(new PointCloud());
        pcl::transformPointCloud(*scanFilteredByKNN, *curStaticScan, curPose);
        PointCloud::Ptr curDynamicScan(new PointCloud());
        pcl::transformPointCloud(*dynamicFeature, *curDynamicScan, curPose);

        // push_back only no ground part
        staticScanBuffer.push(curStaticScan);
        dynamicScanBuffer.push(curDynamicScan);
        staticScanTimeBuffer.push(keyFrameMsg->cloud_raw.header.stamp.toSec());

        PointCloud::Ptr cornerCloudFiltered(new PointCloud());
        PointCloud::Ptr surfCloudFiltered(new PointCloud());
        sensor_msgs::PointCloud2 oriCornerCloudMsg;
        sensor_msgs::PointCloud2 oriSurfCloudMsg;

        // filter feature points by knn
        if (!curLocalMapDS->empty()) {
            // filter corner feature by knn
            lock_guard<mutex> lock_corner(cornerFeatureMutex);
            while (!cornerCloudQueue.empty() && cornerCloudQueue.front()->header.stamp.toSec() < keyFrameMsg->cloud_raw.header.stamp.toSec()) {
                cornerCloudQueue.pop();
            }
            if (cornerCloudQueue.empty()) {
                return;
            }
            lock_guard<mutex> lock_surf(surfFeatureMutex);
            while (!surfCloudQueue.empty() && surfCloudQueue.front()->header.stamp.toSec() < keyFrameMsg->cloud_raw.header.stamp.toSec()) {
                surfCloudQueue.pop();
            }
            if (surfCloudQueue.empty()) {
                return;
            }
            double timeLaserCloudCornerLast = cornerCloudQueue.front()->header.stamp.toSec();
            double timeLaserCloudSurfLast = surfCloudQueue.front()->header.stamp.toSec();
            double timeLaserCloudFullRes = keyFrameMsg->cloud_raw.header.stamp.toSec();
            
            if (timeLaserCloudCornerLast != timeLaserCloudFullRes || timeLaserCloudSurfLast != timeLaserCloudFullRes) {
                ROS_INFO_STREAM("no valid surf features and corner features");
                return;
            }

            PointCloud::Ptr laserCloudCornerLast(new PointCloud());
            PointCloud::Ptr laserCloudSurfLast(new PointCloud());
            pcl::fromROSMsg(*cornerCloudQueue.front(), *laserCloudCornerLast);
            pcl::fromROSMsg(*surfCloudQueue.front(), *laserCloudSurfLast);
            oriCornerCloudMsg = *cornerCloudQueue.front();
            oriSurfCloudMsg = *surfCloudQueue.front();
            cornerCloudQueue.pop();
            surfCloudQueue.pop();
            
            // filter origin corner cloud
            for (auto const& p : *laserCloudCornerLast) {
                int K = 1;
                PointXYZI p_trans;
                transformPoint(p, p_trans, curPose);
                vector<int> pointIdxKNNSearch(K);
                vector<float> pointKNNSquaredDistance(K);
                int _res = kdtreeInitialFilter->nearestKSearch(p_trans, K, pointIdxKNNSearch, pointKNNSquaredDistance);
                if (pointDistance(p) > 30 || _res <= 0 || pointKNNSquaredDistance[0] < 0.3 || pointKNNSquaredDistance[0] > 2) {
                    cornerCloudFiltered->push_back(p);
                } else if(kdtreeGround->nearestKSearch(p, K, pointIdxKNNSearch, pointKNNSquaredDistance) && pointKNNSquaredDistance[0]< 0.1) {
                    cornerCloudFiltered->push_back(p);
                } else {
                    // cornerCloudFiltered->push_back(p);
                }
            }
            // filter origin surf cloud
            for (auto const& p : *laserCloudSurfLast) {
                int K = 1;
                PointXYZI p_trans;
                transformPoint(p, p_trans, curPose);
                vector<int> pointIdxKNNSearch(K);
                vector<float> pointKNNSquaredDistance(K);
                int _res = kdtreeInitialFilter->nearestKSearch(p_trans, K, pointIdxKNNSearch, pointKNNSquaredDistance);
                if (pointDistance(p) > 30 || _res <= 0 || pointKNNSquaredDistance[0] < 0.3 || pointKNNSquaredDistance[0] > 2) {
                    surfCloudFiltered->push_back(p);
                } else if(kdtreeGround->nearestKSearch(p, K, pointIdxKNNSearch, pointKNNSquaredDistance) && pointKNNSquaredDistance[0]< 0.1) {
                    surfCloudFiltered->push_back(p);
                } else {
                    // surfCloudFiltered->push_back(p);
                }
            }
        }

        if (!scanFilteredByKNN->empty()) {
            *scanFilteredByKNN += *curScanGround;
            sensor_msgs::PointCloud2 scanFilteredMsg;
            pcl::toROSMsg(*scanFilteredByKNN, scanFilteredMsg);
            scanFilteredMsg.header.frame_id = "velodyne";
            scanFilteredMsg.header.stamp = keyFrameMsg->cloud_raw.header.stamp;
            sensor_msgs::PointCloud2 dynamicFeatureMsg;
            pcl::toROSMsg(*dynamicFeature, dynamicFeatureMsg);
            dynamicFeatureMsg.header.frame_id = "velodyne";
            dynamicFeatureMsg.header.stamp = keyFrameMsg->cloud_raw.header.stamp;
            pubScanFilteredbyKNN.publish(scanFilteredMsg);
            pubDynamicFeature.publish(dynamicFeatureMsg);
        }

        if (!surfCloudFiltered->empty()) {
            sensor_msgs::PointCloud2 surfCloudFilteredMsg;
            pcl::toROSMsg(*surfCloudFiltered, surfCloudFilteredMsg);
            surfCloudFilteredMsg.header.frame_id = "velodyne";
            surfCloudFilteredMsg.header.stamp = keyFrameMsg->cloud_raw.header.stamp;
            pubSurfCloudFiltered.publish(surfCloudFilteredMsg);
        }

        if (!cornerCloudFiltered->empty()) {
            sensor_msgs::PointCloud2 cornerCloudFilteredMsg;
            pcl::toROSMsg(*cornerCloudFiltered, cornerCloudFilteredMsg);
            cornerCloudFilteredMsg.header.frame_id = "velodyne";
            cornerCloudFilteredMsg.header.stamp = keyFrameMsg->cloud_raw.header.stamp;
            pubCornerCloudFiltered.publish(cornerCloudFilteredMsg);
        }

    }

        
    if (frame_count % interval == 0) {
        scanVec.push_back(curCloudDS);
        poseVec.push_back(curPose);
        // rawScanVecFull.push_back(curCloudDS);
        scanOdomVecFull.push_back(keyFrameMsg->odomMsg);
        originOdomFull.push_back(curPose);
        curCapacity++;  
    }

    if(curCapacity == maxCapacity) {
        curCapacity = 0;
        curStaticSubmapTime = keyFrameMsg->cloud_raw.header.stamp.toSec();
        // handle the submap
        std::unique_lock<std::mutex> lck(mtxFilter);
        handleFlag = true;
        scanVecBuffer = scanVec;
        poseVecBuffer = poseVec;
        conditionV.notify_all();
        // reset param
        scanVec.clear();
        poseVec.clear();
    }
    ROS_INFO_STREAM("handle single scan cost: " << t1.toc() * 1000 << "ms");
}

void handleSubmapThread() {
    while (ros::ok()){
        std::unique_lock<std::mutex> lck(mtxFilter);
        conditionV.wait(lck, [&](){return handleFlag;});
        id++;
        cout << "handling submap " << id << endl; 
        common::TicToc t2;
        df->onlineProcess(scanVecBuffer, poseVecBuffer);

        // publish cloud
        sensor_msgs::PointCloud2 staticCloudMsg;
        sensor_msgs::PointCloud2 dynamicCloudMsg;
        // sensor_msgs::PointCloud2 rawSubmap;
        // transCloud
        PointCloud::Ptr staticCloud(new PointCloud());
        PointCloud::Ptr dynamicCloud(new PointCloud());
        PointCloud::Ptr staticNogroundCloud(new PointCloud());

        pcl::transformPointCloud(*(df->static_cluster_vis), *staticCloud, poseVecBuffer[0]);
        pcl::transformPointCloud(*(df->dynamic_cluster_vis), *dynamicCloud, poseVecBuffer[0]);
        // change
        pcl::transformPointCloud(*(df->staticNogroundCloud), *lastStaticNogroundSubmap, poseVecBuffer[0]);

        {
            lock_guard<mutex> _lock(staticSubmapMutex);
            staticSubmapVec.push_back(staticCloud);
            dynamicSubmapVec.push_back(dynamicCloud);
            *lastDynamicSubmap = *dynamicCloud;
        }

        pcl::toROSMsg(*staticCloud, staticCloudMsg);
        pcl::toROSMsg(*dynamicCloud, dynamicCloudMsg);

        staticCloudMsg.header.frame_id = "camera_init";
        dynamicCloudMsg.header.frame_id = "camera_init";
        pubStaticCloud.publish(staticCloudMsg);
        pubdynamicCloud.publish(dynamicCloudMsg);

        // filter origin scan by static submap
        if (filterScanInDR) {
            kdtreeFilter->setInputCloud(staticCloud);
            for (int id =  0; id < scanVecBuffer.size(); ++id) {
                PointCloud::Ptr filteredKeyFrame(new PointCloud());
                PointCloud::Ptr dynamicKeyFrame(new PointCloud());
                // for every point in keyframe
                for(const auto& p : *scanVecBuffer[id]) {
                    PointXYZI _p;
                    transformPoint(p, _p, poseVecBuffer[id]);
                    int K = 1;
                    vector<int> pointIdxNKNSearch(K);
                    vector<float> pointNKNSquaredDistance(K);
                    if (kdtreeFilter->nearestKSearch(_p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 && 
                        pointNKNSquaredDistance[0] < 0.1) {
                            filteredKeyFrame->push_back(p);
                    }
                    else {
                        dynamicKeyFrame->push_back(p);
                    }
                }
                staticScanVecFull.push_back(filteredKeyFrame);
                dynamicScanVecFull.push_back(dynamicKeyFrame);
            }       
        }
        df->resetParameter();
        scanVecBuffer.clear();
        poseVecBuffer.clear();
        cout << "handling submap done, cost: " << t2.toc() * 1000 << "ms" << endl;
        lastStaticSubmapTime = curStaticSubmapTime;
        handleFlag = false;
    }
}

// void updateFullOdom() {
//     cout << "full pose size: " << scanOdomVecFull.size() << endl;
//     cout << "key frame path size: " << keyFramePath.poses.size() << endl;
//     if(scanOdomVecFull.size() == 0 || keyFramePath.poses.size() == 0) {
//         return;
//     }
//     int scanSize = scanOdomVecFull.size();
//     int kfSzie = keyFramePath.poses.size();
//     int nextKeyFrameid = 0;
//     updatedOdomFull.clear();

//     if(scanOdomVecFull[0].header.stamp == keyFramePath.poses[0].header.stamp) {
//         updatedOdomFull.push_back(pose2Matrix4f(keyFramePath.poses[0].pose));
//         nextKeyFrameid++;
//     }
//     else {
//         updatedOdomFull.push_back(pose2Matrix4f(scanOdomVecFull[0].pose.pose));
//     }
//     for (int i = 1; i < scanSize; ++i) {
//         if(nextKeyFrameid < kfSzie && scanOdomVecFull[i].header.stamp == keyFramePath.poses[nextKeyFrameid].header.stamp) {
//             updatedOdomFull.push_back(pose2Matrix4f(keyFramePath.poses[nextKeyFrameid].pose));
//             nextKeyFrameid++;
//         }
//         else {
//             Eigen::Matrix4f relativeTrans = pose2Matrix4f(scanOdomVecFull[i-1].pose.pose).inverse() * pose2Matrix4f(scanOdomVecFull[i].pose.pose);
//             Eigen::Matrix4f curTrans = updatedOdomFull.back() * relativeTrans;
//             updatedOdomFull.push_back(curTrans);
//         }
//     }
// }

// void keyFramePathHandler(const nav_msgs::Path::ConstPtr& pathMsg) {
//     keyFramePath = *pathMsg;
//     updateFullOdom();
// }

void PGOOdomHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
    lock_guard<mutex> lock(deltaOdomMutex);
    while (!rawOdomQueue.empty()) {
        if (rawOdomQueue.front().header.stamp.toSec() <= odomMsg->header.stamp.toSec())
            rawOdomQueue.pop();
    }
    curPGOPose = odom2Matrix4f(*odomMsg);
    odomDelta = Eigen::Matrix4f::Identity();
}

void cornerFeatureHandler(const sensor_msgs::PointCloud2::ConstPtr& cloudIn) {
    lock_guard<mutex> lock(cornerFeatureMutex);
    cornerCloudQueue.push(cloudIn);
}

void surfFeatureHandler(const sensor_msgs::PointCloud2::ConstPtr& cloudIn) {
    lock_guard<mutex> lock(surfFeatureMutex);
    surfCloudQueue.push(cloudIn);
}

bool saveMap(dynamic_removal::save_mapRequest& req, dynamic_removal::save_mapResponse& res) {
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        if (req.destination == "" || req.resolution < 0.01) {
            cout << "invalid param" << endl;
            return false;
        }
        PointCloud::Ptr globalStaticCloud(new PointCloud());
        PointCloud::Ptr globalDynamicCloud(new PointCloud());
        PointCloud::Ptr globalStaticCloudDS(new PointCloud());
        PointCloud::Ptr globalDynamicCloudDS(new PointCloud());

        string saveMapDir = req.destination;
        for (int i = 0; i < staticSubmapVec.size(); ++i) {
            cout << "precessing submap " << i << endl;
            *globalStaticCloud += *staticSubmapVec[i];
            *globalDynamicCloud += *dynamicSubmapVec[i];
        }
        // down sample
        float resolution = req.resolution;
        pcl::VoxelGrid<PointXYZI> downsizeFilter;
        downsizeFilter.setLeafSize(resolution, resolution, resolution);
        downsizeFilter.setInputCloud(globalStaticCloud);
        downsizeFilter.filter(*globalStaticCloudDS);
        downsizeFilter.setInputCloud(globalDynamicCloud);
        downsizeFilter.filter(*globalDynamicCloudDS);
        pcl::io::savePCDFileBinary(saveMapDir + "/gloablStaticMapDS.pcd", *globalStaticCloudDS);
        pcl::io::savePCDFileBinary(saveMapDir + "/gloablDynamicMapDS.pcd", *globalDynamicCloudDS);

        // save tatal map by merging scan
        cout << "total scan size: " << staticScanVecFull.size() << endl;
        cout << "saving map by merging scans ..." << endl;
        PointCloud::Ptr globalStaticCloudMerge(new PointCloud());
        PointCloud::Ptr globalDynamicCloudMerge(new PointCloud());

        string _filename1 = "/home/eric/a_ros_ws/lio_sam_lrc/logs/original_pose.log";
        string _filename2 = "/home/eric/a_ros_ws/lio_sam_lrc/logs/updated_pose.log";
        std::fstream stream1(_filename1.c_str(), std::fstream::out);
        std::fstream stream2(_filename2.c_str(), std::fstream::out);

        for (int i = 0; i < staticScanVecFull.size(); ++i) {
            PointCloud::Ptr thisStaticScan(new PointCloud());
            PointCloud::Ptr thisDynamicScan(new PointCloud());
            if (!updatedOdomFull.empty()) {
                pcl::transformPointCloud(*staticScanVecFull[i], *thisStaticScan, updatedOdomFull[i]);
                pcl::transformPointCloud(*dynamicScanVecFull[i], *thisDynamicScan, updatedOdomFull[i]);
            }
            else {
                pcl::transformPointCloud(*staticScanVecFull[i], *thisStaticScan, originOdomFull[i]);
                pcl::transformPointCloud(*dynamicScanVecFull[i], *thisDynamicScan, originOdomFull[i]);
            }

            *globalStaticCloudMerge += *thisStaticScan;
            *globalDynamicCloudMerge += *thisDynamicScan;

            stream1 << originOdomFull[i](0, 3) << " " << originOdomFull[i](1, 3) << " " <<  originOdomFull[i](2, 3) << endl;
            stream2 << updatedOdomFull[i](0, 3) << " " << updatedOdomFull[i](1, 3) << " " <<  updatedOdomFull[i](2, 3) << endl;
        }
        pcl::io::savePCDFileBinary(saveMapDir + "/gloablStaticMapMerge.pcd", *globalStaticCloudMerge);
        pcl::io::savePCDFileBinary(saveMapDir + "/gloablDynamicMapMerge.pcd", *globalDynamicCloudMerge);
        cout << "done" << endl;
        cout << "****************************************************" << endl;

        res.success = true;
        return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dynamic_filter_offline");
    ros::NodeHandle nh;
    string _config;

    filterScanInDR = false;
    kdtreeFilter.reset(new pcl::KdTreeFLANN<PointXYZI>());
    kdtreeInitialFilter.reset(new pcl::KdTreeFLANN<PointXYZI>());
    kdtreeGround.reset(new pcl::KdTreeFLANN<PointXYZI>());
    curLocalMap.reset(new PointCloud());
    curLocalMapDS.reset(new PointCloud());
    lastStaticNogroundSubmap.reset(new PointCloud());
    lastDynamicSubmap.reset(new PointCloud());

    curLocalmapDownSizeFilter.setLeafSize(0.3, 0.3, 0.3);
    nh.getParam("dataset_config", _config);
    nh.getParam("filterScanInDR", filterScanInDR);
    nh.getParam("interval", interval);

    odomDelta = Eigen::Matrix4f::Identity();
    curPGOPose = Eigen::Matrix4f::Identity();

    cout << "use config file: " << endl;
    cout << "      " << _config << endl; 
    cout << "filterScanInDR: " << filterScanInDR << endl;
    df = new dynamicFilter(_config);
    groundSegSingleScan.reset(new travel::TravelGroundSeg<PointXYZI>());
    groundSegSingleScan->setParams(df->max_range, df->min_range, df->resolution, 
                                df->num_iter, df->num_lpr, df->num_min_pts, df->th_seeds, 
                                df->th_dist, df->th_outlier, df->th_normal, df->th_weight, 
                                df->th_lcc_normal_similiarity, df->th_lcc_planar_model_dist, df->th_obstacle,
                                df->refine_mode, df->visualization_flag);

    maxCapacity = df->submapMaxSize;
    std::thread handleThread(handleSubmapThread);
    // subscribe lidar poses and lidar cloud
    pubStaticCloud       = nh.advertise<sensor_msgs::PointCloud2>("/lio_sam/static_cloud", 1);
    pubdynamicCloud      = nh.advertise<sensor_msgs::PointCloud2>("/lio_sam/dynamic_cloud", 1);
    pubScanFilteredbyKNN = nh.advertise<sensor_msgs::PointCloud2>("/lio_sam/scanFiltered", 1);
    pubCornerCloudFiltered = nh.advertise<sensor_msgs::PointCloud2>("/lio_sam/cornerCloudFiltered", 1);
    pubSurfCloudFiltered = nh.advertise<sensor_msgs::PointCloud2>("/lio_sam/surfCloudFiltered", 1);
    pubStaticLocalMap    = nh.advertise<sensor_msgs::PointCloud2>("/lio_sam/staticLocalMap",1);
    pubDynamicFeature = nh.advertise<sensor_msgs::PointCloud2>("/lio_sam/dynamicFeature", 1);
    
    srvSaveMap = nh.advertiseService("dynamic_removal/save_map", &saveMap);
    
    ros::Subscriber subSlamInfo      = nh.subscribe<dynamic_removal::keyScan>("/lio_sam/slam_info", 100, &laserKeyFrameHandler, ros::TransportHints().tcpNoDelay());
    ros::Subscriber subPGOOdom       = nh.subscribe<nav_msgs::Odometry>("/aft_pgo_odom", 100, &PGOOdomHandler);
    ros::Subscriber subCornerFeature = nh.subscribe<sensor_msgs::PointCloud2>("laser_cloud_corner_last", 100, &cornerFeatureHandler);
    ros::Subscriber subSurfFeature   = nh.subscribe<sensor_msgs::PointCloud2>("laser_cloud_surf_last", 100, &surfFeatureHandler); 


    // signal(SIGINT, signal_handler); // to exit program when ctrl+c
    ros::MultiThreadedSpinner spinner(4);
    ros::spin();
    // handleThread.join();
    return 0;
}