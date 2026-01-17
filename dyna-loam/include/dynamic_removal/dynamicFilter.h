// 综合所有的离线操作
// 流程如下
// 可以接收slam模块的点云信息，也可以读取scan和相应的poses
// 将读取的scan形成submap，对submap地面滤除
// 将非地面点云和scan做visibility-check，粗略区分动态点云和静态点云，然后进行revert操作
// 保存点云的xyz，以及intensity，和相对帧数
// 对非地面点云进行有条件的聚类，区分动态聚类和静态聚类
// 对每一个动态聚类进一步排查，判断是否可能是误杀的聚类
// 将子图分成静态子图和动态子图，保存静态子图，丢弃动态部分
// 清空所有的中间变量，等待下一次的子图
#ifndef __DYNAMICFILTER_HH__
#define __DYNAMICFILTER_HH__

// std
#include <iostream>
#include <memory>
#include <cstdlib>
#include <math.h>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>

// log
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

// 3rd party
#include "common/pcl_utils/pcl_utils.h"
#include "common/nlohmann/json.hpp"
#include "dynamic_removal/visibilityMethod.h"
#include "dynamic_removal/travel/tgs.hpp"
#include "dynamic_removal/tictoc.hpp"

using namespace std;
using json = nlohmann::json;
using PointXYZI = pcl::PointXYZI;
using PointXYZIN = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointXYZI>;

float CLUSTER_DIS;
class dynamicFilter
{
public:
    // log
    std::shared_ptr<spdlog::logger> logger;

    // param
    string jsonFile;
    
    int rImgRow;
    int rImgCol;
    int nearbyCapacity; // visibility method revert module, serach near n points
    int submapMaxSize;

    float leafSize;
    float nearDisThresh;
    float revertMaxDis;
    float revertMinDis;
    float eigenValDiffThresh;
    float z_tolerance;
    float dynamicLowThresh;
    float dynamicHighThresh;
    float clusterDis;
    int maxDynamicClusterSize;

    // travel
    double max_range;
    double min_range;
    double resolution;
    int num_iter;
    int num_lpr;
    int num_min_pts;
    double th_seeds;
    double th_dist;
    double th_outlier;
    double th_normal;
    double th_weight;
    double th_lcc_normal_similiarity;
    double th_lcc_planar_model_dist;
    double th_obstacle;
    bool refine_mode;
    bool visualization_flag;

    // the frame of submap is the first lidar scan
    PointCloud::Ptr rawSubmapDS;
    PointCloud::Ptr groundCloud;
    PointCloud::Ptr nogroundCloud;
    PointCloud::Ptr labeledNogroundCloud;
    PointCloud::Ptr dynamic_cluster_vis;
    PointCloud::Ptr static_cluster_vis;
    PointCloud::Ptr staticNogroundCloud;

    vector<PointCloud::Ptr> scanDSVec;
    vector<Eigen::Matrix4f> poseVec;
    vector<Eigen::Matrix4f> poseVecIncreInv;

    // member variable
    vector<Eigen::Isometry3f> lidarPoses;

    vector<vector<Eigen::Vector3f>> lastClusterVec;

    std::shared_ptr<travel::TravelGroundSeg<PointXYZI>> travel_ground_seg;

    removert::VisibilityMethod vm;

public:
    dynamicFilter(/* args */);

    dynamicFilter(string _config);

    ~dynamicFilter();

    void readParam(json& data);

    void allocateMemory();

    void readLidarPoses();

    void onlineProcess(const vector<PointCloud::Ptr>& _cloud_vec, const vector<Eigen::Matrix4f>& _pose_vec);

    void groundSegment();

    void visibilitySeg();

    void clusterSeg();

    // judge if the cluster belong to dynamic cluster, cloud is the nogroundCloud, cloudIndices is the cluster 
    bool judgeDynamicCluster(PointCloud::Ptr cloud, pcl::PointIndices& cloudIndices);

    void resetParameter();

    float pointDistance(PointXYZI p) {
        return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    }

    float pointDistance(PointXYZI p1, PointXYZI p2) {
        return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }
};

#endif

