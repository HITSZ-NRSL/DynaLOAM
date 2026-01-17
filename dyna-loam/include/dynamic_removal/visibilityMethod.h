#ifndef __VISIBILITY_METHOD__HH__
#define __VISIBILITY_METHOD__HH__

// 输入为一个submap，一系列scan和他相应的位姿，能够做到利用rangeMat的不同，将submap区分为动态submap和静态submap

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/image_encodings.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
 
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>

#include <vector>
#include <cmath>
#include <set>
#include <algorithm>
#include <utility>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <unordered_set>
using namespace std;

namespace removert {
using PointXYZI = pcl::PointXYZI;
struct SphericalPoint
{
float az; // azimuth 
float el; // elevation
float r; // radius
};

inline float rad2deg(float radians) 
{ 
    return radians * 180.0 / M_PI; 
}

inline float deg2rad(float degrees) 
{ 
    return degrees * M_PI / 180.0; 
}

SphericalPoint cart2sph(const PointXYZI & _cp);

template<typename T>
cv::Mat convertColorMappedImg (const cv::Mat &_src, std::pair<T, T> _caxis)
{
T min_color_val = _caxis.first;
T max_color_val = _caxis.second;

cv::Mat image_dst;
image_dst = 255 * (_src - min_color_val) / (max_color_val - min_color_val);
image_dst.convertTo(image_dst, CV_8UC1);

cv::applyColorMap(image_dst, image_dst, cv::COLORMAP_JET);

return image_dst;
}

std::set<int> convertIntVecToSet(const std::vector<int> & v);

sensor_msgs::ImagePtr cvmat2msg(const cv::Mat &_img);

void pubRangeImg(cv::Mat& _rimg, sensor_msgs::ImagePtr& _msg, image_transport::Publisher& _publiser, std::pair<float, float> _caxis);
}

namespace removert {

class VisibilityMethod {

using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointXYZI>;


public:
    PointCloud::Ptr rawCloud;
    PointCloud::Ptr dynamicCloud;
    PointCloud::Ptr staticCloud;   
    PointCloud::Ptr pcaRevertedCloud; 
    PointCloud::Ptr searchCloud;

    std::shared_ptr<spdlog::logger> logger_;

    float rimg_color_min_;
    float rimg_color_max_;
    std::pair<float, float> kRangeColorAxis; // meter
    std::pair<float, float> kRangeColorAxisForDiff; // meter

    cv::Mat mapMat;
    cv::Mat scanMat; 
    cv::Mat diffMat;
    cv::Mat submapIndexMat;

    // 标记为动态点的点云在submap中的索引
    unordered_set<int> dynamicPointIdxSet;
    // 动态点云中revert的点云在submap中的索引
    unordered_set<int> revertedPointIdxSet;

    pcl::KdTreeFLANN<PointXYZI>::Ptr kdtreeRawSubmap;

    // param
    int nearbyCapacity{30};
    double z_tolerance{1};
    double revertMaxDis{50};
    double revertMinDis{10};
    double eigenValDiff{0.2};
    double nearDisThresh{0.2};


public:
    VisibilityMethod();
    ~VisibilityMethod();
    void allocateMemory();
    void getScanMat(cv::Mat& _scanMat);
    void getMapMat(cv::Mat& _matMat);
    void getDiffMat(cv::Mat& _diffMat);
    
    void distinguishCloud(PointCloud::Ptr _submapCloud, vector<PointCloud::Ptr> _localCloudVec, 
    vector<Eigen::Matrix4f> _localTransVec, std::pair<int, int>& _rImgShape);

    void submap2rangeMat(const PointCloud::Ptr _submap, const std::pair<int, int>& _rImgSize, Eigen::Matrix4f& _trans);
    void curScan2rangeMat(const PointCloud::Ptr _curScan, const std::pair<int, int>& _rImgSize);
    void calculateDiffMat(const std::pair<int, int>& _rImgSize);
    void revertStaticPoint();
    void computePCA(PointCloud::Ptr _cloudIn, Eigen::Matrix3d& _masterEigenVal, Eigen::Matrix3d& _masterEigenVec);
    void resetParameter();

    float pointDistance(PointXYZI p) {
        return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    }

    float pointDistance(PointXYZI p1, PointXYZI p2) {
        return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }

};

}
#endif