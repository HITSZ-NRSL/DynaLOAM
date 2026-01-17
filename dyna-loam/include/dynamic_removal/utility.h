#ifndef __UTILITY_REMOVERT_H__
#define __UTILITY_REMOVERT_H__

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

// #include <filesystem> // requires gcc version >= 8

namespace removert{
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

#endif