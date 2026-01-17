// segment cloud 

// std
#include <iostream>
#include <memory>
#include <algorithm>

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


// log
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "common/pcl_utils/pcl_utils.h"
#include "common/nlohmann/json.hpp"

#include "dynamic_removal/travel/aos.hpp"

using namespace std;
using json = nlohmann::json;

using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointXYZI>;

bool customCondition(const PointXYZI& seedPoint, const PointXYZI& candidatePoint, float squaredDistance) {
    float _x = abs(seedPoint.x - candidatePoint.x);
    float _y = abs(seedPoint.y - candidatePoint.y);
    float _xy = max(5.0 / sqrt(seedPoint.x * seedPoint.x  + seedPoint.y * seedPoint.y), 0.1);
    
    // float _z = _xy * (seedPoint.z - candidatePoint.z);
    float _z = abs(seedPoint.z - candidatePoint.z); 
    float _dis = sqrt(_x * _x + _y * _y + _z * _z);
    return _dis < 0.5; 
}

bool judgeDynamicCluster(PointCloud::Ptr cloud, pcl::PointIndices& cloudIndices) {
    int dynamic_count = 0;
    int cluster_count = cloudIndices.indices.size();
    for (auto id : cloudIndices.indices) {
        if(cloud->points[id].intensity == 0) {
            dynamic_count++;
        }
    }
    float dynamic_rate = float(dynamic_count) / float(cluster_count);
    
    return dynamic_rate > 0.2;
}

int main(int argc, char** argv) {
    string cloud_file_dir = "/home/eric/data/binhai/slam/file/";
    string json_file = "/home/eric/a_ros_ws/dyna_loam_ws/src/dyna-loam/config/segment.json";

    auto logger = spdlog::stdout_color_mt("console");  
    fstream f(json_file);
    json data = json::parse(f);
    string cloud_file = data["cloud_file"].get<string>();
    logger->info("processing cloud file: {}", cloud_file);
    cloud_file = cloud_file_dir + cloud_file;
    
    PointCloud::Ptr no_ground_submap (new PointCloud());
    pcl_utils::readPCDFile(no_ground_submap, cloud_file);
    logger->info("no_ground_submap cloud size: {}", no_ground_submap->points.size());

    
    pcl::search::KdTree<PointXYZI>::Ptr tree(new pcl::search::KdTree<PointXYZI>);
    tree->setInputCloud(no_ground_submap);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::ConditionalEuclideanClustering<PointXYZI> ec;
    ec.setClusterTolerance (1);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(10000);
    // ec.setSearchMethod(tree);
    ec.setInputCloud(no_ground_submap);
    ec.setConditionFunction(&customCondition);
    ec.segment(cluster_indices);
    logger->info("cluster size: {}", cluster_indices.size());

    // 可视化聚类点云
    PointCloud::Ptr no_ground_cloud_vis(new PointCloud());
    for (int i = 0; i < cluster_indices.size(); ++i) {
        for (int j = 0; j < cluster_indices[i].indices.size(); ++j) {
            int idx = cluster_indices[i].indices[j];
            PointXYZI p = no_ground_submap->points[idx];
            p.intensity = (i % 5) * 5;
            no_ground_cloud_vis->points.push_back(p);
        }
    }
    logger->info("cloud size: {}", no_ground_cloud_vis->size());

    pcl::visualization::PCLVisualizer vis("vis");
    pcl::visualization::PointCloudColorHandlerGenericField<PointXYZI> single_color(no_ground_cloud_vis, "intensity");
    vis.addPointCloud<PointXYZI>(no_ground_cloud_vis, single_color, "segment");
    vis.spin();


    PointCloud::Ptr dynamic_cluster_vis(new PointCloud());
    PointCloud::Ptr static_cluster_vis(new PointCloud());
    // 对每个点云簇筛选判断
    for (int i = 0; i < cluster_indices.size(); ++i) {
        bool dynamic_flag = judgeDynamicCluster(no_ground_submap, cluster_indices[i]);
        if (dynamic_flag) {
            for (int id : cluster_indices[i].indices) {
                dynamic_cluster_vis->points.push_back(no_ground_submap->points[id]);
            }
        }
        else {
            for (int id : cluster_indices[i].indices) {
                static_cluster_vis->points.push_back(no_ground_submap->points[id]);
            }
        }
    }
    pcl::visualization::PCLVisualizer vis2("vis2");

    pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> static_handler(static_cluster_vis, 0, 255, 0);
    vis2.addPointCloud(static_cluster_vis, static_handler, "static");

    pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> dynamic_handler(dynamic_cluster_vis, 255, 0, 0);
    vis2.addPointCloud(dynamic_cluster_vis, dynamic_handler, "dynamic");

    vis2.spin();
    

}