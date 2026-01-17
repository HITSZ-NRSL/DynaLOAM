// 构建一个subamp，然后选择最优的参数，利用八叉树的滤除方案对点云分割
#include <iostream>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
// octomap
#include <octomap/octomap.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry> 

// log
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "common/pcl_utils/pcl_utils.h"
#include "common/nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

int main(int argc, char const *argv[])
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn(new pcl::PointCloud<pcl::PointXYZI>());
    string cloudFileDir = "/home/eric/data/simulation_demo_dynamic/demo_dyna_4/velodyne/";
    string cloudFile;
    string poseFile = "/home/eric/data/simulation_demo_dynamic/demo_dyna_4/pose.txt";
    string jsonFile = "/home/eric/a_ros_ws/lio_sam_lrc/src/dynamic-removal/config/octomapConfig.json";
    string outputbt = "/home/eric/a_ros_ws/lio_sam_lrc/src/dynamic-removal/file/demo.bt";
    string outputGlobal = "/home/eric/a_ros_ws/lio_sam_lrc/src/dynamic-removal/file/global_once.pcd";
    string outputDynamic = "/home/eric/a_ros_ws/lio_sam_lrc/src/dynamic-removal/file/dynamic_once.pcd";
    string outputStatic = "/home/eric/a_ros_ws/lio_sam_lrc/src/dynamic-removal/file/static_once.pcd";

    fstream f(jsonFile);
    json data = json::parse(f);

    auto logger = spdlog::stdout_color_mt("console");    

    ifstream fin(poseFile);
    vector<Eigen::Isometry3f> poses;
    string str;
    while(getline(fin, str)) {
        float data[12];
        stringstream strstream(str);
        string out;
        int i = 0;
        while(strstream >> out) {
            float a = stof(out);
            data[i] = a;
            i++;
        }
        Eigen::Isometry3f t = Eigen::Isometry3f::Identity();
        t(0, 0) = data[0], t(0, 1) = data[1], t(0, 2) = data[2], t(0, 3) = data[3];
        t(1, 0) = data[4], t(1, 1) = data[5], t(1, 2) = data[6], t(1, 3) = data[7];
        t(2, 0) = data[8], t(2, 1) = data[9], t(2, 2) = data[10], t(2, 3) = data[11];
        poses.push_back(t);
    }
    fin.close();
    logger->info("totally has {} poses", poses.size());

    octomap::OcTree tree(data["resolution"].get<float>());
    tree.setProbHit(data["probability_hit"].get<float>());
    tree.setProbMiss(data["probability_miss"].get<float>());
    tree.setClampingThresMin(data["threshold_min"].get<float>());
    tree.setClampingThresMax(data["threshold_max"].get<float>());
    tree.setOccupancyThres(data["threshold_occupancy"].get<float>());



    vector<int> frameVec = data["keyFrames"].get<vector<int>>();
    bool saveSubmap = false;

    char fmt[6];
    // save raw cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr globalCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr curCloud(new pcl::PointCloud<pcl::PointXYZI>());
    if(data["saveRawCloud"].get<bool>() == true) {
        for (int i = 0; i < frameVec.size(); i++) {
            if(i >= poses.size())
                continue;
            logger->info("processing cloud {}", frameVec[i]);
            int frame = frameVec[i];
            sprintf(fmt, "%06d", frame);
            cloudFile = cloudFileDir + string(fmt) + ".bin";
            pcl_utils::readBinFile(curCloud, cloudFile);
            pcl::transformPointCloud(*curCloud, *curCloud, poses[frame].matrix());
            *globalCloud += *curCloud;
        }
        pcl_utils::savePCDFile(globalCloud, outputGlobal);
    }


    // ray-tracing cloud
    for(int i = 0; i < frameVec.size(); i++) {
        if(i >= poses.size())
            continue;
        logger->info("ray_tracing cloud  {}", frameVec[i]);
        int frame = frameVec[i];
        sprintf(fmt, "%06d", frame);
        cloudFile = cloudFileDir + string(fmt) + ".bin";
        pcl_utils::readBinFile(cloudIn, cloudFile);
        pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*cloudIn, *tmp, poses[frame].matrix());
        octomap::Pointcloud cloud_octo;
        for (auto &p : tmp->points) {
            cloud_octo.push_back(p.x, p.y, p.z);
        }
        tree.insertPointCloud( cloud_octo, 
                octomap::point3d(poses[frame](0, 3), poses[frame](1, 3), poses[frame](2, 3)));
    }

    tree.updateInnerOccupancy();
    // 存储octomap
    tree.writeBinary(outputbt);
    cout<<"done."<<endl;

    // distinguish dynamic cloud and static cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr 
    dynamicCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr 
    staticCloud(new pcl::PointCloud<pcl::PointXYZI>());  
    // 假设已经恢复了地面点，目前暂且考虑将0.1以下的点认为是地面点恢复  
    for (auto &p : globalCloud->points) {
        octomap::OcTreeNode* node = tree.search(p.x, p.y, p.z);
        if(node == nullptr) {
            continue;
        }
        if(tree.isNodeOccupied(node) || p.z < -1.1) {
            staticCloud->points.push_back(p);
        }
        else if(!tree.isNodeOccupied(node) && p.z > -1.1) {
            dynamicCloud->points.push_back(p);
        }
        else 
            staticCloud->points.push_back(p);
            
        // if(tree.isNodeOccupied(node)) {
        //     staticCloud->points.push_back(p);
        // }
        // else 
        //     dynamicCloud->points.push_back(p); 
    }
    staticCloud->height = 1; 
    staticCloud->width = staticCloud->size();
    dynamicCloud->height = 1;
    dynamicCloud->width = dynamicCloud->size();
    logger->info("staticCloud size: {}", staticCloud->size());
    logger->info("dynamicCloud size: {}", dynamicCloud->size());
    pcl_utils::savePCDFile(staticCloud, outputStatic);
    pcl_utils::savePCDFile(dynamicCloud, outputDynamic);
    // calculate RR and PR

    int preserved_static_num = 0;
    int total_static_num = 0;
    int preserved_dynamic_num = 0;
    int total_dynamic_num = 0;
    for (auto &p : *globalCloud) {
        if(p.intensity == 0.0)
            total_static_num++;
        else 
            total_dynamic_num++;
    }
    for (auto p : *staticCloud) {
        if(p.intensity == 100.0)
            preserved_dynamic_num++;
        else
            preserved_static_num++;
    }

    logger->info("total_static_num: {}", total_static_num);
    logger->info("total_dynamic_num: {}", total_dynamic_num);
    logger->info("preserved_static_num: {}", preserved_static_num);
    logger->info("preserved_dynamic_num: {}", preserved_dynamic_num);


    float PR = (float)preserved_static_num / (float)total_static_num;
    float RR = 1 - (float)preserved_dynamic_num / (float)total_dynamic_num;

    logger->info("PR: {}", PR);
    logger->info("RR: {}", RR);

    return 0;
}

