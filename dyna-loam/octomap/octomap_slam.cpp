#include <iostream>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
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
#include "dynamic_removal/tictoc.hpp"

using namespace std;
using json = nlohmann::json;

using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointXYZI>;

void exportSubmap(octomap::OcTree &tree, PointCloud::Ptr cloudRaw, 
PointCloud::Ptr cloudStatic, PointCloud::Ptr cloudDynamic) {
    for (auto &p : cloudRaw->points) {
        octomap::OcTreeNode* node = tree.search(p.x, p.y, p.z);
        if(node == nullptr) {
            continue;
        }
        if(tree.isNodeOccupied(node)) {
            cloudStatic->points.push_back(p);
        }
        else if(!tree.isNodeOccupied(node)) {
            cloudDynamic->points.push_back(p);
        }
        else 
            cloudStatic->points.push_back(p);
    }
}


int main(int argc, char const *argv[])
{
    PointCloud::Ptr cloudIn(new PointCloud());
    string jsonFile = "/home/eric/a_ros_ws/dyna_loam_ws/src/dyna-loam/config/octomapConfig.json";
    fstream f(jsonFile);
    json data = json::parse(f);
    auto logger = spdlog::stdout_color_mt("console");    
    string cloudFileDir  = data["cloudFileDir"].get<string>();
    string poseFile      = data["poseFile"].get<string>();
    string outputBt      = data["outputBt"].get<string>();
    string outputGlobal  = data["outputGlobal"].get<string>();
    string outputDynamic = data["outputDynamic"].get<string>();
    string outputStatic  = data["outputStatic"].get<string>(); 
    int startPose        = data["startPose"].get<int>();
    int endPose          = data["endPose"].get<int>();
    int submapCapacity   = data["submapCapacity"].get<int>();
    bool saveFlag        = data["saveFlag"].get<bool>();
    int frameInterval    = data["frameInterval"].get<int>();
    double leafSize      = data["leafSize"].get<double>();
    bool evaluationFlag  = data["evaluation"].get<bool>();

    octomap::OcTree tree(data["resolution"].get<float>());
    tree.setProbHit(data["probability_hit"].get<float>());
    tree.setProbMiss(data["probability_miss"].get<float>());
    tree.setClampingThresMin(data["threshold_min"].get<float>());
    tree.setClampingThresMax(data["threshold_max"].get<float>());
    tree.setOccupancyThres(data["threshold_occupancy"].get<float>());

    pcl::VoxelGrid<PointXYZI> downSizeFilter; 
    downSizeFilter.setLeafSize(leafSize, leafSize, leafSize);

    ifstream fin(poseFile);
    vector<Eigen::Isometry3f> poses;
    string str;
    while(getline(fin, str)) {
        float _data[12];
        stringstream strstream(str);
        string out;
        int i = 0;
        while(strstream >> out) {
            float a = stof(out);
            _data[i] = a;
            i++;
        }
        Eigen::Isometry3f t = Eigen::Isometry3f::Identity();
        t(0, 0) = _data[0], t(0, 1) = _data[1], t(0, 2) = _data[2], t(0, 3) = _data[3];
        t(1, 0) = _data[4], t(1, 1) = _data[5], t(1, 2) = _data[6], t(1, 3) = _data[7];
        t(2, 0) = _data[8], t(2, 1) = _data[9], t(2, 2) = _data[10], t(2, 3) = _data[11];
        poses.push_back(t);
    }
    fin.close();
    logger->info("totally has {} poses", poses.size());

    int posesNum = poses.size();
    vector<PointCloud::Ptr> RawSubmapVec;
    vector<PointCloud::Ptr> StaticSubmapVec;
    vector<PointCloud::Ptr> DynamicSubmapVec;

    int curSubmapSize = 0;
    vector<Eigen::Matrix4f> submapPoseVec;
    vector<Eigen::Matrix4f> tatalSubmapPoseVec;

    char fmt[7];
    PointCloud::Ptr curRawSubmap(new PointCloud());
    int submapIdx = 0;
    float compute_time = 0;
    common::TicToc t1;
    for(int i = startPose; i < endPose && i < posesNum; i++) {
        if (i % frameInterval != 0) 
            continue;
        // read cloud file
        logger->info("processing cloud {}", i);
        sprintf(fmt, "%06d", i);
        string cloudFile = cloudFileDir + string(fmt) + ".bin";
        PointCloud::Ptr curCloud(new PointCloud());
        PointCloud::Ptr curCloudDS(new PointCloud());
        pcl_utils::readBinFile(curCloud, cloudFile);
        // downSample
        downSizeFilter.setInputCloud(curCloud);
        downSizeFilter.filter(*curCloudDS);

        pcl::transformPointCloud(*curCloudDS, *curCloudDS, poses[i].matrix());
        submapPoseVec.push_back(poses[i].matrix());
        *curRawSubmap += *curCloudDS;

        // insert cloud to tree
        t1.tic();
        octomap::Pointcloud cloudOcto;
        for (const auto &p : curCloudDS->points) {
            cloudOcto.push_back(p.x, p.y, p.z);
        }
        tree.insertPointCloud(cloudOcto, octomap::point3d(poses[i](0, 3), 
        poses[i](1, 3), poses[i](2, 3)));
        curSubmapSize++;
        // submap ray-tracing
        // if(curSubmapSize < submapCapacity) {
        //     compute_time += t1.toc();
        //     continue;
        // }
        // else {
        //     logger->info("processing submap {}", submapIdx);
        //     submapIdx++;
        //     curSubmapSize = 0;
        //     tree.updateInnerOccupancy();
        //     PointCloud::Ptr curStaticSubmap(new PointCloud());
        //     PointCloud::Ptr curDynamicSubmap(new PointCloud());
        //     PointCloud::Ptr thisRawSubmap(new PointCloud());
        //     pcl::copyPointCloud(*curRawSubmap, *thisRawSubmap);
        //     exportSubmap(tree, curRawSubmap, curStaticSubmap, curDynamicSubmap);
        //     RawSubmapVec.push_back(thisRawSubmap);
        //     StaticSubmapVec.push_back(curStaticSubmap);
        //     DynamicSubmapVec.push_back(curDynamicSubmap);

        //     // save submap cloud and pose
        //     tatalSubmapPoseVec.push_back(submapPoseVec.front());

        //     submapPoseVec.clear();
        //     // clear 
        //     tree.clear();
        //     curRawSubmap->clear();
        // }
        compute_time += t1.toc();
    }

    tree.updateInnerOccupancy();
    PointCloud::Ptr curStaticSubmap(new PointCloud());
    PointCloud::Ptr curDynamicSubmap(new PointCloud());
    PointCloud::Ptr thisRawSubmap(new PointCloud());
    pcl::copyPointCloud(*curRawSubmap, *thisRawSubmap);
    exportSubmap(tree, curRawSubmap, curStaticSubmap, curDynamicSubmap);
    RawSubmapVec.push_back(thisRawSubmap);
    StaticSubmapVec.push_back(curStaticSubmap);
    DynamicSubmapVec.push_back(curDynamicSubmap);

    logger->info("tatal Submap num: {}", RawSubmapVec.size());

    // visualization
    PointCloud::Ptr rawGlobalCloud(new PointCloud());
    PointCloud::Ptr staticGlobalCloud(new PointCloud());
    PointCloud::Ptr dynamicGlobalCloud(new PointCloud());
    for (int i = 0; i < RawSubmapVec.size(); ++i) {
        *rawGlobalCloud += *RawSubmapVec[i];
    }

    for (int i = 0; i < StaticSubmapVec.size(); ++i) {
        *staticGlobalCloud += *StaticSubmapVec[i];
    }

    for (int i = 0; i < DynamicSubmapVec.size(); ++i) {
        *dynamicGlobalCloud += *DynamicSubmapVec[i];
    }

    logger->info("raw global cloud size: {}", rawGlobalCloud->size());
    logger->info("static global cloud size: {}", staticGlobalCloud->size());
    logger->info("dynamic global cloud size: {}", dynamicGlobalCloud->size());

    // calculate PP and PR
    if (evaluationFlag) {
        std::vector<int> DYNAMIC_CLASSES = {252, 253, 254, 255, 256, 257, 258, 259};
        int total_static_points      = 0;
        int total_dynamic_points     = 0;
        int preserved_static_points  = 0;
        int preserved_dynamic_points = 0;
        // voxelization
        pcl::PointCloud<PointXYZI>::Ptr rawGlobalmapDS(new pcl::PointCloud<PointXYZI>());
        pcl::VoxelGrid<PointXYZI> downSizeFilterGlobal;
        downSizeFilterGlobal.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterGlobal.setInputCloud(rawGlobalCloud);
        downSizeFilterGlobal.filter(*rawGlobalmapDS);
        cout << "raw global map downsample size: " << rawGlobalmapDS->size() << endl;

        pcl::PointCloud<PointXYZI>::Ptr filteredGlobalmapDS(new pcl::PointCloud<PointXYZI>());
        downSizeFilterGlobal.setInputCloud(staticGlobalCloud);
        downSizeFilterGlobal.filter(*filteredGlobalmapDS);
        
        pcl::PointCloud<PointXYZI>::Ptr preservedDynamic(new pcl::PointCloud<PointXYZI>());
        pcl::PointCloud<PointXYZI>::Ptr wrongKillStatic(new pcl::PointCloud<PointXYZI>());
        pcl::PointCloud<PointXYZI>::Ptr rightStatic(new pcl::PointCloud<PointXYZI>());
        pcl::PointCloud<PointXYZI>::Ptr rightDynamic(new pcl::PointCloud<PointXYZI>());

        pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree;
        kdtree.reset(new pcl::KdTreeFLANN<PointXYZI>());
        kdtree->setInputCloud(filteredGlobalmapDS);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        for (auto& _p : rawGlobalmapDS->points) {
            kdtree->radiusSearch(_p, 0.15, pointSearchInd, pointSearchSqDis);
            uint32_t float2int      = static_cast<uint32_t>(_p.intensity);
            uint32_t semantic_label = float2int & 0xFFFF;
            uint32_t inst_label     = float2int >> 16;
            bool     is_static      = true;
            for (int class_num: DYNAMIC_CLASSES) {
                if (semantic_label == class_num) { 
                    is_static = false;
                }
            }
            if (is_static) {
                total_static_points++;
                if (pointSearchInd.empty()) {
                    // wrong kill
                    wrongKillStatic->push_back(_p);
                    
                }
                else {
                    // right static
                    rightStatic->push_back(_p);
                    preserved_static_points++;
                }
            }
            else {
                total_dynamic_points++;
                if (pointSearchInd.empty()) {
                    // right kill
                    rightDynamic->push_back(_p);
                }
                else {
                    // preserved dynamic
                    preservedDynamic->push_back(_p);
                    preserved_dynamic_points++;
                }
            }
        } 
        cout << "total_static_points: " << total_static_points << endl;
        cout << "total_dynamic_points:" << total_dynamic_points << endl;
        cout << "preserved_static_points:" << preserved_static_points << endl;
        cout << "preserved_dynamic_points:" << preserved_dynamic_points << endl;
        
        float PR = (float)preserved_static_points / (float)total_static_points;
        float RR = 1 - (float)preserved_dynamic_points / (float)total_dynamic_points;
        float F1 = 2 * PR * RR / (PR + RR);
        float compute_time_pre_frame = compute_time / (float)posesNum;

        cout << "Precision rate: " << PR << endl;
        cout << "Recall rate:    " << RR << endl;
        cout << "F1 score :      " << F1 << endl;
        cout << "compute time per frame: " << compute_time_pre_frame * 1000 << endl; 

        // Green
        pcl::visualization::PCLVisualizer vis_res("vis_res");
        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> static_handler(rightStatic, 0, 255, 0);
        vis_res.addPointCloud(rightStatic, static_handler, "static");

        // White
        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> dynamic_handler(rightDynamic, 255, 255, 255);
        vis_res.addPointCloud(rightDynamic, dynamic_handler, "dynamic");

        // Red
        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> pd_handler(preservedDynamic, 255, 0, 0);
        vis_res.addPointCloud(preservedDynamic, pd_handler, "pd");

        // Blue
        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> wk_handler(wrongKillStatic, 0, 0, 255);
        vis_res.addPointCloud(wrongKillStatic, wk_handler, "wk");

        vis_res.spin();
    }
        

    // save cloud
    if(!saveFlag) {
        logger->info("not save global cloud");
    }
    else {
        PointCloud::Ptr globalRawCloud(new PointCloud());
        PointCloud::Ptr globalStaticCloud(new PointCloud());
        PointCloud::Ptr globalDynamicCloud(new PointCloud());
        PointCloud::Ptr globalRawCloudDS(new PointCloud());
        PointCloud::Ptr globalStaticCloudDS(new PointCloud());
        PointCloud::Ptr globalDynamicCloudDS(new PointCloud());

        pcl::VoxelGrid<PointXYZI> voxelDownsize;
        voxelDownsize.setLeafSize(0.2, 0.2, 0.2);
        
        for(auto &cloud : RawSubmapVec) {
            *globalRawCloud += *cloud;
        }
        globalRawCloud->height = 1; 
        globalRawCloud->width = globalRawCloud->size();
        voxelDownsize.setInputCloud(globalRawCloud);
        voxelDownsize.filter(*globalRawCloudDS);
        pcl_utils::savePCDFile(globalRawCloudDS, outputGlobal);

        for(auto &cloud : StaticSubmapVec) {
            *globalStaticCloud += *cloud;
        }
        globalStaticCloud->height = 1; 
        globalStaticCloud->width = globalStaticCloud->size();

        voxelDownsize.setInputCloud(globalStaticCloud);
        voxelDownsize.filter(*globalStaticCloudDS);
        pcl_utils::savePCDFile(globalStaticCloudDS, outputStatic);

        for(auto &cloud : DynamicSubmapVec) {
            *globalDynamicCloud += *cloud;
        }
        globalDynamicCloud->height = 1; 
        globalDynamicCloud->width = globalDynamicCloud->size();

        voxelDownsize.setInputCloud(globalDynamicCloud);
        voxelDownsize.filter(*globalDynamicCloudDS);
        pcl_utils::savePCDFile(globalDynamicCloudDS, outputDynamic);
        logger->info("globalRawCloud size: {}", globalRawCloud->size());
        logger->info("globalStaticCloud size: {}", globalStaticCloud->size());
        logger->info("globalDynamicCloud size: {}", globalDynamicCloud->size());
    }
    return 0;
}


