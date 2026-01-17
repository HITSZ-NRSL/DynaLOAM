// generate submap of slam results of lio-sam, just for vision
// std
#include <iostream>
#include <memory>

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

// log
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "common/pcl_utils/pcl_utils.h"
#include "common/nlohmann/json.hpp"

#include "dynamic_removal/visibilityMethod.h"
#include "dynamic_removal/travel/tgs.hpp"

using namespace std;
using json = nlohmann::json;

using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointXYZI>;

int main(int argc, char** argv) {
    
    string cloudFileDir = "/home/eric/data/binhai/slam/velodyne/";
    string poseFile = "/home/eric/data/binhai/slam/pose.txt";
    string saveFileDir = "/home/eric/data/binhai/slam/file/";
    string jsonFile = "/home/eric/a_ros_ws/lio_sam_lrc/src/dynamic-removal/config/submap.json";
    string cloudFile;

    auto logger = spdlog::stdout_color_mt("console");  

    fstream f(jsonFile);
    json data = json::parse(f);
    bool saveCloud = data["saveCloud"].get<bool>();
    int interval = data["interval"].get<int>();
    double leafSize = data["leafSize"].get<double>();

    pcl::VoxelGrid<PointXYZI> downSizeFilter;
    downSizeFilter.setLeafSize(leafSize, leafSize, leafSize);
    

    // read pose.txt
    ifstream fin(poseFile);
    vector<Eigen::Isometry3f> poses;
    vector<Eigen::Matrix4f> posesInv;
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
        posesInv.push_back(t.matrix().inverse());
    }
    fin.close();
    logger->info("totally has {} poses", poses.size());
    int poseNum = poses.size();

    // read cloud file
    char fmt[6];
    PointCloud::Ptr RawSubmap(new PointCloud());  

    vector<PointCloud::Ptr> scanVec;
    vector<Eigen::Matrix4f> poseInvVec;

    int startPose = data["startPose"].get<int>();
    int endPose   = data["endPose"].get<int>(); 

    for (int i = startPose; i < poseNum && i < endPose; i++) {
        if (i % interval != 0)
            continue;
        logger->info("processing cloud {}", i);
        sprintf(fmt, "%06d", i);
        cloudFile = cloudFileDir + string(fmt) + ".bin";
        PointCloud::Ptr curCloud(new PointCloud());
        PointCloud::Ptr curCloudDS(new PointCloud());
        pcl_utils::readBinFile(curCloud, cloudFile);

        for (int k = 0; k < curCloud->size(); ++k) {
            curCloud->points[k].intensity = 10 * i;
        }
        downSizeFilter.setInputCloud(curCloud);
        downSizeFilter.filter(*curCloudDS);
        scanVec.push_back(curCloudDS);
        poseInvVec.push_back(posesInv[i]);
           
        pcl::transformPointCloud(*curCloudDS, *curCloudDS, poses[i].matrix());
        
        *RawSubmap += *curCloudDS;
    }

    // ground cloud filter
    // pcl::PassThrough<PointXYZI> passZ;
    // passZ.setFilterFieldName("z");
    // passZ.setFilterLimits(-2, 50);
    // passZ.setInputCloud(RawSubmap);
    // passZ.filter(*RawSubmap);

    // pcl::PassThrough<PointXYZI> passX;
    // passX.setFilterFieldName("x");
    // passX.setFilterLimits(-20, 20);
    // passX.setInputCloud(RawSubmap);
    // passX.filter(*RawSubmap);

    // pcl::PassThrough<PointXYZI> passY;
    // passY.setFilterFieldName("y");
    // passY.setFilterLimits(-20, 0);
    // passY.setInputCloud(RawSubmap);
    // passY.filter(*RawSubmap);
    

    // 对于scanMat和submap，如果submapMat上的点比对应的scanMat上的点更近的话，那么submap上的点是动态的，应该滤除

    std::shared_ptr<travel::TravelGroundSeg<PointXYZI>> travel_ground_seg;
    travel_ground_seg.reset(new travel::TravelGroundSeg<PointXYZI>());

    travel_ground_seg->setParams(50, 0, 5, 
                                3, 20, 10, 0.4, 
                                0.3, 0.3, 0.707, 1.5, 
                                1.5, 1.5, 1.5,
                                true, true);
    
    PointCloud::Ptr raw_pc(new PointCloud());
    PointCloud::Ptr ground_pc(new PointCloud());
    PointCloud::Ptr noground_pc(new PointCloud());
    double ground_seg_time = 0;
    pcl::copyPointCloud(*RawSubmap, *raw_pc);
    travel_ground_seg->estimateGround(*raw_pc, *ground_pc, *noground_pc, ground_seg_time);

    

    logger->info("ground seg time: {}", ground_seg_time);

    bool visGround = data["visGround"].get<bool>();

    // if(visGround) {
    //     pcl::visualization::PCLVisualizer vis("vis");
    //     pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> noground_handler(noground_pc, 0.0, 255.0, 0.0);  // g    // b
    //     vis.addPointCloud(noground_pc, noground_handler, "noground_pc");
    //     pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> ground_handler(ground_pc, 255.0, 0.0, 0.0);  // g    // b
    //     vis.addPointCloud(ground_pc, ground_handler, "ground_pc");
    //     vis.spin();
    // }

    removert::VisibilityMethod vm;
    int rImgRow = data["rImgRow"].get<int>();
    int rImgCol = data["rImgCol"].get<int>();
    int nearbyCapacity = data["nearbyCapacity"].get<int>();
    double nearDisThresh = data["nearDisThresh"].get<double>();
    double revertMaxDis = data["revertMaxDis"].get<double>();
    double revertMinDis = data["revertMinDis"].get<double>();
    double eigenValDiff = data["eigenValDiff"].get<double>();

    std::pair<int, int> rimgShape{rImgRow, rImgCol};
    vm.nearbyCapacity = nearbyCapacity;
    vm.nearDisThresh = nearDisThresh;
    vm.revertMaxDis = revertMaxDis;
    vm.revertMinDis = revertMinDis;
    vm.eigenValDiff = eigenValDiff;
    vm.distinguishCloud(noground_pc, scanVec, poseInvVec, rimgShape);
    
    // cv::Mat mapMat;
    // cv::Mat scanMat;
    // cv::Mat diffMat;
    // vm.getMapMat(mapMat);
    // vm.getScanMat(scanMat);
    // vm.getDiffMat(diffMat);

    // cv::imshow("mapMat", mapMat);
    // cv::imshow("scanMat", scanMat);
    // cv::imshow("diffMat", diffMat);
    // cv::waitKey(0);
    logger->info("ground_pc cloud size: {}", ground_pc->size());
    logger->info("noground pc cloud size: {}", noground_pc->size());
    logger->info("dynamic cloud size: {}", vm.dynamicCloud->size());
    logger->info("static cloud size: {}", vm.staticCloud->size());
    logger->info("pca reverted cloud size: {}", vm.pcaRevertedCloud->size());

    bool visStatic = data["visStatic"].get<bool>();
    bool visDynamic = data["visDynamic"].get<bool>();
    bool visRevert = data["visRevert"].get<bool>();

    if(visStatic || visDynamic || visGround) {
        pcl::visualization::PCLVisualizer vis("vis");
        if (visStatic) {
            pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> static_handler(vm.staticCloud, 0.0, 255.0, 0.0);  // g   
            vis.addPointCloud(vm.staticCloud, static_handler, "static");
        }
        if (visDynamic) {
            pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> dynamic_handler(vm.dynamicCloud, 255.0, 0.0, 0.0);  // r
            vis.addPointCloud(vm.dynamicCloud, dynamic_handler, "dynamic");
        }
        if (visRevert) {
            pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> revert_handler(vm.pcaRevertedCloud, 0.0, 0.0, 255.0);  // b
            vis.addPointCloud(vm.pcaRevertedCloud, revert_handler, "revert");
        }
        if (visGround) {
            pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> ground_handler(ground_pc, 255.0, 255.0, 0.0);  // g    // b
            vis.addPointCloud(ground_pc, ground_handler, "ground_pc");
        }

        vis.spin();
    }

    // 非地面点云需要做一个聚类，然后判断每个聚类簇的动态属性

    // 下面尝试基于rangeMap的聚类方法

    PointCloud::Ptr labeled_noground_cloud(new PointCloud());
    // 动态点云的intensity为0，静态点云的intensity为100
    for (auto p : vm.staticCloud->points) {
        p.intensity = 100;
        labeled_noground_cloud->points.push_back(p);
    }

    for (auto p : vm.dynamicCloud->points) {
        p.intensity = 0;
        labeled_noground_cloud->points.push_back(p);
    }
    labeled_noground_cloud->width = labeled_noground_cloud->size();
    labeled_noground_cloud->height = 1;
    labeled_noground_cloud->is_dense = true;

    if (saveCloud) {
        string cloudName = "raw_submap_" + to_string(startPose) + "_" + to_string(endPose) + ".pcd";
        cloudName = saveFileDir + cloudName;
        pcl_utils::savePCDFile(RawSubmap, cloudName);

        cloudName = "noground_submap_" + to_string(startPose) + "_" + to_string(endPose) + ".pcd";
        cloudName = saveFileDir + cloudName;
        pcl_utils::savePCDFile(noground_pc, cloudName);

        cloudName = "labeled_noground_submap_" + to_string(startPose) + "_" + to_string(endPose) + ".pcd";
        cloudName = saveFileDir + cloudName;
        pcl_utils::savePCDFile(labeled_noground_cloud, cloudName);
    }


    return 0;
}