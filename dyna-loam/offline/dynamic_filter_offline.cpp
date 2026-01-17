// offline process
// input begin pose, end pose, interval,
// every time 
#include "dynamic_removal/dynamicFilter.h"
#include "dynamic_removal/tictoc.hpp"
using namespace std;
using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
// param 
string cloudFileDir;
string poseFile;
string saveFileDir;
string configFile;

bool saveCloudFlag; 
bool isKittiFormat;
bool evaluation;

bool visRawCloud;
bool visGroundSeg;
bool visVisibilitySeg;
bool visClusterSeg; 
bool visDynamic;
bool visStatic;
bool visEvaluation;

int maxPoseNum;
int startPose;
int endPose;
int frameInterval;
int submapMaxSize;
double leafSize;
double z_max;

// variable
pcl::VoxelGrid<PointXYZI> downSizeFilter;
std::shared_ptr<spdlog::logger> logger;
vector<Eigen::Isometry3f> lidarPoses; // all lidar poses
vector<Eigen::Matrix4f> scanPoses; // chosen lidar poses
vector<PointCloud::Ptr> scanDSVec; // chosen lidar scans

vector<PointCloud::Ptr> staticSubmapVec;
vector<PointCloud::Ptr> dynamicSubmapVec;
vector<PointCloud::Ptr> groundSubmapVec;

PointCloud::Ptr rawGlobalmap;
pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree;

vector<int> DYNAMIC_CLASSES = {252, 253, 254, 255, 256, 257, 258, 259};

float pointDistance(PointXYZI p) {
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

void readLidarPoses() {
    fstream fin(poseFile);
    string _str;
    while (getline(fin, _str)) {
        float _data[12];
        stringstream strstream(_str);
        string _out;
        int i = 0;
        while(strstream >> _out) {
            float a = stof(_out);
            _data[i] = a;
            i++;
        }
        Eigen::Isometry3f t = Eigen::Isometry3f::Identity();
        t(0, 0) = _data[0], t(0, 1) = _data[1], t(0, 2) = _data[2], t(0, 3) = _data[3];
        t(1, 0) = _data[4], t(1, 1) = _data[5], t(1, 2) = _data[6], t(1, 3) = _data[7];
        t(2, 0) = _data[8], t(2, 1) = _data[9], t(2, 2) = _data[10], t(2, 3) = _data[11];
        lidarPoses.push_back(t);
    }
    fin.close();
    logger->info("totally read {} poses", lidarPoses.size());
    maxPoseNum = lidarPoses.size();
}

void readLidarFile() {
    if (startPose >= endPose) {
        logger->error("pose range not valid, shutdown");
        exit(-1);
    }
    for (int i = startPose; i < min(endPose, maxPoseNum); ++i) {
        if (i % frameInterval != 0) {
            continue;
        }
        logger->info("processing cloud {}", i);
        char fmt[7];
        std::snprintf(fmt, sizeof(fmt), "%06d", i);
        string cloudFile = cloudFileDir + string(fmt) + ".bin";
        PointCloud::Ptr curCloud(new PointCloud());
        PointCloud::Ptr curCloudDS(new PointCloud());
        pcl_utils::readBinFile(curCloud, cloudFile);
        pcl::KdTreeFLANN<PointXYZI>::Ptr scan_kdtree;
        scan_kdtree.reset(new pcl::KdTreeFLANN<PointXYZI>());

        for (auto& p : *curCloud) {
                uint32_t float2int      = static_cast<uint32_t>(p.intensity);
                uint32_t semantic_label = float2int & 0xFFFF;
                uint32_t inst_label     = float2int >> 16;
                bool     is_static      = true;
                for (int class_num: DYNAMIC_CLASSES) {
                    if (semantic_label == class_num) { 
                        is_static = false;
                    }
                    if (pointDistance(p) < 2) {
                        is_static = false;
                    }
                }

                // // if point close to ground, define static
                if (p.z < z_max && p.z > -2) {
                    is_static = true;
                }
                
                if(!is_static) {
                    p.intensity = 0.5;
                }
                else {
                    p.intensity = 0;
                }
        }
        downSizeFilter.setInputCloud(curCloud);
        downSizeFilter.filter(*curCloudDS);

        if(!evaluation) {
            for (auto &p : *curCloudDS) {
                p.intensity = 0;
            }
        }

        scanDSVec.push_back(curCloudDS);
        scanPoses.push_back(lidarPoses[i].matrix());

        PointCloud::Ptr curCloudTrans(new PointCloud());
        PointCloud::Ptr curCloudDSTrans(new PointCloud());
        pcl::transformPointCloud(*curCloud, *curCloudTrans, lidarPoses[i].matrix());
        *rawGlobalmap += *curCloudTrans;
    }
    logger->info("chosen lidar scan num: {}", scanDSVec.size());
}

void scanMatching(vector<PointCloud::Ptr>& cloud_vec, vector<Eigen::Matrix4f>& pose_vec) {
    for (int i = 1; i < cloud_vec.size(); ++i) {
        // trans cur cloud to last lidar frame
        // relative pose
        Eigen::Matrix4f transBetween = pose_vec[i-1].inverse() * pose_vec[i];
        kdtree->setInputCloud(cloud_vec[i-1]);
        for (auto &p : *cloud_vec[i]) {
            // trans p to last frame
            Eigen::Vector4f vp{p.x, p.y, p.z, 1.0};
            Eigen::Vector4f _vp = transBetween * vp;
            PointXYZI _p;
            _p.x = _vp(0);
            _p.y = _vp(1);
            _p.z = _vp(2);
            vector<int> ind;
            vector<float> dis;
            kdtree->radiusSearch(_p, 0.5, ind, dis);
            if(ind.empty() == true) {
                p.intensity += 200;
            }
            else {
                p.intensity += 0;
            }
        }
    }
}

void visualization(dynamicFilter& df) {
    if(visRawCloud) {
        pcl::visualization::PCLVisualizer vis_raw("vis_raw");
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> single_color(df.rawSubmapDS, "intensity");
        vis_raw.addPointCloud(df.rawSubmapDS, single_color, "static");
        while (!vis_raw.wasStopped()) {
            vis_raw.spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
	}

    if(visGroundSeg) {
        pcl::visualization::PCLVisualizer vis_ground("vis_ground");
        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> ground_handler(df.groundCloud, 0, 255, 0);
        vis_ground.addPointCloud(df.groundCloud, ground_handler, "ground");

        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> noground_handler(df.nogroundCloud, 255, 0, 0);
        vis_ground.addPointCloud(df.nogroundCloud, noground_handler, "noground");
        while (!vis_ground.wasStopped()) {
            vis_ground.spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
    }

    if(visVisibilitySeg) {
        pcl::visualization::PCLVisualizer vis_visibility("vis_dynamic");
        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> static_handler(df.vm.staticCloud, 0, 255, 0);
        vis_visibility.addPointCloud(df.vm.staticCloud, static_handler, "static");

        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> dynamic_handler(df.vm.dynamicCloud, 255, 0, 0);
        vis_visibility.addPointCloud(df.vm.dynamicCloud, dynamic_handler, "dynamic");

        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> revert_handler(df.vm.pcaRevertedCloud, 0, 255, 0);
        vis_visibility.addPointCloud(df.vm.pcaRevertedCloud, revert_handler, "revert");

        while (!vis_visibility.wasStopped()) {
            vis_visibility.spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
        vis_visibility.close();
    }

    if(visClusterSeg) {
        pcl::visualization::PCLVisualizer vis_cluster("vis_cluster");
        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> static_handler(df.static_cluster_vis, 0, 255, 0);
        vis_cluster.addPointCloud(df.static_cluster_vis, static_handler, "static");

        pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> dynamic_handler(df.dynamic_cluster_vis, 255, 0, 0);
        vis_cluster.addPointCloud(df.dynamic_cluster_vis, dynamic_handler, "dynamic");
        while (!vis_cluster.wasStopped()) {
            vis_cluster.spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
    }
}

int main(int argc, char **argv)
{
    logger = spdlog::stdout_color_mt("console_main");
    rawGlobalmap.reset(new PointCloud());
    kdtree.reset(new pcl::KdTreeFLANN<PointXYZI>());

    string jsonFile = "/home/eric/a_ros_ws/dyna_loam_ws/src/dyna-loam/config/dynamicFilter_offline.json";
    fstream f(jsonFile);
    json data = json::parse(f);
    cloudFileDir   = data["cloudFileDir"].get<string>();
    poseFile       = data["poseFile"].get<string>();
    saveFileDir    = data["saveFileDir"].get<string>();
    configFile = data["configFile"].get<string>();

    saveCloudFlag    = data["saveCloudFlag"].get<bool>(); 
    isKittiFormat = data["isKittiFormat"].get<bool>();
    evaluation = data["evaluation"].get<bool>();

    visRawCloud    = data["visRawCloud"].get<bool>();
    visGroundSeg   = data["visGroundSeg"].get<bool>();
    visVisibilitySeg = data["visVisibilitySeg"].get<bool>();
    visClusterSeg  = data["visClusterSeg"].get<bool>(); 
    visDynamic     = data["visDynamic"].get<bool>();
    visStatic      = data["visStatic"].get<bool>();
    visEvaluation  = data["visEvaluation"].get<bool>();

    startPose      = data["startPose"].get<int>();
    endPose        = data["endPose"].get<int>();
    frameInterval  = data["frameInterval"].get<int>();
    submapMaxSize  = data["submapMaxSize"].get<int>();
    leafSize       = data["leafSize"].get<double>();
    z_max          = data["z_max"].get<double>();
    downSizeFilter.setLeafSize(leafSize, leafSize, leafSize);

    readLidarPoses();
    readLidarFile();
    
    dynamicFilter df(configFile);
    vector<PointCloud::Ptr> cloud_vec;
    vector<Eigen::Matrix4f> pose_vec;
    int tatal_scan_num = scanDSVec.size();
    int submap_id = 0;
    double compute_time = 0;
    common::TicToc t1;
    
    for (int i = 0; i < tatal_scan_num; ++i) {
        cloud_vec.push_back(scanDSVec[i]);
        pose_vec.push_back(scanPoses[i]);
        if(cloud_vec.size() == submapMaxSize) {
            logger->info("processing submap {}", submap_id);
            submap_id++;
            PointCloud::Ptr staticCloud(new PointCloud());
            PointCloud::Ptr dynamicCloud(new PointCloud()); 
            t1.tic();
            df.onlineProcess(cloud_vec, pose_vec);
            compute_time += t1.toc();
            pcl::transformPointCloud(*(df.static_cluster_vis), *staticCloud, pose_vec[0]);
            pcl::transformPointCloud(*(df.dynamic_cluster_vis), *dynamicCloud, pose_vec[0]);
            // visualization
            visualization(df);
            staticSubmapVec.push_back(staticCloud);
            dynamicSubmapVec.push_back(dynamicCloud);
            df.resetParameter();
            cloud_vec.clear();
            pose_vec.clear();
        }
    }

    PointCloud::Ptr globalStaticCloud(new PointCloud());
    PointCloud::Ptr globalDynamicCloud(new PointCloud());
    for (int i = 0; i < staticSubmapVec.size(); ++i) {
        *globalStaticCloud += *staticSubmapVec[i];
        *globalDynamicCloud += *dynamicSubmapVec[i];
    }

    if (visDynamic || visStatic) {
        // Green
        pcl::visualization::PCLVisualizer vis_res("visDyna");
        if (visStatic) {
            pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> static_handler(globalStaticCloud, 0, 255, 0);
            vis_res.addPointCloud(globalStaticCloud, static_handler, "static");
        }
        if (visDynamic) {
            pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> dynamic_handler(globalStaticCloud, 255, 0, 0);
            vis_res.addPointCloud(globalDynamicCloud, dynamic_handler, "dynamic");
        }
        vis_res.spin();
    }

    // calculate results
    if (evaluation) {
        int total_static_points      = 0;
        int total_dynamic_points     = 0;
        int preserved_static_points  = 0;
        int preserved_dynamic_points = 0;

        // voxelization
        PointCloud::Ptr rawGlobalmapDS(new PointCloud());
        pcl::VoxelGrid<PointXYZI> downSizeFilterGlobal;
        downSizeFilterGlobal.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterGlobal.setInputCloud(rawGlobalmap);
        downSizeFilterGlobal.filter(*rawGlobalmapDS);
        logger->info("raw global map downsample size: {}", rawGlobalmapDS->size());

        PointCloud::Ptr filteredGlobalmapDS(new PointCloud());
        downSizeFilterGlobal.setInputCloud(globalStaticCloud);
        downSizeFilterGlobal.filter(*filteredGlobalmapDS);
        
        PointCloud::Ptr preservedDynamic(new PointCloud());
        PointCloud::Ptr wrongKillStatic(new PointCloud());
        PointCloud::Ptr rightStatic(new PointCloud());
        PointCloud::Ptr rightDynamic(new PointCloud());

        pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree;
        kdtree.reset(new pcl::KdTreeFLANN<PointXYZI>());
        kdtree->setInputCloud(filteredGlobalmapDS);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        for (auto& p : *rawGlobalmapDS) {
            kdtree->radiusSearch(p, 0.15, pointSearchInd, pointSearchSqDis);
            if(p.intensity == 0) {
                total_static_points++;
                if (pointSearchInd.empty()) {
                    // wrong kill
                    wrongKillStatic->push_back(p);
                }
                else {
                    // right static
                    rightStatic->push_back(p);
                    preserved_static_points++;
                }
            }
            else {
                total_dynamic_points++;
                if (pointSearchInd.empty()) {
                    // right kill
                    rightDynamic->push_back(p);
                }
                else {
                    // preserved dynamic
                    preservedDynamic->push_back(p);
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
        float compute_time_pre_frame = compute_time / float(maxPoseNum);

        logger->info("Precision rate: {}", PR);
        logger->info("Recall rate:    {}", RR);
        logger->info("F1 score :      {}", F1);
        logger->info("compute time pre frame: {}", compute_time_pre_frame * 1000);
        
        if (visEvaluation) {
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
            while (!vis_res.wasStopped()) {
                vis_res.spinOnce(100);
                boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            }
        }
        if (saveCloudFlag) {
            pcl::io::savePCDFileBinary(saveFileDir + "/preservedStatic.pcd", *rightStatic);
            pcl::io::savePCDFileBinary(saveFileDir + "/preservedDynamic.pcd", *preservedDynamic);
        }
    }
    
    if (saveCloudFlag) {
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        
        pcl::io::savePCDFileBinary(saveFileDir + "/globalStaticMap.pcd", *globalStaticCloud);
        pcl::io::savePCDFileBinary(saveFileDir + "/globalDynamicMap.pcd", *globalDynamicCloud);

        cout << "done" << endl;
        cout << "****************************************************" << endl;
    }
    return 0;
}
