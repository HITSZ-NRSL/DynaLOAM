// offline process
// input begin pose, end pose, interval,
// every time 
#include "dynamic_removal/dynamicFilter.h"
using namespace std;
// param 
string cloudFileDir;
string poseFile;
string saveFileDir;
bool saveCloudFlag; 
bool useScanMatching;
bool visRawCloud;
bool visGroundSeg;
bool visVisibilitySeg;
bool visClusterSeg; 
bool visDynamic;
bool visRevert;
bool visStatic;
bool visGround;
int maxPoseNum;
int startPose;
int endPose;
int frameInterval;
int submapMaxSize;
double leafSize;

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
PointCloud::Ptr rawGlobalmapDS;
pcl::KdTreeFLANN<PointXYZI>::Ptr kdtree;
pcl::KdTreeFLANN<PointXYZI>::Ptr kdtreeFilter;

// for scan-context descriptor and saving
vector<PointCloud::Ptr> rawKeyFrameVec;
vector<PointCloud::Ptr> filteredKeyFrameVec;
vector<Eigen::Matrix4f> filteredKeyFramePoses;

float pointDistance(PointXYZI p) {
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

void transformPoint(const PointXYZI& point_in, PointXYZI& point_out, const Eigen::Matrix4f& _trans) {
    point_out.x = _trans(0, 0) * point_in.x + _trans(0, 1) * point_in.y + _trans(0, 2) * point_in.z + _trans(0, 3);
    point_out.y = _trans(1, 0) * point_in.x + _trans(1, 1) * point_in.y + _trans(1, 2) * point_in.z + _trans(1, 3);
    point_out.z = _trans(2, 0) * point_in.x + _trans(2, 1) * point_in.y + _trans(2, 2) * point_in.z + _trans(2, 3);
}

void readLidarPoses() {
    fstream fin(poseFile);
    string _str;
    while (getline(fin, _str)) {
        stringstream ss(_str);
        vector<string> _res;
        string item;
        while(getline(ss, item, ' '))  {
            _res.push_back(item);
        }
        vector<float> _pose = {stof(_res[2]), stof(_res[3]), stof(_res[4]), stof(_res[5]), stof(_res[6]), stof(_res[7]), stof(_res[8])};
        Eigen::Quaternionf _q(_pose[6], _pose[3], _pose[4], _pose[5]);
        Eigen::Vector3f _trans(_pose[0], _pose[1], _pose[2]);
        Eigen::Isometry3f _isom = Eigen::Isometry3f::Identity();
        _isom.rotate(_q);
        _isom.pretranslate(_trans);
        lidarPoses.push_back(_isom);
    }
    fin.close();
    logger->info("totally read {} poses", lidarPoses.size());
    maxPoseNum = lidarPoses.size();
}

void readLidarFile() {
    if (startPose >= endPose || endPose > maxPoseNum) {
        logger->error("pose range not valid, shutdown");
        exit(-1);
    }
    for (int i = startPose; i < endPose; ++i) {
        if (i % frameInterval != 0) {
            continue;
        }
        logger->info("reading keyFrame {}", i);
        string cloudFile = cloudFileDir + to_string(i) + ".pcd";
        PointCloud::Ptr curCloud(new PointCloud());
        PointCloud::Ptr curCloudDS(new PointCloud());
        pcl_utils::readPCDFile(curCloud, cloudFile);

        downSizeFilter.setInputCloud(curCloud);
        downSizeFilter.filter(*curCloudDS);
        scanDSVec.push_back(curCloudDS);
        scanPoses.push_back(lidarPoses[i].matrix());

        // save raw cloud
        // PointCloud::Ptr curCloudTrans(new PointCloud());
        // PointCloud::Ptr curCloudDSTrans(new PointCloud());
        // pcl::transformPointCloud(*curCloud, *curCloudTrans, lidarPoses[i].matrix());
        // pcl::transformPointCloud(*curCloudDS, *curCloudDSTrans, lidarPoses[i].matrix());
        // *rawGlobalmap += *curCloudTrans;
        // *rawGlobalmapDS += *curCloudDSTrans;
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
        // visualization
        // pcl::visualization::PCLVisualizer vis_scan("vis_scan");
        // pcl::visualization::PointCloudColorHandlerGenericField<PointXYZI> single_color(cloud_vec[i], "intensity");
        // vis_scan.addPointCloud<PointXYZI>(cloud_vec[i], single_color, "vis_scan");
        // vis_scan.spin();
    }
}

int main(int argc, char **argv)
{
    logger = spdlog::stdout_color_mt("console_main");
    rawGlobalmap.reset(new PointCloud());
    rawGlobalmapDS.reset(new PointCloud());
    kdtree.reset(new pcl::KdTreeFLANN<PointXYZI>());
    kdtreeFilter.reset(new pcl::KdTreeFLANN<PointXYZI>());

    string jsonFile = "/home/eric/a_ros_ws/lio_sam_lrc/src/dynamic-removal/config/dynamicFilter_offline_hm.json";
    fstream f(jsonFile);
    json data = json::parse(f);
    cloudFileDir   = data["cloudFileDir"].get<string>();
    poseFile       = data["poseFile"].get<string>();
    saveFileDir    = data["saveFileDir"].get<string>();
    saveCloudFlag    = data["saveCloudFlag"].get<bool>(); 
    useScanMatching = data["useScanMatching"].get<bool>();
    visRawCloud    = data["visRawCloud"].get<bool>();
    visGroundSeg   = data["visGroundSeg"].get<bool>();
    visVisibilitySeg = data["visVisibilitySeg"].get<bool>();
    visClusterSeg  = data["visClusterSeg"].get<bool>(); 
    visDynamic     = data["visDynamic"].get<bool>();
    visRevert      = data["visRevert"].get<bool>();
    visStatic      = data["visStatic"].get<bool>();
    visGround      = data["visGround"].get<bool>();
    startPose      = data["startPose"].get<int>();
    endPose        = data["endPose"].get<int>();
    frameInterval  = data["frameInterval"].get<int>();
    submapMaxSize  = data["submapMaxSize"].get<int>();
    leafSize       = data["leafSize"].get<double>();
    downSizeFilter.setLeafSize(leafSize, leafSize, leafSize);

    readLidarPoses();
    readLidarFile();
    
    dynamicFilter df;
    vector<PointCloud::Ptr> cloud_vec;
    vector<Eigen::Matrix4f> pose_vec;
    int tatal_scan_num = scanDSVec.size();
    int submap_id = 0;
    for (int i = 0; i < tatal_scan_num; ++i) {
        cloud_vec.push_back(scanDSVec[i]);
        pose_vec.push_back(scanPoses[i]);
        PointCloud::Ptr rawKeyFrame(new PointCloud());
        pcl::copyPointCloud(*scanDSVec[i], *rawKeyFrame);
        rawKeyFrameVec.push_back(rawKeyFrame);
        filteredKeyFramePoses.push_back(scanPoses[i]);
        for (auto& p : *scanDSVec[i]) {
            p.intensity = 0;
        }
        if(cloud_vec.size() == submapMaxSize) {
            logger->info("processing submap {}", submap_id);
            submap_id++;
            PointCloud::Ptr staticSubmap(new PointCloud());
            PointCloud::Ptr dynamicSubmap(new PointCloud()); 
            if (useScanMatching) {
                 scanMatching(cloud_vec, pose_vec);
            }
            df.onlineProcess(cloud_vec, pose_vec);
            pcl::transformPointCloud(*(df.static_cluster_vis), *staticSubmap, pose_vec[0]);
            pcl::transformPointCloud(*(df.dynamic_cluster_vis), *dynamicSubmap, pose_vec[0]);
            staticSubmapVec.push_back(staticSubmap);
            dynamicSubmapVec.push_back(dynamicSubmap);

            // save the filtered keyframe
            // build kd-tree of submap
            // each raw key frame correspond to a lidar pose
            // staticSubmap is in map frame
            kdtreeFilter->setInputCloud(staticSubmap);
            // for every key frame
            for (int id =  0; id < rawKeyFrameVec.size(); ++id) {
                PointCloud::Ptr filteredKeyFrame(new PointCloud());
                // for every point in keyframe
                for(const auto& p : *rawKeyFrameVec[id]) {
                    PointXYZI _p;
                    transformPoint(p, _p, pose_vec[id]);
                     int K = 1;
                    vector<int> pointIdxNKNSearch(K);
                    vector<float> pointNKNSquaredDistance(K);
                    if (kdtreeFilter->nearestKSearch(_p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 && 
                        pointNKNSquaredDistance[0] < 0.1) {
                            filteredKeyFrame->push_back(p);
                    }
                }
                filteredKeyFrameVec.push_back(filteredKeyFrame);
                int frame_id = i - rawKeyFrameVec.size() +1  + id;
                logger->info("rawKeyFrame {} size: {}, filteredKeyFrame {} size: {}", frame_id, rawKeyFrameVec[id]->size(), frame_id, filteredKeyFrame->size());
            }       
            df.resetParameter();
            cloud_vec.clear();
            pose_vec.clear();
            rawKeyFrameVec.clear();
        }
    }

    PointCloud::Ptr globalStaticCloud(new PointCloud());
    PointCloud::Ptr globalDynamicCloud(new PointCloud());
    for (int i = 0; i < staticSubmapVec.size(); ++i) {
        *globalStaticCloud += *staticSubmapVec[i];
        *globalDynamicCloud += *dynamicSubmapVec[i];
    }

    pcl::visualization::PCLVisualizer vis_res("vis_res");
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> static_handler(globalStaticCloud, 0, 255, 0);
    vis_res.addPointCloud(globalStaticCloud, static_handler, "static");

    pcl::visualization::PointCloudColorHandlerCustom<PointXYZI> dynamic_handler(globalDynamicCloud, 255, 0, 0);
    vis_res.addPointCloud(globalDynamicCloud, dynamic_handler, "dynamic");

    vis_res.spin();

    if (saveCloudFlag) {
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        
        pcl::io::savePCDFileBinary(saveFileDir + "/gloablStaticMap.pcd", *globalStaticCloud);
        pcl::io::savePCDFileBinary(saveFileDir + "/gloablDynamicMap.pcd", *globalDynamicCloud);
        cout << "done" << endl;
        cout << "****************************************************" << endl;
    }

    return 0;
}
