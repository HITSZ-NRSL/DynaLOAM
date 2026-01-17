#include "dynamic_removal/dynamicFilter.h"

dynamicFilter::dynamicFilter() {
    #ifdef JSON_FILE
        jsonFile = JSON_FILE;
    #endif
    
    fstream f(jsonFile);
    json data = json::parse(f);
    readParam(data);
    allocateMemory();
}

dynamicFilter::dynamicFilter(string _config) {
    fstream f(_config);
    json data = json::parse(f);
    readParam(data);
    allocateMemory();
}

dynamicFilter::~dynamicFilter() {
    
}

void dynamicFilter::readParam(json& data) {
    max_range = data["max_range"].get<double>();
    min_range = data["min_range"].get<double>();
    resolution = data["resolution"].get<double>();
    num_iter = data["num_iter"].get<int>(); 
    num_lpr = data["num_lpr"].get<int>();
    num_min_pts = data["num_min_pts"].get<int>();
    th_seeds = data["th_seeds"].get<double>();
    th_dist = data["th_dist"].get<double>();
    th_outlier = data["th_outlier"].get<double>();
    th_normal = data["th_normal"].get<double>();
    th_weight = data["th_weight"].get<double>();
    th_lcc_normal_similiarity = data["th_lcc_normal_similiarity"].get<double>();
    th_lcc_planar_model_dist = data["th_lcc_planar_model_dist"].get<double>();
    th_obstacle = data["th_obstacle"].get<double>();
    refine_mode = data["refine_mode"].get<bool>();;

    rImgRow = data["rImgRow"].get<int>();
    rImgCol = data["rImgCol"].get<int>();
    nearbyCapacity = data["nearbyCapacity"].get<int>();
    z_tolerance = data["z_tolerance"].get<float>();
    nearDisThresh = data["nearDisThresh"].get<float>();
    revertMaxDis = data["revertMaxDis"].get<float>();
    revertMinDis = data["revertMinDis"].get<float>();
    eigenValDiffThresh = data["eigenValDiffThresh"].get<float>();

    submapMaxSize = data["submapMaxSize"].get<int>();
    dynamicLowThresh = data["dynamicLowThresh"].get<float>();
    dynamicHighThresh = data["dynamicHighThresh"].get<float>();
    clusterDis = data["clusterDis"].get<float>();
    maxDynamicClusterSize = data["maxDynamicClusterSize"].get<int>();
    CLUSTER_DIS = clusterDis;

    vm.nearbyCapacity = nearbyCapacity;
    vm.revertMaxDis = revertMaxDis;
    vm.revertMinDis = revertMinDis;
    vm.eigenValDiff = eigenValDiffThresh;
    vm.z_tolerance  = z_tolerance;
}

void dynamicFilter::allocateMemory() {
    logger = spdlog::stdout_color_mt("console");

    rawSubmapDS.reset(new PointCloud());
    groundCloud.reset(new PointCloud());
    nogroundCloud.reset(new PointCloud());
    labeledNogroundCloud.reset(new PointCloud());
    dynamic_cluster_vis.reset(new PointCloud());
    static_cluster_vis.reset(new PointCloud());
    staticNogroundCloud.reset(new PointCloud());

    travel_ground_seg.reset(new travel::TravelGroundSeg<PointXYZI>());
    travel_ground_seg->setParams(max_range, min_range, resolution, 
                                num_iter, num_lpr, num_min_pts, th_seeds, 
                                th_dist, th_outlier, th_normal, th_weight, 
                                th_lcc_normal_similiarity, th_lcc_planar_model_dist, th_obstacle,
                                refine_mode, visualization_flag);
}

void dynamicFilter::groundSegment() {
    double ground_seg_time;
    travel_ground_seg->estimateGround(*rawSubmapDS, *groundCloud, *nogroundCloud, ground_seg_time);
    // logger->info("ground seg time: {}", ground_seg_time);
}

void dynamicFilter::visibilitySeg() {
    std::pair<int, int> rimgShape{rImgRow, rImgCol};
    vm.distinguishCloud(nogroundCloud, scanDSVec, poseVecIncreInv, rimgShape);
    // logger->info("ground cloud size: {}", groundCloud->size());
    // logger->info("noground cloud size: {}", nogroundCloud->size());
    // logger->info("noground cloud = dynamic cloud + static cloud");
    // logger->info("dynamic cloud size: {}", vm.dynamicCloud->size());
    // logger->info("static cloud size: {}", vm.staticCloud->size());
    // logger->info("pca reverted cloud size: {}", vm.pcaRevertedCloud->size());

    // dynamic cloud intensity remain original
    for (auto &p : vm.staticCloud->points) {
        labeledNogroundCloud->points.push_back(p);
    }
    // dynamic cloud intensity plus 200
    for (auto p : vm.dynamicCloud->points) {
        p.intensity += 200;
        labeledNogroundCloud->points.push_back(p);
    }
}

bool customCondition(const PointXYZI& seedPoint, const PointXYZI& candidatePoint, float squaredDistance) {
    float _x = abs(seedPoint.x - candidatePoint.x);
    float _y = abs(seedPoint.y - candidatePoint.y);
    float _xy = min(sqrt(_x * _x + _y * _y) * 0.01, 1.0);
    float _z = _xy * (seedPoint.z - candidatePoint.z);
    
    float _dis = sqrt(_x * _x + _y * _y + _z * _z);
    return _dis < CLUSTER_DIS; 
}

bool dynamicFilter::judgeDynamicCluster(PointCloud::Ptr cloud, pcl::PointIndices& cloudIndices) {
    // dynamic rate
    // 太近的聚类认为是自身噪点
    if(cloudIndices.indices.size() < 30 && pointDistance(cloud->points[cloudIndices.indices[0]]) < 10) {
            return true;
    }
    // 太小的聚类根据距离取舍
    if(cloudIndices.indices.size() < 10) {
        if(pointDistance(cloud->points[cloudIndices.indices[0]]) < revertMinDis) {
            return true;
        }
        else 
            return false;
    }
    // 太大的聚类默认为静态聚类
    if (cloudIndices.indices.size() > maxDynamicClusterSize) {
        return false;
    }
    vector<PointXYZI> pointVec;
    int dynamic_count = 0;
    int cluster_count = cloudIndices.indices.size();
    for (auto id : cloudIndices.indices) {
        PointXYZI p = cloud->points[id];
        if(cloud->points[id].intensity >= 200) {
            dynamic_count++;
            p.intensity -= 200;
        }
        pointVec.push_back(p);
    }
    float dynamic_rate = float(dynamic_count) / float(cluster_count);
    // 如果上一帧判断为很大概率的动态聚类和当前动态聚类的位置以及运动方向都类似，那么这个聚类很可能是动态聚类

    // cluster中的动态种子点太少，认为是静态聚类
    if(dynamic_rate < dynamicLowThresh)
        return false;
    if (dynamic_rate > dynamicHighThresh)
        return true;
    // 根据点云的时间戳排序
    sort(pointVec.begin(), pointVec.end(), [](PointXYZI p1, PointXYZI p2){return int(p1.intensity) < int(p2.intensity);});
    int num = pointVec.size();
    PointXYZI frontCentroid;
    PointXYZI midCentroid;
    PointXYZI backCentroid;
    int minFrame = pointVec[0].intensity;
    int maxFrame = pointVec.back().intensity;
    int frameNum = maxFrame - minFrame + 1;
    for (int i = 0; i < num / 3; ++i) {
        frontCentroid.x += pointVec[i].x;
        frontCentroid.y += pointVec[i].y;
        frontCentroid.z += pointVec[i].z;
    }
    frontCentroid.x /= (num / 3);
    frontCentroid.y /= (num / 3);
    frontCentroid.z /= (num / 3);

    for (int i = num / 3; i < num * 2/ 3; ++i) {
        midCentroid.x += pointVec[i].x;
        midCentroid.y += pointVec[i].y;
        midCentroid.z += pointVec[i].z;
    }
    midCentroid.x /= (num / 3);
    midCentroid.y /= (num / 3);
    midCentroid.z /= (num / 3);

    for (int i = num * 2/ 3; i < num; ++i) {
        backCentroid.x += pointVec[i].x;
        backCentroid.y += pointVec[i].y;
        backCentroid.z += pointVec[i].z;
    }
    backCentroid.x /= (num / 3);
    backCentroid.y /= (num / 3);
    backCentroid.z /= (num / 3);

    Eigen::Vector3f firstDir;
    firstDir << midCentroid.x - frontCentroid.x, midCentroid.y - frontCentroid.y, midCentroid.z - frontCentroid.z;
    Eigen::Vector3f secondDir;
    secondDir << backCentroid.x - midCentroid.x, backCentroid.y - midCentroid.y, backCentroid.z - midCentroid.z;

    // if dir is close
    float theta = firstDir.dot(secondDir) / firstDir.norm() / secondDir.norm();
    theta = std::acos(theta);


    float centroidDis = pointDistance(frontCentroid, backCentroid);
    // cout << "front centriod :" << frontCentroid.x << " " << frontCentroid.y << " " << frontCentroid.z << endl;
    // cout << "back centriod :" << backCentroid.x << " " << backCentroid.y << " " << backCentroid.z << endl;
    // cout << "centroidDis: " << centroidDis << endl;

    
    // 判断点云聚类簇是否相近
    if (!lastClusterVec.empty()) {
    }

    if(centroidDis < 0.5)
        return false;
    if(theta > 0.5)
        return false;
    // 保存上一帧的理想的聚类簇和他的运动方向
    if (num >= 30 && pointDistance(backCentroid) < 30) {
        lastClusterVec.clear();
        vector<Eigen::Vector3f> cluster_msg;
        Eigen::Vector3f cluster_pos (backCentroid.x, backCentroid.y, backCentroid.z);
        cluster_msg.push_back(cluster_pos);
        cluster_msg.push_back(secondDir);
        lastClusterVec.push_back(cluster_msg);
    }
    return true;
}

void dynamicFilter::clusterSeg() {
    pcl::search::KdTree<PointXYZI>::Ptr tree(new pcl::search::KdTree<PointXYZI>);
    tree->setInputCloud(labeledNogroundCloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::ConditionalEuclideanClustering<PointXYZI> ec;
    ec.setClusterTolerance (clusterDis);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(100000);
    // ec.setSearchMethod(tree);
    ec.setInputCloud(labeledNogroundCloud);
    ec.setConditionFunction(&customCondition);
    ec.segment(cluster_indices);
    // logger->info("cluster size: {}", cluster_indices.size());
    
    // 对每个点云簇筛选判断
    for (int i = 0; i < cluster_indices.size(); ++i) {
        bool dynamic_flag = judgeDynamicCluster(labeledNogroundCloud, cluster_indices[i]);
        if (dynamic_flag) {
            for (int id : cluster_indices[i].indices) {
                dynamic_cluster_vis->points.push_back(labeledNogroundCloud->points[id]);
            }
        }
        else {
            for (int id : cluster_indices[i].indices) {
                static_cluster_vis->points.push_back(labeledNogroundCloud->points[id]);
            }
        }
    }
    // static cluster + ground cloud = static cloud
    *staticNogroundCloud = *static_cluster_vis;
    *static_cluster_vis += *groundCloud;
    // logger->info("static_cluster_vis size: {}", static_cluster_vis->size());
    // logger->info("dynamic_cluster_vis size: {}", dynamic_cluster_vis->size());
}

void dynamicFilter::resetParameter() {
    scanDSVec.clear();
    poseVec.clear();
    poseVecIncreInv.clear();
    lidarPoses.clear();

    // cloud reset
    rawSubmapDS->clear();
    groundCloud->clear();
    nogroundCloud->clear();
    labeledNogroundCloud->clear();
    dynamic_cluster_vis->clear();
    static_cluster_vis->clear();
    staticNogroundCloud->clear();

    // vm reset
    vm.resetParameter();

} 

void dynamicFilter::onlineProcess(const vector<PointCloud::Ptr>& _cloud_vec, const vector<Eigen::Matrix4f>& _pose_vec) {
    // input downsampled cloud and lidar poses
    bool fisrtPose = true;
    Eigen::Matrix4f startPoseMatrix = Eigen::Matrix4f::Identity();

    for (int i = 0; i < _cloud_vec.size(); ++i) {
        PointCloud::Ptr curCloudDS(new PointCloud());
        PointCloud::Ptr curCloudDSTrans(new PointCloud());
        for(auto p : _cloud_vec[i]->points) {
            p.intensity += i;
            curCloudDS->push_back(p);
        }
        scanDSVec.push_back(curCloudDS);
        poseVec.push_back(_pose_vec[i]);

        if(fisrtPose) {
            fisrtPose = false;
            poseVecIncreInv.push_back(Eigen::Matrix4f::Identity());
            startPoseMatrix = _pose_vec[0];
            *rawSubmapDS += *curCloudDS;
        }
        else {
            Eigen::Matrix4f increPoseMatrix = startPoseMatrix.inverse() * _pose_vec[i];
            poseVecIncreInv.push_back(increPoseMatrix.inverse());
            pcl::transformPointCloud(*curCloudDS, *curCloudDSTrans, increPoseMatrix);
            *rawSubmapDS += *curCloudDSTrans;
        }
    }
    // logger->info("rawSubmapDS cloud size: {}", rawSubmapDS->size());
    // output lidar cloud is in the first local lidar frame
    groundSegment();
    visibilitySeg();
    clusterSeg();
}
