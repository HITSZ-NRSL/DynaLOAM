#include "dynamic_removal/visibilityMethod.h"

namespace removert{

SphericalPoint cart2sph(const PointXYZI & _cp) {
SphericalPoint sph_point {
    std::atan2(_cp.y, _cp.x) , 
    std::atan2(_cp.z, std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y)),
    std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y + _cp.z*_cp.z)
};    
return sph_point;
}

std::set<int> convertIntVecToSet(const std::vector<int> & v) { 
    std::set<int> s; 
    for (int x : v) { 
        s.insert(x); 
    } 
    return s; 
} 

void pubRangeImg(cv::Mat& _rimg, 
                sensor_msgs::ImagePtr& _msg,
                image_transport::Publisher& _publiser,
                std::pair<float, float> _caxis) {
    cv::Mat scan_rimg_viz = convertColorMappedImg(_rimg, _caxis);
    _msg = cvmat2msg(scan_rimg_viz);
    _publiser.publish(_msg);    
} // pubRangeImg

sensor_msgs::ImagePtr cvmat2msg(const cv::Mat &_img) {
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _img).toImageMsg();
  return msg;
}

void transformPoint(const pcl::PointXYZI& pIn, pcl::PointXYZI& pOut, Eigen::Matrix4f& _trans) {
    pOut.x = _trans(0, 0) * pIn.x + _trans(0, 1) * pIn.y + _trans(0, 2) * pIn.z + _trans(0, 3);
    pOut.y = _trans(1, 0) * pIn.x + _trans(1, 1) * pIn.y + _trans(1, 2) * pIn.z + _trans(1, 3);
    pOut.z = _trans(2, 0) * pIn.x + _trans(2, 1) * pIn.y + _trans(2, 2) * pIn.z + _trans(2, 3);
    pOut.intensity = pIn.intensity;
}
}

namespace removert {
VisibilityMethod::VisibilityMethod() {
    rimg_color_min_ = 0;
    rimg_color_max_ = 50;
    kRangeColorAxis = std::pair<float, float> {rimg_color_min_, rimg_color_max_}; // meter
    kRangeColorAxisForDiff = std::pair<float, float>{0.0, 0.5}; // meter 
    logger_ = spdlog::stdout_color_mt("console1");  
    allocateMemory();

}

VisibilityMethod::~VisibilityMethod() {
}

void VisibilityMethod::allocateMemory() {
    rawCloud.reset(new PointCloud());
    dynamicCloud.reset(new PointCloud());
    staticCloud.reset(new PointCloud());
    pcaRevertedCloud.reset(new PointCloud());
    searchCloud.reset(new PointCloud());
    kdtreeRawSubmap.reset(new pcl::KdTreeFLANN<PointXYZI>());
}

void VisibilityMethod::getScanMat(cv::Mat& _scanMat) {
    _scanMat = convertColorMappedImg(this->scanMat, kRangeColorAxis);
}

void VisibilityMethod::getMapMat(cv::Mat& _matMat) {
    _matMat = convertColorMappedImg(this->mapMat, kRangeColorAxis);
}

void VisibilityMethod::getDiffMat(cv::Mat& _diffMat) {
    _diffMat = convertColorMappedImg(this->diffMat, kRangeColorAxis);
}


void VisibilityMethod::calculateDiffMat(const std::pair<int, int>& _rImgSize) {
    if(mapMat.empty() || scanMat.empty())
        return;
    this->diffMat = cv::Mat(_rImgSize.first, _rImgSize.second, CV_32FC1, cv::Scalar::all(0.0));
    for (int i = 0; i < _rImgSize.first; ++i) {
        for (int j = 0; j < _rImgSize.second; ++j) {
            if (mapMat.at<float>(i, j) != 500 && scanMat.at<float>(i, j) != 500 && mapMat.at<float>(i, j) < scanMat.at<float>(i, j) - 0.1) {
                diffMat.at<float>(i, j) = scanMat.at<float>(i, j);
                dynamicPointIdxSet.insert((int)submapIndexMat.at<float>(i, j));
            }
        }
    }    
}


void VisibilityMethod::submap2rangeMat(const PointCloud::Ptr _submap, const std::pair<int, int>& _rImgSize, Eigen::Matrix4f& _trans) {
    const int kNumRimgRow = _rImgSize.first;
    const int kNumRimgCol = _rImgSize.second;
    this->mapMat = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(500));
    this->submapIndexMat = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(-1.0));
    int num_points = _submap->size();

    #pragma omp parallel for num_threads(16)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
        PointXYZI thisPoint;
        transformPoint(_submap->points[pt_idx], thisPoint, _trans) ;
        SphericalPoint sph_point = cart2sph(thisPoint);
        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0, 30]
        // @ min and max is just for the easier (naive) boundary checks. 
        int lower_bound_row_idx {0}; 
        int lower_bound_col_idx {0};
        int upper_bound_row_idx {kNumRimgRow - 1}; 
        int upper_bound_col_idx {kNumRimgCol - 1};
        float _az = -1 * sph_point.az - M_PI_2;
        if(_az < - M_PI) {
            _az += 2 * M_PI;
        }

        // if(dynamicPointIdxsph_pointSet.find(pt_idx) != dynamicPointIdxSet.end())
        //     continue;

        int pixel_idx_row = int(std::min(std::max(std::round(kNumRimgRow * (1 - (removert::rad2deg(sph_point.el) + (50/float(2.0))) / (50 - float(0.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
        int pixel_idx_col = int(std::min(std::max(std::round(kNumRimgCol * ((removert::rad2deg(_az) + (360/float(2.0))) / (360 - float(0.0)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));
        float curr_range = sph_point.r; 
        if (curr_range <mapMat.at<float>(pixel_idx_row, pixel_idx_col) ) {
            mapMat.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
            submapIndexMat.at<float>(pixel_idx_row, pixel_idx_col) = pt_idx;
        }
    }
}

void VisibilityMethod::curScan2rangeMat(const PointCloud::Ptr _curScan, const std::pair<int, int>& _rImgSize) {
    const int kNumRimgRow = _rImgSize.first;
    const int kNumRimgCol = _rImgSize.second;
    this->scanMat = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(500));
    int num_points = _curScan->size();

    #pragma omp parallel for num_threads(16)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
        PointXYZI& this_point = _curScan->points[pt_idx];
        SphericalPoint sph_point = cart2sph(this_point);
        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0, 30]
        // @ min and max is just for the easier (naive) boundary checks. 
        int lower_bound_row_idx {0}; 
        int lower_bound_col_idx {0};
        int upper_bound_row_idx {kNumRimgRow - 1}; 
        int upper_bound_col_idx {kNumRimgCol - 1};
        float _az = -1 * sph_point.az - M_PI_2;
        if(_az < - M_PI) {
            _az += 2 * M_PI;
        }
        int pixel_idx_row = int(std::min(std::max(std::round(kNumRimgRow * (1 - (removert::rad2deg(sph_point.el) + (50/float(2.0))) / (50 - float(0.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
        int pixel_idx_col = int(std::min(std::max(std::round(kNumRimgCol * ((removert::rad2deg(_az) + (360/float(2.0))) / (360 - float(0.0)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));

        float curr_range = sph_point.r; 
        if (curr_range < scanMat.at<float>(pixel_idx_row, pixel_idx_col) ) {
            scanMat.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
        }
    }
}

void VisibilityMethod::revertStaticPoint() {
    kdtreeRawSubmap->setInputCloud(staticCloud);
    for (auto it = dynamicPointIdxSet.begin(); it != dynamicPointIdxSet.end(); ++it) {
        int thisDPointID = *it;
        PointXYZI thisDPoint = rawCloud->points[thisDPointID];
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        // 如果动态点云太高，则认为是误杀点
        if(thisDPoint.z > z_tolerance) {
            pcaRevertedCloud->push_back(thisDPoint);
            revertedPointIdxSet.insert(thisDPointID);
            continue;
        }
        // 如果动态点云太远，认为是误杀点
        if(pointDistance(thisDPoint) > revertMaxDis) {
            pcaRevertedCloud->push_back(thisDPoint);
            revertedPointIdxSet.insert(thisDPointID);
            continue;
        }

        // 如果点云过近，不恢复
        if(pointDistance(thisDPoint) < revertMinDis) {
            continue;
        }

        kdtreeRawSubmap->nearestKSearch(thisDPoint, nearbyCapacity, pointSearchInd, pointSearchSqDis);
        
        // 如果动态点云找不到最近邻的几个点，跳过
        if(pointSearchSqDis[nearbyCapacity - 1] > nearDisThresh)
            continue;
        
        Eigen::Matrix3d ori_masterEigenVal;
        Eigen::Matrix3d ori_masterEigenVec;
        Eigen::Matrix3d new_masterEigenVal;
        Eigen::Matrix3d new_masterEigenVec;

        // 计算周围静态点云的法向量
        PointCloud::Ptr nearbyCloud(new PointCloud());
        for (auto i : pointSearchInd) { 
            nearbyCloud->push_back(staticCloud->points[i]);
        }

        computePCA(nearbyCloud, ori_masterEigenVal, ori_masterEigenVec);

        float L = (ori_masterEigenVal(0, 0) - ori_masterEigenVal(1, 1)) / ori_masterEigenVal(0, 0);
        float P = (ori_masterEigenVal(1, 1) - ori_masterEigenVal(2, 2)) / ori_masterEigenVal(0, 0);
        float S = ori_masterEigenVal(2, 2) / ori_masterEigenVal(0, 0);

        Eigen::Vector3f ori_feature(L, P, S);
        ori_feature = ori_feature.normalized();

        // judge by eigen value
        if(ori_masterEigenVal(2, 2) * 10 > ori_masterEigenVal(0, 0))
            continue;

        // 计算加入动态点云后点云的法向量,从大到小
        nearbyCloud->push_back(thisDPoint);
        computePCA(nearbyCloud, new_masterEigenVal, new_masterEigenVec);
        
        L = (new_masterEigenVal(0, 0) - new_masterEigenVal(1, 1)) / new_masterEigenVal(0, 0);
        P = (new_masterEigenVal(1, 1) - new_masterEigenVal(2, 2)) / new_masterEigenVal(0, 0);
        S = new_masterEigenVal(2, 2) / new_masterEigenVal(0, 0);

        Eigen::Vector3f new_feature(L, P, S);
        new_feature = new_feature.normalized();

        // 判断特征值的差距
        Eigen::Matrix3d& A = new_masterEigenVal;
        Eigen::Matrix3d& B = ori_masterEigenVal;
        double dis = pow(A(0, 0) - B(0, 0), 2) + pow(A(1, 1) - B(1, 1), 2) + pow(A(2, 2) - B(2, 2), 2);
        auto dis_feature = ori_feature.cross(new_feature);
        float distance = dis_feature.norm();

        if (distance < eigenValDiff) {
            pcaRevertedCloud->push_back(thisDPoint);
            revertedPointIdxSet.insert(thisDPointID);
        }
    }
}

void VisibilityMethod::computePCA(PointCloud::Ptr _cloudIn, Eigen::Matrix3d& _masterEigenVal, Eigen::Matrix3d& _masterEigenVec) {
    Eigen::Matrix3d masterCovarianceMat;
    Eigen::Vector4d masterCentroidVec;
    pcl::compute3DCentroid(*_cloudIn, masterCentroidVec);
    pcl::computeCovarianceMatrix(*_cloudIn, masterCentroidVec, masterCovarianceMat);
    Eigen::EigenSolver<Eigen::Matrix3d> masterEigenSolver(masterCovarianceMat);
    Eigen::Matrix3d masterEigenVal = masterEigenSolver.pseudoEigenvalueMatrix();
    Eigen::Matrix3d masterEigenVec = masterEigenSolver.pseudoEigenvectors();
    vector<pair<double, int>> eigenValVec;
    for (int i = 0; i < 3; ++i) {
        eigenValVec.push_back({masterEigenVal(i, i), i});
    }
    sort(eigenValVec.begin(), eigenValVec.end(), [](pair<double, int> p1, pair<double, int> p2){
        return p1.first > p2.first;
    });

    _masterEigenVal = Eigen::Matrix3d::Identity();
    for (int i = 0; i < eigenValVec.size(); ++i) {
        int id = eigenValVec[i].second;
        if(_masterEigenVal(i, i) > 0) {
            _masterEigenVal(i, i) = masterEigenVal(id, id);
            _masterEigenVec.block<3, 1>(0, i) = masterEigenVec.block<3, 1>(0, id);
        }
        else {
            _masterEigenVal(i, i) = -1 * masterEigenVal(id, id);
            _masterEigenVec.block<3, 1>(0, i) = -1 * masterEigenVec.block<3, 1>(0, id);
        }
        
    }



}

void VisibilityMethod::distinguishCloud(PointCloud::Ptr _submapCloud, vector<PointCloud::Ptr> _localCloudVec, 
    vector<Eigen::Matrix4f> _localTransVec, std::pair<int, int>& _rImgShape) {
        
    // 将submap投影成rangeMat
    // 保存submap点云的索引为mat
    // 将localScan投影成rangeMat
    // 将diffmat对应的点云索引保存进一个Vec
    // 待循环结束后，将submap根据索引区分为static和dynamic
    assert(_localCloudVec.size() == _localTransVec.size());
    rawCloud = _submapCloud;
    // most cost time part
    for (int i = 0; i < _localCloudVec.size(); ++i) {
        submap2rangeMat(_submapCloud, _rImgShape, _localTransVec[i]);
        curScan2rangeMat(_localCloudVec[i], _rImgShape);
        calculateDiffMat(_rImgShape);
    }
    for (int i = 0; i < _submapCloud->size(); ++i) {
        if(dynamicPointIdxSet.find(i) != dynamicPointIdxSet.end()) {
            dynamicCloud->points.push_back(_submapCloud->points[i]);
        }
        else {
            staticCloud->points.push_back(_submapCloud->points[i]);
        }
    }

    // 进一步revert误删的动态点云
    // 原始点云构建kd-tree，寻找当前动态点云周围的10个原始点云，如果10个原始点云都离得比较近，并且添加动态点云和删除动态点云后的法向量几乎不变，那就恢复
    // 计算动态点周围是否能形成一个小平面，判断点云的入射角是否过大，如果过大，那可能是误杀静态点。
    revertStaticPoint();
    // 增加动态点，将动态点云周围一小部分的点都当做动态点生长。

    // 划分动态点云和静态点云
    dynamicCloud->clear();
    for (auto it = dynamicPointIdxSet.begin(); it != dynamicPointIdxSet.end(); ++it) {
        if(revertedPointIdxSet.find(*it) == revertedPointIdxSet.end()) {
            dynamicCloud->push_back(_submapCloud->points[*it]);
        }
        else {
            staticCloud->push_back(_submapCloud->points[*it]);
        }
    }
    
}

void VisibilityMethod::resetParameter() {
    rawCloud->clear();
    dynamicCloud->clear();
    staticCloud->clear();
    pcaRevertedCloud->clear();
    searchCloud->clear();

    dynamicPointIdxSet.clear();
    revertedPointIdxSet.clear();
}
}