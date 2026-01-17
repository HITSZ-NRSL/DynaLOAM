#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>

#include <iostream>

using namespace std;

int getRandomNumber(int a, int b) {
    static const double fraction = 1.0 / (RAND_MAX + 1.0);
    return a + static_cast<int>((b - a + 1) * (std::rand() * fraction));
}

int main(int argc, char const *argv[])
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    
    for (int i = 0; i < 20; ++i) {
        pcl::PointXYZI thisPoint;
        thisPoint.x = getRandomNumber(0, 50);
        thisPoint.y = getRandomNumber(0, 50);
        thisPoint.z = getRandomNumber(-1, 1);
        thisPoint.intensity = 0;
        cloud->push_back(thisPoint);
    }

    Eigen::Matrix3d masterCovarianceMat;
    Eigen::Vector4d masterCentroidVec;
    pcl::compute3DCentroid(*cloud, masterCentroidVec);
    pcl::computeCovarianceMatrix(*cloud, masterCentroidVec, masterCovarianceMat);
    cout << masterCentroidVec << endl;
    cout << "---------------" << endl;
    cout << masterCovarianceMat << endl;
    cout << "---------------" << endl;
    Eigen::EigenSolver<Eigen::Matrix3d> masterEigenSolver(masterCovarianceMat);
    Eigen::Matrix3d masterEigenVal = masterEigenSolver.pseudoEigenvalueMatrix();
    Eigen::Matrix3d masterEigenVec = masterEigenSolver.pseudoEigenvectors();
    cout << masterEigenVal << endl;
    cout << "---------------" << endl;
    cout << masterEigenVec << endl;
    cout << "---------------" << endl;

    vector<pair<double, int>> eigenValVec;
    for (int i = 0; i < 3; ++i) {
        eigenValVec.push_back({masterEigenVal(i, i), i});
    }
    sort(eigenValVec.begin(), eigenValVec.end(), [](pair<double, int> &p1, pair<double, int> &p2){
        return p1.first < p2.first;
    });

    Eigen::Matrix3d _masterEigenVal = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d _masterEigenVec;

    for (int i = 0; i < eigenValVec.size(); ++i) {
        int id = eigenValVec[i].second;
        _masterEigenVal(i, i) = masterEigenVal(id, id);
        _masterEigenVec.block<3, 1>(0, i) = masterEigenVec.block<3, 1>(0, id);
    }
    cout << _masterEigenVal << endl;
    cout << "---------------" << endl;
    cout << _masterEigenVec << endl;
    cout << "---------------" << endl;








    return 0;
}
