#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <unordered_map>
#include <execution>

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

#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/LinearMath/Quaternion.h> // to Quaternion_to_euler
#include <tf/LinearMath/Matrix3x3.h> // to Quaternion_to_euler
#include <tf/transform_datatypes.h> // createQuaternionFromRPY
#include <tf_conversions/tf_eigen.h> // tf <-> eigen
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>


using namespace std;

using namespace gtsam;

using std::cout;
using std::endl;

template <typename T>
sensor_msgs::PointCloud2 pcl_to_pcl_ros(pcl::PointCloud<T> cloud, string frame_id="map")
{
  sensor_msgs::PointCloud2 cloud_ROS;
  pcl::toROSMsg(cloud, cloud_ROS);
  cloud_ROS.header.frame_id = frame_id;
  return cloud_ROS;
}

///// transformation
template <typename T>
pcl::PointCloud<T> tf_pcd(const pcl::PointCloud<T> &cloud_in, const Eigen::Matrix4d &pose_tf)
{
	if (cloud_in.size() == 0) return cloud_in;
	pcl::PointCloud<T> pcl_out_ = cloud_in;
	std::for_each(pcl_out_.begin(), pcl_out_.end(), [&](T &pt)
	{
		float x_ = pt.x;
		float y_ = pt.y;
		float z_ = pt.z;
		pt.x = pose_tf(0, 0) * x_ + pose_tf(0, 1) * y_ + pose_tf(0, 2) * z_ + pose_tf(0, 3);
		pt.y = pose_tf(1, 0) * x_ + pose_tf(1, 1) * y_ + pose_tf(1, 2) * z_ + pose_tf(1, 3);
		pt.z = pose_tf(2, 0) * x_ + pose_tf(2, 1) * y_ + pose_tf(2, 2) * z_ + pose_tf(2, 3);
	});
  return pcl_out_;
}

struct pose_pcd
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcd;
    Eigen::Matrix4d pose_eig = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d pose_corrected_eig = Eigen::Matrix4d::Identity();
    double timestamp;
    int idx;
    bool processed = false;
    pose_pcd(){
        pcd.reset(new pcl::PointCloud<pcl::PointXYZI>());
    };
    pose_pcd(const nav_msgs::Odometry &odom_in, const sensor_msgs::PointCloud2 &pcd_in, const int &idx_in) {
        pcd.reset(new pcl::PointCloud<pcl::PointXYZI>());
        tf::Quaternion q_(odom_in.pose.pose.orientation.x, odom_in.pose.pose.orientation.y, odom_in.pose.pose.orientation.z, odom_in.pose.pose.orientation.w);
        tf::Matrix3x3 m_(q_);
        Eigen::Matrix3d tmp_rot_mat_;
        tf::matrixTFToEigen(m_, tmp_rot_mat_);
        pose_eig.block<3, 3>(0, 0) = tmp_rot_mat_;
        pose_eig(0, 3) = odom_in.pose.pose.position.x;
        pose_eig(1, 3) = odom_in.pose.pose.position.y;
        pose_eig(2, 3) = odom_in.pose.pose.position.z;
        pose_corrected_eig = pose_eig;
        pcl::fromROSMsg(pcd_in, *pcd);
        // pcd = tf_pcd(tmp_pcd_, pose_eig.inverse()); //FAST-LIO publish data in world frame, so save it in LiDAR frame
        timestamp = odom_in.header.stamp.toSec();
        idx = idx_in;
    }
};

pose_pcd currentFrame;
sensor_msgs::NavSatFix::ConstPtr currGPS;

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;

std::mutex odomMutex;
std::mutex scanMutex;
std::mutex realtime_pose_mutex;
std::mutex keyframes_mutex;
std::mutex graph_mutex;
std::mutex lc_mutex;

Eigen::Matrix4d odom_delta(Eigen::Matrix4d::Identity());
Eigen::Matrix4d m_last_corrected_pose(Eigen::Matrix4d::Identity());

std::vector<pose_pcd> keyFrames;
int current_keyframe_idx = 0;
nav_msgs::Path pathOri;

// for loop closure detection
std::map<int, int> loopIndexContainer; // 记录存在的回环对
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<pcl::PointXYZ>());
double historyKeyframeSearchRadius = 20;
double historyKeyframeSearchTimeDiff = 30;
int historyKeyframeSearchNum = 25;

// sc loop manager
SCManager scManager;
double scDistThres, scMaximumRadius;

// gtsam
bool gtSAMgraphMade = false;
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values correctEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

// pcl
pcl::PointCloud<PointType_>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType_>());
pcl::PointCloud<PointType_>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType_>());
pcl::PointCloud<pcl::PointXYZ>::Ptr odomCloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::VoxelGrid<PointType_> downSizeFilterScancontext;
pcl::VoxelGrid<PointType_> downSizeFilterICP;
pcl::PointCloud<PointType_>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType_>());
pcl::VoxelGrid<PointType_> downSizeFilterMapPGO;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.
bool loopAddedFlag = false;
bool laserCloudMapPGORedraw = true;
bool useGPS = true;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory, pgSCDsDirectory;
std::string odomKITTIformat;
std::fstream pgG2oSaveStream, pgTimeSaveStream;
std::vector<std::string> edges_str; // used in writeEdge

// ros
ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO, pubPathOri;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubLoopConstraintEdge;
ros::Publisher pubOriginalOdom;
ros::Publisher pubCorrectedOdom;
ros::Publisher pubCorrectedCurrentScan;

void voxelize_pcd(pcl::VoxelGrid<pcl::PointXYZI> &voxelgrid, pcl::PointCloud<pcl::PointXYZI> &pcd_in)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr before_(new pcl::PointCloud<pcl::PointXYZI>);
  *before_ = pcd_in;
  voxelgrid.setInputCloud(before_);
  voxelgrid.filter(pcd_in);
  return;
}

Eigen::Matrix4d Pose6DtoMatrix4d(const Pose6D& pose) {
    Eigen::Affine3d _m;
    pcl::getTransformation(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw, _m);
    return _m.matrix();
}

Pose6D Matrix4dtoPose6D(const Eigen::Matrix4d& mat) {
    Pose6D p;
    Eigen::Vector3d rpy = mat.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    p.x = mat(0, 3);
    p.y = mat(1, 3);
    p.z = mat(2, 3);
    p.roll = rpy(0);
    p.pitch = rpy(1);
    p.yaw = rpy(2);
    return p;
}

gtsam::Pose3 Matrix4dtoGTSAMPose3(const Eigen::Matrix4d& mat) {
double r_, p_, y_;
  tf::Matrix3x3 mat_;
  tf::matrixEigenToTF(mat.block<3, 3>(0, 0), mat_);
  mat_.getRPY(r_, p_, y_);
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(r_, p_, y_), gtsam::Point3(mat(0, 3), mat(1, 3), mat(2, 3)));
}

Eigen::Matrix4d GTSAMPose3toMatrix4d(const gtsam::Pose3& _p) {
    Eigen::Matrix4d pose_eig_out_ = Eigen::Matrix4d::Identity();
	tf::Quaternion quat_ = tf::createQuaternionFromRPY(_p.rotation().roll(), _p.rotation().pitch(), _p.rotation().yaw());
    tf::Matrix3x3 mat_(quat_);
	Eigen::Matrix3d tmp_rot_mat_;
    tf::matrixTFToEigen(mat_, tmp_rot_mat_);
    pose_eig_out_.block<3, 3>(0, 0) = tmp_rot_mat_;
    pose_eig_out_(0, 3) = _p.translation().x();
    pose_eig_out_(1, 3) = _p.translation().y();
    pose_eig_out_(2, 3) = _p.translation().z();
    return pose_eig_out_;
}

geometry_msgs::PoseStamped gtsam_pose_to_pose_stamped(const gtsam::Pose3 &gtsam_pose_in, string frame_id="map")
{
	tf::Quaternion quat_ = tf::createQuaternionFromRPY(gtsam_pose_in.rotation().roll(), gtsam_pose_in.rotation().pitch(), gtsam_pose_in.rotation().yaw());
	geometry_msgs::PoseStamped pose_;
	pose_.header.frame_id = frame_id;
	pose_.pose.position.x = gtsam_pose_in.translation().x();
	pose_.pose.position.y = gtsam_pose_in.translation().y();
	pose_.pose.position.z = gtsam_pose_in.translation().z();
	pose_.pose.orientation.w = quat_.getW();
	pose_.pose.orientation.x = quat_.getX();
	pose_.pose.orientation.y = quat_.getY();
	pose_.pose.orientation.z = quat_.getZ();
	return pose_;
}

geometry_msgs::PoseStamped pose_eig_to_pose_stamped(const Eigen::Matrix4d &pose_eig_in, string frame_id="map")
{
	double r_, p_, y_;
  tf::Matrix3x3 mat_;
  tf::matrixEigenToTF(pose_eig_in.block<3, 3>(0, 0), mat_);
  mat_.getRPY(r_, p_, y_);
  tf::Quaternion quat_ = tf::createQuaternionFromRPY(r_, p_, y_);
	geometry_msgs::PoseStamped pose_;
	pose_.header.frame_id = frame_id;
	pose_.pose.position.x = pose_eig_in(0, 3);
	pose_.pose.position.y = pose_eig_in(1, 3);
	pose_.pose.position.z = pose_eig_in(2, 3);
	pose_.pose.orientation.w = quat_.getW();
	pose_.pose.orientation.x = quat_.getX();
	pose_.pose.orientation.y = quat_.getY();
	pose_.pose.orientation.z = quat_.getZ();
	return pose_;
}

std::string padZeros(int val, int num_digits = 6) 
{
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

std::string getVertexStr(const int _node_idx, const gtsam::Pose3& _Pose)
{
    gtsam::Point3 t = _Pose.translation();
    gtsam::Rot3 R = _Pose.rotation();

    std::string curVertexInfo {
        "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgVertexSaveStream << curVertexInfo << std::endl;
    // vertices_str.emplace_back(curVertexInfo);
    return curVertexInfo;
}

void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose, std::vector<std::string>& edges_str)
{
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgEdgeSaveStream << curEdgeInfo << std::endl;
    edges_str.emplace_back(curEdgeInfo);
}

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");
 
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

// void saveGTSAMgraphG2oFormat(const gtsam::Values& _estimates)
// {
//     // save pose graph (runs when programe is closing)
//     // cout << "****************************************************" << endl; 
//     cout << "Saving the posegraph ..." << endl; // giseop

//     pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);

//     int pose_idx = 0;
//     for(const auto& _pose6d: keyframePoses) {
//         gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);    
//         pgG2oSaveStream << getVertexStr(pose_idx, pose) << endl;
//         pose_idx++;
//     }
//     for(auto& _line: edges_str)
//         pgG2oSaveStream << _line << std::endl;

//     pgG2oSaveStream.close();
// }

// void saveOdometryVerticesKITTIformat(std::string _filename)
// {
//     // ref from gtsam's original code "dataset.cpp"
//     std::fstream stream(_filename.c_str(), std::fstream::out);
//     for(const auto& _pose6d: keyframePoses) {
//         gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
//         Point3 t = pose.translation();
//         Rot3 R = pose.rotation();
//         auto col1 = R.column(1); // Point3
//         auto col2 = R.column(2); // Point3
//         auto col3 = R.column(3); // Point3

//         stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
//                << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
//                << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
//     }
// }

// void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
// {
//     using namespace gtsam;

//     // ref from gtsam's original code "dataset.cpp"
//     std::fstream stream(_filename.c_str(), std::fstream::out);

//     for(const auto& key_value: _estimates) {
//         auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
//         if (!p) continue;

//         const Pose3& pose = p->value();

//         Point3 t = pose.translation();
//         Rot3 R = pose.rotation();
//         auto col1 = R.column(1); // Point3
//         auto col2 = R.column(2); // Point3
//         auto col3 = R.column(3); // Point3

//         stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
//                << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
//                << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
//     }
// }

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	lock_guard<mutex> lock(odomMutex);
	odometryBuf.push(_laserOdometry);
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
	lock_guard<mutex> lock(scanMutex);
	fullResBuf.push(_laserCloudFullRes);
	
} // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
    if(useGPS) {
        gpsBuf.push(_gps);
    }
} // gpsHandler

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2;
    // odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

Pose6D diffTransformation(const Eigen::Matrix4d& _p1, const Eigen::Matrix4d& _p2)
{
    
    Eigen::Matrix4d SE3_delta0 = _p1.inverse() * _p2;
    Eigen::Affine3d SE3_delta; 
    SE3_delta.matrix() = SE3_delta0;
    double dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} 

bool check_if_keyframe(const pose_pcd &pose_pcd_in, const pose_pcd &latest_pose_pcd)
{
  Pose6D dtf = diffTransformation(pose_pcd_in.pose_corrected_eig, latest_pose_pcd.pose_corrected_eig);
  double delta_translation = sqrt(dtf.x * dtf.x + dtf.y * dtf.y + dtf.z * dtf.z);
  double delta_rotation = sqrt(dtf.roll * dtf.roll + dtf.pitch * dtf.pitch + dtf.yaw * dtf.yaw);
  return delta_translation > 2 || delta_rotation > M_PI / 3.0;
}

pcl::PointCloud<PointType_>::Ptr local2global(const pcl::PointCloud<PointType_>::Ptr &cloudIn, const Eigen::Matrix4d& transCur)
{
    pcl::PointCloud<PointType_>::Ptr cloudOut(new pcl::PointCloud<PointType_>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

void pubPath( void )
{
    // pub odom and path 
    gtsam::Values corrected_esti_copy_;
    {
      lock_guard<mutex> lock(realtime_pose_mutex);
      corrected_esti_copy_ = correctEstimate;
    }
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pcl::PointCloud<pcl::PointXYZ> corrected_odoms_;

    pathAftPGO.header.frame_id = "camera_init";
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < corrected_esti_copy_.size(); node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        gtsam::Pose3 pose_ = corrected_esti_copy_.at<gtsam::Pose3>(node_idx);
        geometry_msgs::PoseStamped poseStampAftPGO = gtsam_pose_to_pose_stamped(pose_, "camera_init");
        poseStampAftPGO.header.stamp = ros::Time().fromSec(keyFrames[node_idx].timestamp);
        odomAftPGO.pose.pose = poseStampAftPGO.pose;
        pathAftPGO.poses.push_back(poseStampAftPGO);
        corrected_odoms_.points.emplace_back(pose_.translation().x(), pose_.translation().y(), pose_.translation().z());
    }
    pathAftPGO.header.frame_id = "camera_init";
    odomAftPGO.header.frame_id = "camera_init";
    pathOri.header.frame_id = "camera_init";

    pubOdomAftPGO.publish(odomAftPGO); // last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses
    pubPathOri.publish(pathOri);
    pubCorrectedOdom.publish(pcl_to_pcl_ros(corrected_odoms_, "camera_init")); 
    pubOriginalOdom.publish(pcl_to_pcl_ros(*odomCloud, "camera_init"));
    

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    // br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "aft_pgo"));
} // pubPath

void updatePoses(void)
{
    for (int node_idx=0; node_idx < int(correctEstimate.size()); node_idx++)
    {
        keyFrames[node_idx].pose_corrected_eig = GTSAMPose3toMatrix4d(correctEstimate.at<gtsam::Pose3>(node_idx));
    }
} // updatePoses

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType_>::Ptr& nearKeyframes, const int& key, const int& submap_size)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i; // see https://github.com/gisbi-kim/SC-A-LOAM/issues/7 ack. @QiMingZhenFan found the error and modified as below. 
        if (keyNear < 0 || keyNear >= int(keyFrames.size()) )
            continue;
        *nearKeyframes += * local2global((keyFrames[keyNear].pcd), keyFrames[keyNear].pose_corrected_eig);
        
    }
    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType_>::Ptr cloud_temp(new pcl::PointCloud<PointType_>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud

std::optional<gtsam::Pose3> doICPVirtualRelative( int _curr_kf_idx, int _loop_kf_idx, double &score)
{
    pcl::PointCloud<PointType_>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType_>());
    pcl::PointCloud<PointType_>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType_>());
    Eigen::Matrix4d poseFromMat;
    Eigen::Matrix4d poseToMat;
    // parse pointclouds
    {
        lock_guard<mutex> lock(keyframes_mutex);
        poseFromMat = keyFrames[_curr_kf_idx].pose_corrected_eig;
        poseToMat   =  keyFrames[_loop_kf_idx].pose_corrected_eig;
        int historyKeyframeSearchNum = 5; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
        loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0); // use same root of loop kf idx 
        loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum); 
    }
    sensor_msgs::PointCloud2 loopScanMsg;
    pcl::toROSMsg(*cureKeyframeCloud, loopScanMsg);
    loopScanMsg.header.frame_id = "camera_init";
    pubLoopScanLocal.publish(loopScanMsg);
        
    sensor_msgs::PointCloud2 loopMapMsg;
    pcl::toROSMsg(*targetKeyframeCloud, loopMapMsg);
    loopMapMsg.header.frame_id = "camera_init";
    pubLoopSubmapLocal.publish(loopMapMsg);

    // ICP Settings
    pcl::IterativeClosestPoint<PointType_, PointType_> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType_>::Ptr unused_result(new pcl::PointCloud<PointType_>());
    icp.align(*unused_result);
 
    float loopFitnessScoreThreshold = 1; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        return std::nullopt;
    } else {
        std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
    }

    // Get pose transformation
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    Eigen::Matrix4d _matFrom = correctionLidarFrame.matrix().cast<double>() * poseFromMat;
    Eigen::Matrix4d _matTo = poseToMat;

    gtsam::Pose3 poseFrom = Matrix4dtoGTSAMPose3(_matFrom);
    gtsam::Pose3 poseTo = Matrix4dtoGTSAMPose3(_matTo);
    score = icp.getFitnessScore();
    return poseFrom.between(poseTo);
} // doICPVirtualRelative

void process_pg()
{
    while(1)
    {
		while ( !odometryBuf.empty() && !fullResBuf.empty() )
        {
            //
            // pop and check keyframe is or not  
            // 
			lock_guard<mutex> lock1(odomMutex);
            lock_guard<mutex> lock2(scanMutex);    
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec()) {
                odometryBuf.pop();
            }
                
            if (odometryBuf.empty())
            {
                ROS_DEBUG("waittng for odom data ...");
                break;
            }

            Eigen::Matrix4d last_odom_tf_;
            last_odom_tf_ = currentFrame.pose_eig;
            currentFrame = pose_pcd(*odometryBuf.front(), *fullResBuf.front(), current_keyframe_idx);
            fullResBuf.pop();

            if (!gtSAMgraphMade) {
                keyFrames.push_back(currentFrame);
                // graph
                gtsam::noiseModel::Diagonal::shared_ptr prior_noise_ = 
                gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished()); // rad*rad, meter*meter
                gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, Matrix4dtoGTSAMPose3(currentFrame.pose_eig), prior_noise_));
                initialEstimate.insert(current_keyframe_idx, Matrix4dtoGTSAMPose3(currentFrame.pose_eig)); 
                {
                lock_guard<mutex> lock(realtime_pose_mutex);
                odom_delta = odom_delta * last_odom_tf_.inverse() * currentFrame.pose_eig;
                currentFrame.pose_corrected_eig = m_last_corrected_pose * odom_delta;
                // m_realtime_pose_pub.publish(pose_eig_to_pose_stamped(m_current_frame.pose_corrected_eig, m_map_frame));
                }
                odomCloud->points.emplace_back(currentFrame.pose_eig(0, 3), currentFrame.pose_eig(1, 3), currentFrame.pose_eig(2, 3));
                pathOri.poses.push_back(pose_eig_to_pose_stamped(currentFrame.pose_eig, "camera_init"));
 
                pcl::PointCloud<PointType_>::Ptr correctedScan(new pcl::PointCloud<PointType_>());
                *correctedScan = tf_pcd(*currentFrame.pcd, currentFrame.pose_corrected_eig);
                pubCorrectedCurrentScan.publish(pcl_to_pcl_ros(*correctedScan, "camera_init"));
                current_keyframe_idx++;
                gtSAMgraphMade = true;

            } else {
                {
                lock_guard<mutex> lock(realtime_pose_mutex);
                odom_delta = odom_delta * last_odom_tf_.inverse() * currentFrame.pose_eig;
                currentFrame.pose_corrected_eig = m_last_corrected_pose * odom_delta;
                }
                // 2. cjeck if keyframe
                if (check_if_keyframe(currentFrame, keyFrames.back())) {
                    // 2-2 if sp, save keyframe
                    {
                        keyFrames.push_back(currentFrame);
                        odomCloud->points.emplace_back(currentFrame.pose_eig(0, 3), currentFrame.pose_eig(1, 3), currentFrame.pose_eig(2, 3));
                        pathOri.poses.push_back(pose_eig_to_pose_stamped(currentFrame.pose_eig, "camera_init"));
                    }
                    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_ = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished());
                    gtsam::Pose3 pose_from_ = Matrix4dtoGTSAMPose3(keyFrames[current_keyframe_idx-1].pose_corrected_eig);
                    gtsam::Pose3 pose_to_ = Matrix4dtoGTSAMPose3(currentFrame.pose_corrected_eig);
                    {
                        lock_guard<mutex> lock(graph_mutex);
                        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(current_keyframe_idx-1, current_keyframe_idx, pose_from_.between(pose_to_), odom_noise_));
                        initialEstimate.insert(current_keyframe_idx, pose_to_);
                    }
                    current_keyframe_idx++;

                    // 4. optimize
                    {
                        lock_guard<mutex> lock(graph_mutex);
                        isam->update(gtSAMgraph, initialEstimate);
                        isam->update();
                        if (loopAddedFlag) {
                            isam->update();
                            isam->update();
                            isam->update();
                        }
                        gtSAMgraph.resize(0);
                        initialEstimate.clear();
                    }

                    // 5. correct pose
                    {
                        lock_guard<mutex> lock(realtime_pose_mutex);
                        correctEstimate = isam->calculateEstimate();
                        m_last_corrected_pose = 
                        GTSAMPose3toMatrix4d(correctEstimate.at<gtsam::Pose3>(correctEstimate.size() - 1));
                        odom_delta = Eigen::Matrix4d::Identity();
                    }

                    if (loopAddedFlag) {
                        {
                            lock_guard<mutex> lock(keyframes_mutex);
                            updatePoses();
                        }
                        loopAddedFlag = false;
                    }
                    pcl::PointCloud<PointType_>::Ptr correctedScan(new pcl::PointCloud<PointType_>());
                    *correctedScan = tf_pcd(*currentFrame.pcd, currentFrame.pose_corrected_eig);
                    pubCorrectedCurrentScan.publish(pcl_to_pcl_ros(*correctedScan, "camera_init"));
                }
            }
        }

        // ps. 
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void performSCLoopClosure(void)
{
    if( int(keyFrames.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyFrames.size() - 1; // because cpp starts 0 and ends n-1
        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;
        scLoopICPBuf.push(std::pair<int, int>(curr_node_idx, prev_node_idx));
        // addding actual 6D constraints in the other thread, icp_calculation.
    } 
} // performSCLoopClosure

pcl::PointCloud<pcl::PointXYZ>::Ptr keyFrames2pc(const std::vector<pose_pcd>& _keyFrames){
    pcl::PointCloud<pcl::PointXYZ>::Ptr res( new pcl::PointCloud<pcl::PointXYZ> ) ;
    for( auto _frame : _keyFrames){
        pcl::PointXYZ _p;
        _p.x = _frame.pose_corrected_eig(0, 3);
        _p.y = _frame.pose_corrected_eig(1, 3);
        _p.z = _frame.pose_corrected_eig(2, 3);
        res->points.emplace_back(_p);
    }
    return res;
}

/**
 * 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
*/
bool detectLoopClosureDistance(int& loopKeyCur, int& loopKeyPre)
{
    loopKeyPre = -1;
    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<pcl::PointXYZ>());
    copy_cloudKeyPoses3D = keyFrames2pc(keyFrames);
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    
//     // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for(int i = 0; i < pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if ( abs( keyFrames[id].timestamp - keyFrames[loopKeyCur].timestamp ) > historyKeyframeSearchTimeDiff )
        {
            loopKeyPre = id;
            break;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    return true;
}

void performRSLoopClosure(void)
{
    if(!gtSAMgraphMade) // 如果历史关键帧为空
        return;

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur = current_keyframe_idx;
    int loopKeyPre = -1;
    bool _success = false;
    {
        lock_guard<mutex> lock(keyframes_mutex);
        loopKeyCur = keyFrames.size() - 1;
        loopKeyPre = -1;
        _success = detectLoopClosureDistance(loopKeyCur, loopKeyPre);
    }
    if (_success) {
        lock_guard<mutex> lock(lc_mutex);
        cout << "Loop detected! - between " << loopKeyPre << " and " << loopKeyCur << "" << endl;
        scLoopICPBuf.push(std::pair<int, int>(loopKeyCur, loopKeyPre));
        loopIndexContainer[loopKeyCur] = loopKeyPre ;
    } else 
        return;
} // performRSLoopClosure

/**
 * rviz展示闭环边
*/
void visualizeLoopClosure()
{
    if (loopIndexContainer.empty())
        return;
    gtsam::Values corrected_esti_copy_;
    {
      lock_guard<mutex> lock(realtime_pose_mutex);
      corrected_esti_copy_ = correctEstimate;
    }
    
    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "camera_init"; // camera_init
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "camera_init";
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    {
        lock_guard<mutex> lock(lc_mutex);
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            if (key_cur >= corrected_esti_copy_.size() || key_pre >= corrected_esti_copy_.size())
                continue;
            geometry_msgs::Point p;
            gtsam::Pose3 p1 = corrected_esti_copy_.at<gtsam::Pose3>(key_cur);
            p.x = p1.translation().x();
            p.y = p1.translation().y();
            p.z = p1.translation().z();
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);

            gtsam::Pose3 p2 = corrected_esti_copy_.at<gtsam::Pose3>(key_pre);
            p.x = p2.translation().x();
            p.y = p2.translation().y();
            p.z = p2.translation().z();
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

void process_lcd(void)
{
    float loopClosureFrequency = 2.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        // performSCLoopClosure();
        performRSLoopClosure(); // TODO
        visualizeLoopClosure();
    }
} // process_lcd

void process_icp(void)
{
    while(1)
    {
		while ( !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 ) {
                ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            
            std::pair<int, int> loop_idx_pair;
            {
                lock_guard<mutex> lock(lc_mutex);
                loop_idx_pair = scLoopICPBuf.front();
                scLoopICPBuf.pop();
            }
            
            const int curr_node_idx = loop_idx_pair.first;
            const int prev_node_idx = loop_idx_pair.second;
            double score_;

            auto relative_pose_optional = doICPVirtualRelative(curr_node_idx, prev_node_idx, score_);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                lock_guard<mutex> lock(graph_mutex);
                gtsam::noiseModel::Diagonal::shared_ptr loop_noise_ = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << score_, score_, score_, score_, score_, score_).finished());
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(curr_node_idx, prev_node_idx, relative_pose, loop_noise_));
                // // writeEdge({prev_node_idx, curr_node_idx}, relative_pose, edges_str); // giseop
                loopAddedFlag = true;
            } 
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_icp

void process_viz_path(void)
{
    float hz = 10.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if(gtSAMgraphMade) {
            pubPath();
        }
    }
}

void pubMap(void)
{
    if (pubMapAftPGO.getNumSubscribers() == 0)
        return;
    pcl::PointCloud<pcl::PointXYZI> corrected_map_;
    {
        lock_guard<mutex> lock(keyframes_mutex);
        for (int i = 0; i < keyFrames.size(); i+=5) {
            corrected_map_ += tf_pcd(*keyFrames[i].pcd, keyFrames[i].pose_corrected_eig);
        }
    }

    voxelize_pcd(downSizeFilterMapPGO, corrected_map_);
    
    pubMapAftPGO.publish(pcl_to_pcl_ros(corrected_map_, "camera_init"));
}

void process_viz_map(void)
{
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(gtSAMgraphMade) {
            pubMap();
        }
    }
} // pointcloud_viz

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

    // save directories 
	nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move 

    pgKITTIformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";

    // pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);

    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);

    // pgScansDirectory = save_directory + "Scans/";
    // auto unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    // unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());

    // pgSCDsDirectory = save_directory + "SCDs/"; // SCD: scan context descriptor 
    // unused = system((std::string("exec rm -r ") + pgSCDsDirectory).c_str());
    // unused = system((std::string("mkdir -p ") + pgSCDsDirectory).c_str());

    // system params 
	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.4; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

    double mapVizFilterSize;
	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_body", 100, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry     = nh.subscribe<nav_msgs::Odometry>("Odometry", 100, laserOdometryHandler);
	ros::Subscriber subGPS               = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);
	pubOdomAftPGO                        = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubPathAftPGO                        = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
    pubPathOri                           = nh.advertise<nav_msgs::Path>("original_path", 100);
	pubMapAftPGO                         = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);
	pubLoopScanLocal                     = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal                   = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);
    pubOriginalOdom                      = nh.advertise<sensor_msgs::PointCloud2>("/original_odom", 100);
    pubCorrectedOdom                     = nh.advertise<sensor_msgs::PointCloud2>("/corrected_odom", 100);
    pubCorrectedCurrentScan              = nh.advertise<sensor_msgs::PointCloud2>("/corrected_current_scan", 100);
    pubLoopConstraintEdge                = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1);

	std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 

	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	return 0;
}
