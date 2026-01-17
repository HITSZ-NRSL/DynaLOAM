#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <string>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>

using namespace std;
using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointXYZI>;

vector<int> DYNAMIC_CLASSES = {252, 253, 254, 255, 256, 257, 258, 259};
vector<Eigen::Isometry3f> lidarPoses;
vector<PointCloud::Ptr> static_cloud_vec;
vector<PointCloud::Ptr> dynamic_cloud_vec;

float pointDistance(PointXYZI p) {
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

void readLidarPoses(string& poseFile) {
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
}

void readLidarFile(string& cloud_dir, int count) {
    for (int i = 0; i < count; ++i) {
        std::stringstream frame_filename_str;
        frame_filename_str << std::setfill('0') << std::right << std::setw(6) << i << ".bin";
        string cloud_file = cloud_dir + frame_filename_str.str();
        PointCloud::Ptr curCloud(new PointCloud());
        PointCloud::Ptr curCloudDS(new PointCloud());
        curCloud->clear();
        curCloud->reserve(200000);
        std::fstream in(cloud_file, std::ios::in | std::ios::binary);
        while(in.good() && !in.eof()){
            PointXYZI p;
            in.read((char *)&p.x, sizeof(float));
            in.read((char *)&p.y, sizeof(float));
            in.read((char *)&p.z, sizeof(float));
            in.read((char *)&p.intensity, sizeof(float));
            curCloud->push_back(p);
  	    }
	    in.close();

        pcl::KdTreeFLANN<PointXYZI>::Ptr scan_kdtree;
        scan_kdtree.reset(new pcl::KdTreeFLANN<PointXYZI>());
        PointCloud::Ptr cur_static_cloud(new PointCloud());
        PointCloud::Ptr cur_dynamic_cloud(new PointCloud());

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
                if (p.z < -1.5 && p.z > -2) {
                    is_static = true;
                }
                
                if(!is_static) {
                    cur_dynamic_cloud->push_back(p);
                }
                else {
                    cur_static_cloud->push_back(p);
                }
        }
        static_cloud_vec.push_back(cur_static_cloud);
        dynamic_cloud_vec.push_back(cur_dynamic_cloud);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "no input sequence" << endl;
        return -1;
    }
    std::string seq = argv[1];
    std::string cloud_dir = "/home/eric/data/erasor/" + seq + "/velodyne/";
    std::string pose_file = "/home/eric/data/erasor/" + seq + "/pose.txt";
    cout << "cloud_dir: " << cloud_dir << endl;
    cout << "pose_file: " << pose_file << endl;
    readLidarPoses(pose_file);
    readLidarFile(cloud_dir, lidarPoses.size());
    cout << "tatally read " << static_cloud_vec.size() << " lidar file" << endl;

    // generate gt file
    PointCloud::Ptr global_static_cloud(new PointCloud());
    PointCloud::Ptr global_dynamic_cloud(new PointCloud());
    int n = static_cloud_vec.size();
    for (int i = 0; i < n; ++i) {
        PointCloud::Ptr cur_cloud_trans(new PointCloud());
        pcl::transformPointCloud(*static_cloud_vec[i], *cur_cloud_trans, lidarPoses[i]);
        *global_static_cloud += *cur_cloud_trans;
        pcl::transformPointCloud(*dynamic_cloud_vec[i], *cur_cloud_trans, lidarPoses[i]);
        *global_dynamic_cloud += *cur_cloud_trans;
    }
    cout << "global static cloud size: " << global_static_cloud->size() << endl;
    cout << "global dynamic cloud size: " << global_dynamic_cloud->size() << endl;

    // down sample
    pcl::VoxelGrid<PointXYZI> down_sample;
    down_sample.setLeafSize(0.2, 0.2, 0.2);
    PointCloud::Ptr global_static_cloud_ds(new PointCloud());
    PointCloud::Ptr global_dynamic_cloud_ds(new PointCloud());
    down_sample.setInputCloud(global_static_cloud);
    down_sample.filter(*global_static_cloud_ds);
    down_sample.setInputCloud(global_dynamic_cloud);
    down_sample.filter(*global_dynamic_cloud_ds);
    // save cloud
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    string saveFileDir = "/home/eric/file/gt/" + seq;
    pcl::io::savePCDFileBinary(saveFileDir + "/globalStaticMap.pcd", *global_static_cloud_ds);
    pcl::io::savePCDFileBinary(saveFileDir + "/globalDynamicMap.pcd", *global_dynamic_cloud_ds);
    cout << "done" << endl;
    cout << "****************************************************" << endl;
}