// use kitti dataset to test various loop closure detection algorithm
#include <iostream>
#include <ros/ros.h>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Dense>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <sstream>
#include <iomanip>
#include "LinK3D_Extractor.h"
#include "BoW3D.h"

#include "Scancontext.h"
#include "LidarIris.h"
#include "tic_toc.h"


using namespace std;

string sequence;
string lidar_folder;
string gt_file;
string evaluation_folfer = "/home/eric/a_ros_ws/dyna_loam_ws/src/LoopClosure/evaluation/kitti/";
std::vector<pcl::PointXYZI> traj_vec;
pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
vector<vector<int>> gt;

int N = 1000;

std::vector<vector<int>> getGTFromPose() {
    std::ifstream pose_ifs(gt_file);
    std::string line;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    int index = 0;
    while(getline(pose_ifs, line)) 
    {
        if(line.empty()) break;
        stringstream ss(line);
        float r1,r2,r3,t1,r4,r5,r6,t2,r7,r8,r9,t3;
        ss >> r1 >> r2 >> r3 >> t1 >> r4 >> r5 >> r6 >> t2 >> r7 >> r8 >> r9 >> t3;
        pcl::PointXYZI p;
        p.x = t1;
        p.y = 0;
        p.z = t3;
        p.intensity = index++;
        cloud->push_back(p);
        traj_vec.push_back(p);
    }    
    kdtree.setInputCloud(cloud);
    std::vector<vector<int>> res(5000);
    for(int i = 0; i < cloud->points.size(); i++)
    {
        float radius = 4;
        std::vector<int> ixi;
        std::vector<float> ixf;
        pcl::PointXYZI p = cloud->points[i];
        int cur = p.intensity;
        std::vector<int> nrs;
        kdtree.radiusSearch(p,radius,ixi,ixf);
        for(int j = 0; j < ixi.size(); j++)
        {
            if(cur - cloud->points[ixi[j]].intensity  < 300) 
                continue;
            nrs.push_back(cloud->points[ixi[j]].intensity);
        }
        sort(nrs.begin(), nrs.end());
        res[cur] = nrs;
    }
    
    std::ofstream gt_ofs(evaluation_folfer + "gt/"+ sequence + ".txt");

    for(int i =0; i < cloud->points.size(); i++)
    {
        gt_ofs << i << " ";
        for(int j = 0; j < res[i].size(); j++)
        {
            gt_ofs << res[i][j] << " ";
        }
        gt_ofs << endl;
    }
    return res;
}

void bow3d() {
    //Parameters of LinK3D
    int nScans = 64; //Number of LiDAR scan lines
    float scanPeriod = 0.1; 
    float minimumRange = 0.1;
    float distanceTh = 0.4;
    int matchTh = 6;

    //Parameters of BoW3D
    float thr = 3.5;
    int thf = 5;
    int num_add_retrieve_features = 5;
    vector<int> bow_table;

    BoW3D::LinK3D_Extractor* pLinK3dExtractor = new BoW3D::LinK3D_Extractor(nScans, scanPeriod, minimumRange, distanceTh, matchTh); 
    BoW3D::BoW3D* pBoW3D = new BoW3D::BoW3D(pLinK3dExtractor, thr, thf, num_add_retrieve_features);

    // int N = traj_vec.size();
    std::ofstream bow3d_ofs(evaluation_folfer + "bow3d/"+ sequence + ".txt");

    for (int i = 0; i < N; i++) {
        // load cloud
        std::stringstream lidar_data_path;
        lidar_data_path << lidar_folder << std::setfill('0') << std::setw(6) << i << ".bin";
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        current_cloud->clear();
        current_cloud->reserve(200000);
        std::fstream in(lidar_data_path.str(), std::ios::in | std::ios::binary);
        while(in.good() && !in.eof()){
            pcl::PointXYZI p;
            pcl::PointXYZ p_;
            in.read((char *)&p.x, sizeof(float));
            in.read((char *)&p.y, sizeof(float));
            in.read((char *)&p.z, sizeof(float));
            in.read((char *)&p.intensity, sizeof(float));
            p_.x = p.x;
            p_.y = p.y;
            p_.z = p.z;
            current_cloud->push_back(p_);
        }
	    in.close();
        bow_table.push_back(i);
        BoW3D::Frame* pCurrentFrame = new BoW3D::Frame(pLinK3dExtractor, current_cloud); 
        if(pCurrentFrame->mnId < 1)
        {
            pBoW3D->update(pCurrentFrame);  
        }
        else
        {                
            int loopFrameId = -1;
            Eigen::Matrix3d loopRelR;
            Eigen::Vector3d loopRelt;
            clock_t start, end;
            double _time;       
            start = clock();
            pBoW3D->retrieve(pCurrentFrame, loopFrameId, loopRelR, loopRelt); 
            end = clock();
            _time = ((double) (end - start)) / CLOCKS_PER_SEC;
            pBoW3D->update(pCurrentFrame);               

            if(loopFrameId == -1)
            {
                cout << "-------------------------" << endl;
                cout << "Detection Time: " << _time << "s" << endl;
                cout << "Frame" << pCurrentFrame->mnId << " Has No Loop..." << endl;
            }
            else
            {
                cout << "--------------------------------------" << endl;
                cout << "Detection Time: " << _time << "s" << endl;
                cout << "Frame " << pCurrentFrame->mnId << " Has Loop Frame " << loopFrameId << endl;
                
                cout << "Loop Relative R: " << endl;
                cout << loopRelR << endl;
                                
                cout << "Loop Relative t: " << endl;                
                cout << "   " << loopRelt.x() << " " << loopRelt.y() << " " << loopRelt.z() << endl;

                int cur_id = pCurrentFrame->mnId;

                int cur = i;
                int his = bow_table[loopFrameId];

                if (std::find(gt[cur].begin(),gt[cur].end(),his)!=gt[cur].end()) {
                    bow3d_ofs << cur << " " << his << " " << 1 << std::endl; 
                } else {
                    bow3d_ofs << cur << " " << his << " " << 0 << std::endl;
                }   
            }
        }                       
    }

}

void sc() {
    SCManager scManager;
    // int N = traj_vec.size();
    std::ofstream sc_ofs(evaluation_folfer + "sc/"+ sequence + ".txt");
    vector<int> sc_table;
    for (int i = 0; i < N; i+=1) {
        // load cloud
        std::stringstream lidar_data_path;
        lidar_data_path << lidar_folder << std::setfill('0') << std::setw(6) << i << ".bin";
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        current_cloud->clear();
        current_cloud->reserve(200000);
        std::fstream in(lidar_data_path.str(), std::ios::in | std::ios::binary);
        while(in.good() && !in.eof()){
            pcl::PointXYZI p;
            in.read((char *)&p.x, sizeof(float));
            in.read((char *)&p.y, sizeof(float));
            in.read((char *)&p.z, sizeof(float));
            in.read((char *)&p.intensity, sizeof(float));
            current_cloud->push_back(p);
        }
	    in.close();
        scManager.makeAndSaveScancontextAndKeys(*current_cloud);
        sc_table.push_back(i);
        auto detect_result = scManager.detectLoopClosureID();
        if (detect_result.first != -1) {
            int curr_id = i;
            int loop_id = detect_result.first;
            loop_id = sc_table[loop_id];
            float dis = detect_result.second;
            if (std::find(gt[i].begin(),gt[i].end(),loop_id)!=gt[i].end()) {
                sc_ofs << curr_id << " " << loop_id << " " << dis << " " << 1 << std::endl; 
            } else {
                sc_ofs << curr_id << " " << loop_id << " " << dis << " " << 0 << std::endl;
            }
        }
        

    }
}

void iris() {
    std::ofstream iris_ofs(evaluation_folfer + "iris/"+ sequence + ".txt");
    const int loop_event = 0;
    // const int N = traj_vec.size();
    LidarIris iris(4, 18, 1.6, 0.75, loop_event);
    std::vector<LidarIris::FeatureDesc> dataset(N);
    std::vector<bool> valid_dataset(N, false);
    for(int i = 0; i < N ; i+=1)
    {
        std::stringstream ss;
        ss << setw(6) << setfill('0') << i;
        cout << ss.str()+".bin" << std::endl;

        // kitti velodyne bins
        std::stringstream lidar_data_path;
        lidar_data_path << lidar_folder << std::setfill('0') << std::setw(6) << i << ".bin";
        std::string filename = lidar_data_path.str();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>);
        std::fstream input(filename, std::ios::in | std::ios::binary);
        input.seekg(0, std::ios::beg);
        for (int ii=0; input.good() && !input.eof(); ii++) {
            pcl::PointXYZ point;
            input.read((char *) &point.x, 3*sizeof(float));
            float intensity;
            input.read((char *) &intensity, sizeof(float));
            cloud0->push_back(point);
        }
        cv::Mat1b li1 = LidarIris::GetIris(*cloud0);
        LidarIris::FeatureDesc fd1 = iris.GetFeature(li1);
        dataset[i] = fd1;
        valid_dataset[i] = true;
        float mindis = 1000;
        int loop_id = -1;

        std::vector<int> ixi;
        std::vector<float> ixf;
        kdtree.radiusSearch(traj_vec[i], 20, ixi, ixf);


        for (int j = 0; j < ixi.size(); ++j) {
            int id = traj_vec[ixi[j]].intensity;
            if (id >= i)
                continue;
            if (id >= i - 300)
                continue;
            LidarIris::FeatureDesc fd2 = dataset[id];

            int bias;
            auto dis = iris.Compare(fd1, fd2, &bias);
            if(dis < mindis)
            {
                mindis = dis;
                loop_id = id;
            }
        }

        if(loop_id == -1) 
            continue;
        if (std::find(gt[i].begin(),gt[i].end(),loop_id)!=gt[i].end()) {
            iris_ofs << i << " " << loop_id << " " << mindis << " " << 1 << std::endl; 
            cout << i << " " << loop_id << " " << mindis << " " << 1 << std::endl; 
        } else {
            iris_ofs << i << " " << loop_id << " " << mindis << " " << 0 << std::endl;
            cout << i << " " << loop_id << " " << mindis << " " << 0 << std::endl; 
        }
        
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "wrong input param" << endl;
        return -1;
    }
    string algorithm = argv[1];
    sequence = argv[2];
    // lidar_folder = "/media/eric/Elements SE/data/kitti_odom/dataset/sequences/" + sequence + "/velodyne/";
    lidar_folder = "/media/eric/Elements SE/data/filtered_kitti_odom/05/";
    gt_file = "/media/eric/Elements SE/data/kitti_odom/dataset/poses/"+ sequence +".txt";

    cout << "loop closure test " << endl;
    cout << "use algorithm: " << algorithm << endl;
    cout << "sequence: " << sequence << endl;
    cout << "--------------------------------" << endl;
    common::TicToc t1;
    N = 1500;
    gt = getGTFromPose();
    if (algorithm == "bow3d") {
        bow3d();
    } 
    else if (algorithm == "sc") {
        sc();
    }
    else if (algorithm == "iris") {
        iris();
    } else {
        cout << "no valid loop closure algorithm" << endl;
    }

    cout << "per frame cost: " << t1.toc() / N << endl; 

    return 0;
}