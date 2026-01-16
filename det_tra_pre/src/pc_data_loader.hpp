// Copyright 2021 hsx
#ifndef PC_DATA_LOADER
#define PC_DATA_LOADER

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include <tf/transform_broadcaster.h>
using namespace std;
using namespace cv;
using PointType = pcl::PointXYZINormal;

class Calib {
 public:
        Eigen::Matrix4f R0_rect = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f P2 = Eigen::Matrix4f::Identity();
        float p2[12] = {319.9988245765257, 0, 320.5, 0,
                                 0, 319.9988245765257, 240.5, 0, 
                                 0, 0, 1, 0};
        Eigen::Matrix4f Trv2c = Eigen::Matrix4f::Identity();
        float trv2c[12] = {0,   -1,  0,  -0.01,
                                        0,  0,   -1,  -0.07,
                                        1,  0, 0,   -0.02};
        pcl::PointCloud<PointType>::Ptr p_pc_frstum;
        int img_h = 480;
        int img_w = 640;
        cv::Mat depth_img;

 public:
        Calib() {
            for (int i = 0; i < 12; i++) {
                P2(i/4, i%4) = p2[i];
                Trv2c(i/4, i%4) = trv2c[i];
            }
            // memcpy(P2.data(), p2, sizeof(p2));
            // memcpy(Trv2c.data(), trv2c, sizeof(trv2c));
            // cout<<"P2 "<<P2.block(0, 0, 4, 4)<<endl;
            // cout<<"Trv2c "<<Trv2c.block(0, 0, 4, 4)<<endl;
         }

        ~Calib() {}

        void get_frstum_point(pcl::PointCloud<PointType>::Ptr p_pc) {
            p_pc_frstum.reset(new pcl::PointCloud<PointType>);
            Eigen::Vector3f pt;
            for (auto& p : p_pc->points) {
                if (p.x<0 || p.x >20 || p.y < -5 || p.y > 5 || p.z > 0 || p.z < -1.15 )
                    continue;
                pt = (P2*Trv2c*Eigen::Vector4f(p.x, p.y, p.z, 1)).topRows(3);
                int32_t u = static_cast<int32_t>(pt(0)/pt(2));
                int32_t v = static_cast<int32_t>(pt(1)/pt(2));
                if (u >= 0 && u < img_w && v >= 0 && v < img_h) {
                    p_pc_frstum->points.push_back(p);
                }
            }
        }

        Eigen::Vector3f img2lidar(int u, int v, float depth) {
            Eigen::Vector4f pt;
            pt(0) = ((u - P2(0, 2))*depth ) / P2(0, 0) + P2(0, 3) / (-P2(0, 0));
            pt(1) = ((v - P2(1, 2))*depth ) / P2(1, 1) + P2(1, 3) / (-P2(1, 1));
            pt(2) = depth;
            pt(3) = 1;
            pt = (Trv2c.inverse()*pt);
            return pt.block(0, 0, 3, 0);
        }

        Eigen::Vector2f lidar2img(const Eigen::Vector4f& p_raw){
            Eigen::Vector4f pt;
            pt = (P2*Trv2c*p_raw);
            float u = (pt(0)/pt(2));
            float v = (pt(1)/pt(2));
            return Eigen::Vector2f(u, v);
        }

        Eigen::Vector2f lidar2img(float x, float y, float z){
            Eigen::Vector4f pt;
            pt = (P2*Trv2c*Eigen::Vector4f(x, y, z, 1));
            float u = (pt(0)/pt(2));
            float v = (pt(1)/pt(2));
            return Eigen::Vector2f(u, v);
        }
};

// class PcDataLoader{
//     public:

// }
Calib calib_;
#endif
