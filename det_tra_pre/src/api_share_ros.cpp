// Copyright 2021 hsx
#ifndef API_SHARE_ROS
#define API_SHARE_ROS

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
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
using namespace std;
using PointType = pcl::PointXYZINormal;

namespace ShareMem {
    #define Shm_addrees 1203 // 共享内存地址标识
    typedef struct ShareData {
        double time_stamp;
        double start_time;
        double end_time;
        int  pc_flag;
        int pc_size;
        float pc_array_data[4*100*1000]; // 图像数据一维数据，之前用了cv::Mat不行，因为无法在结构体里初始化大小
        float pc_pre_data[3*100];
        int pc_pre_size;

        int img_flag;
        char img_data[480*640*3];
        int img_pre_object_size;
        float img_pre_data[3*100];

        int odom_flag;
        double odom_array[7];

        float dynamic_obj[3*100];
        int dynamic_obj_flag;
        int dynamic_obj_size;

        int trk_flag;
        int trk_size;
        float trk_his_pos[100*20*3];
        float trk_pred_pos[100*40*3];
    }  ShareData_;

class ShareMemory {
 public:
    int shmid = shmget((key_t)Shm_addrees, sizeof(ShareData), 0666|IPC_CREAT);
    void *shm = shmat(shmid, 0, 0);
    ShareData *p_share_data = reinterpret_cast<ShareData*>(shm);
    int img_h = 480;
    int img_w = 640;

 public:
    ShareMemory() {
        p_share_data->pc_flag = 0;
        p_share_data->img_flag = 0;
        p_share_data->odom_flag = 0;
        p_share_data->img_pre_object_size = 0;
        p_share_data->pc_pre_size = 0;
        p_share_data->dynamic_obj_size =0;
        p_share_data->dynamic_obj_flag = 0;
        p_share_data->trk_flag = 0;
        printf("共享内存地址 ： %p\n", (shm));
    }

    ~ShareMemory() {
        cout << "析构函数执行" << endl;
        DestroyShare();
    }

    void DestroyShare() {
        shmdt(shm);
        shmctl(shmid, IPC_RMID, 0);
        cout << "共享内存已经销毁" << endl;
    }

    void send_pc_array(pcl::PointCloud<PointType>::Ptr p_pc, int points_size) {
        if (p_share_data->pc_flag == 0) {
            if (p_pc== nullptr) {
                printf("pc ptr not exits\n");
                return;
            } else {
                p_share_data->pc_size = points_size;
                for (int i = 0; i < points_size; i++) {
                    p_share_data->pc_array_data[i *4 ] = p_pc->points[i].x;
                    p_share_data->pc_array_data[i *4 + 1] = p_pc->points[i].y;
                    p_share_data->pc_array_data[i *4 + 2] = p_pc->points[i].z;
                    p_share_data->pc_array_data[i *4 + 3] = 0;  // p_pc->points[i].intensity
                }
                p_share_data->pc_flag = 1;
            }
        }
    }

    void get_pc_array(float (*pc_array)[4],int*  points_size) {
        points_size[0] = p_share_data->pc_size;
        for (int i = 0; i < p_share_data->pc_size; i++) {
            pc_array[i][0] = p_share_data->pc_array_data[i *4 ];
            pc_array[i][1] = p_share_data->pc_array_data[i *4 + 1];
            pc_array[i][2] = p_share_data->pc_array_data[i *4 + 2];
            pc_array[i][3] = p_share_data->pc_array_data[i *4 + 3];  
        }
        p_share_data->pc_flag = 2;
    }

    void get_pc_flag(int* flag){
        *flag = p_share_data->pc_flag ;   
    }

    void send_pc_pre_array(float*  pc_pre, int pc_pre_size) {
        if (p_share_data->pc_flag == 2) {
            if (pc_pre== nullptr) {
                printf("文件不存在\n");
                return;
            } else {
                p_share_data->pc_pre_size = pc_pre_size;
                for (int i = 0; i < pc_pre_size*3; i++) {
                    p_share_data->pc_pre_data[i] = pc_pre[i];
                }
                p_share_data->pc_flag = 3;
            }
        }
    }

    void send_img(cv::Mat img) {
        if (p_share_data->img_flag == 0) {
            uchar *cvoutImg = reinterpret_cast<uchar*>(img.data);
            memcpy(p_share_data->img_data, cvoutImg, img_h*img_w*3);
            p_share_data->img_flag = 1;
        }
    }

    void get_img(uchar (*img)[640][3]) {
        for(int r=0;r<480;r++){
            for(int c=0;c<640;c++){
                img[r][c][0] = p_share_data->img_data[r*(640*3)+c*3];
                img[r][c][1] = p_share_data->img_data[r*(640*3)+c*3 +1];
                img[r][c][2] = p_share_data->img_data[r*(640*3)+c*3+2];
            }
        }
        p_share_data->img_flag = 2;
    }

    void get_img_flag(int* flag) {
        *flag = p_share_data->img_flag;
    }

    void send_img_pre_data(float* img_pre, int img_pre_object_size) {
        if (p_share_data->img_flag == 2) {
            p_share_data->img_pre_object_size = img_pre_object_size;
            for (int i = 0; i < img_pre_object_size*3; i++) {
                p_share_data->img_pre_data[i] = img_pre[i];
            }
            p_share_data->img_flag = 3;
        }
    }

    void send_odom(double* odom){
        if(p_share_data->odom_flag==0){
            for(int i=0;i<7;i++)
                p_share_data->odom_array[i] = odom[i];
            p_share_data->odom_flag =3;
        }
    }

    void get_trk_flag(int* flag) {
        *flag = p_share_data->trk_flag;
    }

    void send_trk(const vector<cv::Mat> &trk_his_pos){
        if(p_share_data->trk_flag == 0){
            p_share_data->trk_size = trk_his_pos.size();
            for(int i=0;i<trk_his_pos.size();++i){
                for(int j=0;j<20;++j){
                    p_share_data->trk_his_pos[i*60+j*3] = trk_his_pos[i].at<float>(j,0);
                    p_share_data->trk_his_pos[i*60+j*3+1] = trk_his_pos[i].at<float>(j,1);
                    p_share_data->trk_his_pos[i*60+j*3+2] = trk_his_pos[i].at<float>(j,2);
                }
            }
            p_share_data->trk_flag = 1;
        }
    }

    void get_trk_his_pos(float (*trk_his_pos)[3],int*  trk_size) {
        trk_size[0] = p_share_data->trk_size;
        for (int i = 0; i < p_share_data->trk_size; ++i) {
            for(int p_i = 0;p_i<20;++p_i){
                trk_his_pos[i*20+p_i][0] = p_share_data->trk_his_pos[i *60+p_i*3];
                trk_his_pos[i*20+p_i][1] = p_share_data->trk_his_pos[i *60+p_i*3 + 1];
                trk_his_pos[i*20+p_i][2] = p_share_data->trk_his_pos[i *60+p_i*3 + 2];
            }
        }
        p_share_data->trk_flag = 2;
    }

    void send_trk_pre(float* trk_pred) {
        if (p_share_data->trk_flag == 2) {
            for (int i = 0; i < p_share_data->trk_size*600; ++i) {
                p_share_data->trk_pred_pos[i] = trk_pred[i];
            }
            p_share_data->trk_flag = 3;
        }
    }
};
} // namespace ShareMem

// 按照C语言格式重新打包-python调用
ShareMem::ShareMemory share_men_instance_;
extern "C" {

    void DestroyShare_() {
         share_men_instance_.DestroyShare();
    }

    void get_pc_array(float (*pc_array)[4],int* points_size) {
         share_men_instance_.get_pc_array(pc_array,points_size);
    }

    void get_pc_flag(int* flag){
        share_men_instance_.get_pc_flag(flag);
    }

    void send_pc_pre_array(float*  pc_pre, int pc_pre_size) {
        share_men_instance_.send_pc_pre_array(pc_pre, pc_pre_size);
    }

    void get_img(uchar (*img)[640][3]) {
         share_men_instance_.get_img(img);
    }

    void get_img_flag(int* flag) {
            share_men_instance_.get_img_flag(flag);
    }

    void send_img_pre_data(float* img_pre, int img_pre_object_size) {
        share_men_instance_.send_img_pre_data(img_pre,  img_pre_object_size);
    }

    void get_trk_flag(int* flag){
        share_men_instance_.get_trk_flag(flag);
    }

    void get_trk_his_pos(float (*trk_his_pos)[3],int* trk_size) {
         share_men_instance_.get_trk_his_pos(trk_his_pos,trk_size);
    }

    void send_trk_pre(float*  trk_pred) {
        share_men_instance_.send_trk_pre(trk_pred);
    }

}

#endif
