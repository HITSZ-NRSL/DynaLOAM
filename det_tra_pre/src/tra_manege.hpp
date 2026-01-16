// Copyright 2021 hsx
#ifndef TRA_MANEGE
#define TRA_MANEGE

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
#include <deque>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include <nav_msgs/Odometry.h>
#include"pc_data_loader.hpp"
#include"tra_prediction.hpp"
#include"pc_image_det_handle.hpp"
#include"HungarianAlg.h"
using namespace std;
using namespace cv;
using PointT = pcl::PointXYZINormal;

struct RetData{
    char trk_label;
    float pos_3dbbox[7];
    vector<Eigen::Vector3f> pos_his_master;
    vector<Eigen::Vector3f> pos_pre_master;
};

struct RecordData{
    double time_stamp;
    int trk_id;
    char object_id;
    float cur_abs_pos[3];
    float cur_abs_vel[3];
    vector<Eigen::Vector3f> pre_abs_pos;
};

struct DetOdomData{
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef shared_ptr<DetOdomData> Ptr;

    double start_time_,time_stamp_;
    nav_msgs::OdometryPtr p_tracker_odom_;
    vector<Eigen::Vector4f> pc_det_center_v_;
    vector<int> img_class_v_;
    vector<Eigen::Vector2f> img_center_v_;
    cv::Mat img_;

    DetOdomData(){}
    DetOdomData(double start_time,double time,int pc_pre_size,float* pc_pre_array,
            int img_pre_object_size, float* img_pre_data, char* img_data,double* odom_array){
        start_time_ = start_time;
        time_stamp_ = time;
        p_tracker_odom_.reset(new nav_msgs::Odometry);
        p_tracker_odom_->pose.pose.orientation.x  = odom_array[0];
        p_tracker_odom_->pose.pose.orientation.y  = odom_array[1];
        p_tracker_odom_->pose.pose.orientation.z  = odom_array[2];
        p_tracker_odom_->pose.pose.orientation.w = odom_array[3];
        p_tracker_odom_->pose.pose.position.x  = odom_array[4];
        p_tracker_odom_->pose.pose.position.y  = odom_array[5];
        p_tracker_odom_->pose.pose.position.z  = odom_array[6];

        Eigen::Vector4f c_temp(0,0,0,1);
        for(int i=0;i<pc_pre_size;i++){
            c_temp[0] = pc_pre_array[i*3];
            c_temp[1] = pc_pre_array[i*3+1];
            c_temp[2] = pc_pre_array[i*3+2];
            if(!c_temp.hasNaN())
                pc_det_center_v_.emplace_back(c_temp);
        }

        // Eigen::Vector2f i_c_tmep;
        // for(int i=0;i<img_pre_object_size;i++){
        //     img_class_v_.emplace_back(img_pre_data[i*3]);
        //     i_c_tmep[0] = img_pre_data[i*3+1];
        //     i_c_tmep[1] = img_pre_data[i*3+2];
        //     img_center_v_.emplace_back(i_c_tmep);
        // }

        // img_ = cv::Mat(480,640,CV_8UC3);
        // uchar *cvoutImg = reinterpret_cast<uchar*>(img_.data);
        // memcpy(cvoutImg, img_data,  480*640*3);
    }
};

// template<typename T>
// bool is_element_in_vector(vector<T> v,T element){
// 	vector<T>::iterator it;
// 	it=find(v.begin(),v.end(),element);
// 	if (it!=v.end()){
// 		return true;
// 	}else{
// 		return false;
// 	}
// }

class TraManeger{
 public:
        int max_age_, min_hits_, frame_count_, his_max_length_, fut_max_length_;
        // vel_lenght
        deque<ObjectTrajectory> trackers_;
        int trk_id_;
        Eigen::Vector3f avg_vel_;

 private:
        AssignmentProblemSolver hugAl_;


 public:
        TraManeger(int max_age_f, int min_hits_f, int his_max_length_f, int fut_max_length_f):
        max_age_(max_age_f),
        min_hits_(min_hits_f),
        his_max_length_(his_max_length_f),
        fut_max_length_(fut_max_length_f),
        trk_id_(0),
        avg_vel_(Eigen::Vector3f::Zero()){}

        void associate_data(const vector<Eigen::Vector2f>& trackers, const vector<Eigen::Vector2f>& detections, assignments_t& match,const double threhold = 640) {  // assignments_t& unmatched_detections, assignments_t& unmatched_trackers,  float distance_threshold=0.2
            distMatrix_t M;
            for (int i=0;i<detections.size();++i) {
                for (int j=0;j<trackers.size();++j) {
                    M.push_back((detections[i]-trackers[j]).norm());
                }
            }
            // assignments_t ass_index;
            hugAl_.Solve(M, trackers.size(), detections.size(), match);
            for (int i = 0; i < trackers.size(); ++i) {
                if(match[i] >= 0 && (trackers[i]-detections[match[i]]).norm()>threhold){ 
                    match[i] = -1;
                } 
            }
        }

        void associate_data(const vector<Eigen::Vector3f>& trackers, const vector<Eigen::Vector3f>& detections, assignments_t& match,
            const double threhold=100) {  // assignments_t& unmatched_detections, assignments_t& unmatched_trackers,  float distance_threshold=0.2
            distMatrix_t M;
            for (int i=0;i<detections.size();++i) {
                for (int j=0;j<trackers.size();++j) {
                    M.push_back((detections[i]-trackers[j]).norm());
                }
            }
            // assignments_t ass_index;
            // cout<<"data"<<endl;
            hugAl_.Solve(M, trackers.size(), detections.size(), match);
            // for (int i = 0; i < trackers.size(); ++i) {
            //     if (match[i] >= 0) {  //&&dis_mat[i][match[i]].norm()>threhold
            //         Eigen::Vector3f dis_v(detections[match[i]] - trackers[i]);
            //         float norm_tmp = dis_v.norm();
            //         // cout<<"trk_id: "<<trackers_[i].hits<<" norm: "<<norm_tmp<<endl;
            //         if(norm_tmp<0.3){
            //             continue;
            //         }else if(norm_tmp>2){
            //             match[i] = -1;
            //             continue;
            //         }
            //         // if(trackers_[i].hits<min_hits_){
            //         //     continue;
            //         // }
            //         // cout<<"norm: "<<norm_tmp<<endl;
            //         Eigen::Vector3f A(Eigen::Vector3f::Zero());
            //         for(int mat_idx=0;mat_idx<2;++mat_idx){
            //             float tmp =0.4+0.075*fabs(trackers_[i].vel_abs_estimate_[mat_idx]) ; //  + (trackers_[i].time_since_update-1)*0.4
            //             // cout<<"tmp "<<tmp<<" ";
            //             A[mat_idx]= tmp;
            //         }
            //         // cout<<"vel"<<endl;
            //         // cout<<trackers_[i].vel_abs_estimate_.transpose()<<endl;
            //         // cout<<"A"<<endl;
            //         // cout<<A.transpose()<<endl;
            //         // cout<<"dis"<<endl;
            //         // cout<<dis_v.transpose()<<endl;
            //         bool is_in_thegama = false;
            //         for(int mat_idx=0;mat_idx<2;++mat_idx){
            //             // float posity = A[mat_idx]/sqrt(2*M_PI)*exp(-0.5*dis_v[mat_idx]*A[mat_idx]*A[mat_idx]*dis_v[mat_idx]);
            //             // float thegama_posity = A[mat_idx]/sqrt(2*M_PI)*exp(-0.5);
            //             // cout<<"idx "<<mat_idx<<"po "<<to_string(posity)<<" "<<to_string(thegama_posity)<<endl;
            //             // if(posity>=thegama_posity)
            //             //     is_in_thegama = true;
            //             // else{
            //             //     is_in_thegama = false;
            //             //     break;
            //             // }
            //             if(fabs(dis_v[mat_idx])<=A[mat_idx])
            //                 is_in_thegama = true;
            //             else{
            //                 is_in_thegama = false;
            //                 break;
            //             }
            //         }   
            //         if(!is_in_thegama)
            //             match[i] = -1;
            //         // cout<<"match "<<match[i]<<endl;
            //     }
            // }
            for (int i = 0; i < trackers.size(); ++i) {
                if(match[i] >= 0){
                    // cout<<fabs(trackers_[i].vel_abs_estimate_[0])<<" "<<fabs(trackers_[i].vel_abs_estimate_[1])<<" "<<fabs(trackers_[i].vel_abs_estimate_[2])<<" "<<
                    //         ((trackers[i]-detections[match[i]])[0])<<" "<<((trackers[i]-detections[match[i]])[1])<<" "<<((trackers[i]-detections[match[i]])[2])<<" "<<
                    //         fabs(trackers_[i].time_since_update)<<endl;
                    if ((trackers[i]-detections[match[i]]).norm() > threhold) {
                        match[i] = -1;
                    }
                } 
            }
        }

        void update(const DetOdomData::Ptr& p_detOdomData, vector<Eigen::Vector2f>  center_uv,  vector<RetData>& ret, vector<RecordData>&  record_data
                                , vector<Eigen::Vector2f>& trks_pos_pre_ret,vector<Eigen::Vector3f>& ret_pre_pos_world) {
            vector<Eigen::Vector3f> trks_pos_world_pre;
            trks_pos_world_pre.reserve(trackers_.size());
            for (int i = 0; i < trackers_.size(); ++i) { // frame world
                trks_pos_world_pre.emplace_back(Eigen::Vector3f(trackers_[i].get_tra_pre_next()[0], trackers_[i].get_tra_pre_next()[1],
                                                                                        trackers_[i].get_tra_pre_next()[2]));
            }
            // cout<<"update trks pre ok "<<endl;

            vector<Eigen::Vector3f> det_pc_pos_world(p_detOdomData->pc_det_center_v_.size());
            vector<int> det_pc_not_update_index(p_detOdomData->pc_det_center_v_.size());
            for (int i = 0; i < det_pc_pos_world.size(); ++i) {
                det_pc_not_update_index[i] = i;
                det_pc_pos_world[i] = change2world(p_detOdomData->pc_det_center_v_[i], p_detOdomData->p_tracker_odom_);
            }

            assignments_t trk_pc_match;
            associate_data(trks_pos_world_pre, det_pc_pos_world, trk_pc_match,0.7);    // if too many it will be very slow 

            TraData det_data;
            for (int trk_index = 0; trk_index < trackers_.size(); ++trk_index) {
                if (trk_pc_match[trk_index] >= 0) {
                    det_data.time_stamp = p_detOdomData->time_stamp_;
                    det_data.pos_3dbbox[0] = det_pc_pos_world[trk_pc_match[trk_index]](0);
                    det_data.pos_3dbbox[1] = det_pc_pos_world[trk_pc_match[trk_index]](1);
                    det_data.pos_3dbbox[2] = det_pc_pos_world[trk_pc_match[trk_index]](2);
                    trackers_[trk_index].update(det_data);
                    det_pc_not_update_index[trk_pc_match[trk_index]] = -1;
                }
            }

            for (auto img_unmatched_idx:det_pc_not_update_index) {
                if (img_unmatched_idx >= 0) {
                    det_data.time_stamp = p_detOdomData->time_stamp_;
                    det_data.pos_3dbbox[0] = det_pc_pos_world[img_unmatched_idx](0);
                    det_data.pos_3dbbox[1] = det_pc_pos_world[img_unmatched_idx](1);
                    det_data.pos_3dbbox[2] = det_pc_pos_world[img_unmatched_idx](2);
                    trackers_.push_back(ObjectTrajectory(trk_id_++, 1, his_max_length_,
                        fut_max_length_, det_data,avg_vel_));
                }
            }

            // vector<RetData> ret;
            // vector<RecordData> record_data;
            Eigen::Vector3f pos_pre_master;
            vector<Eigen::Vector3f> vel_v;
            deque<ObjectTrajectory> new_trackers;
            for (int trk_idx=0; trk_idx < trackers_.size(); ++trk_idx) {
                RetData ret_trk;
                RecordData record_data_trk;
                TraData cur_state = trackers_[trk_idx].get_state();

                this->trackers_[trk_idx].predict(); // if don not have hits 3 , it will always age 0

                ret_pre_pos_world.emplace_back(change2master(Eigen::Vector4f(cur_state.pos_3dbbox[0], cur_state.pos_3dbbox[1],
                                                                                                                                                            cur_state.pos_3dbbox[2], 1), p_detOdomData->p_tracker_odom_));

                if ((trackers_[trk_idx].time_since_update < max_age_) && (trackers_[trk_idx].hits  >= min_hits_ )) {  //|| this->frame_count <= this->min_hits

                    Eigen::Vector3f cur_pos_master = Eigen::Vector3f(cur_state.pos_3dbbox[0], cur_state.pos_3dbbox[1],
                                                                                                                                                            cur_state.pos_3dbbox[2]);
                    ret_trk.trk_label = trackers_[trk_idx].tra_label;
                    ret_trk.pos_3dbbox[0] = cur_pos_master(0);
                    ret_trk.pos_3dbbox[1] = cur_pos_master(1);
                    ret_trk.pos_3dbbox[2] = cur_pos_master(2);
                    for (int i = 0; i < trackers_[trk_idx].his_pos.size(); ++i) {
                        ret_trk.pos_his_master.push_back(Eigen::Vector3f(trackers_[trk_idx].his_pos[i].pos_3dbbox[0], trackers_[trk_idx].his_pos[i].pos_3dbbox[1],
                                trackers_[trk_idx].his_pos[i].pos_3dbbox[2]));
                    }

                    for (int i=0; i< trackers_[trk_idx].fut_pos_pre.size(); i++) {
                        ret_trk.pos_pre_master.push_back(Eigen::Vector3f(trackers_[trk_idx].fut_pos_pre[i][0], trackers_[trk_idx].fut_pos_pre[i][1],
                                trackers_[trk_idx].fut_pos_pre[i][2]));
                        // record data
                        record_data_trk.pre_abs_pos.push_back(Eigen::Vector3f(trackers_[trk_idx].fut_pos_pre[i][0], trackers_[trk_idx].fut_pos_pre[i][1],
                                                                                                                                                                                                    trackers_[trk_idx].fut_pos_pre[i][2]));
                    }
                    ret.push_back(ret_trk);

                    // record data
                    record_data_trk.time_stamp = p_detOdomData->start_time_;
                    record_data_trk.trk_id = trackers_[trk_idx].trk_id_;
                    record_data_trk.object_id = trackers_[trk_idx].tra_label;
                    record_data_trk.cur_abs_pos[0] = cur_state.pos_3dbbox[0];
                    record_data_trk.cur_abs_pos[1] = cur_state.pos_3dbbox[1];
                    record_data_trk.cur_abs_pos[2] = cur_state.pos_3dbbox[2];
                    // record_data_trk.cur_abs_vel[0] = this->trackers[trk_idx].vel_abs_estimate[0];
                    // record_data_trk.cur_abs_vel[1] = this->trackers[trk_idx].vel_abs_estimate[1];
                    // record_data_trk.cur_abs_vel[2] = this->trackers[trk_idx].vel_abs_estimate[2];
                    record_data.push_back(record_data_trk);

                    vel_v.emplace_back(trackers_[trk_idx].vel_abs_estimate_);

                    // debug show
                    pos_pre_master = change2master(Eigen::Vector4f(trackers_[trk_idx].cur_pos.back().pos_3dbbox[0], trackers_[trk_idx].cur_pos.back().pos_3dbbox[1],
                                                                                        trackers_[trk_idx].cur_pos.back().pos_3dbbox[2], 1), p_detOdomData->p_tracker_odom_);
                    trks_pos_pre_ret.emplace_back(calib_.lidar2img(pos_pre_master(0), pos_pre_master(1), pos_pre_master(2)));
                }

                if (trackers_[trk_idx].time_since_update < max_age_) {
                    new_trackers.emplace_back(trackers_[trk_idx]);
                }
            }
            if(!vel_v.empty()){
                avg_vel_ .setZero();
                for(auto& v:vel_v)
                    avg_vel_ += v;
                avg_vel_ = avg_vel_/vel_v.size();
            }

            trackers_ = new_trackers;
        }
};
#endif
