// Copyright 2021 hsx
#ifndef TRA_PREDICTION
#define TRA_PREDICTION

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
#include <deque>
#include"pc_data_loader.hpp"
#include<random>
#include<unordered_set>
using namespace std;
using namespace cv;
using PointT = pcl::PointXYZINormal;

typedef struct TraData {
    double time_stamp;
    float pos_3dbbox[7];
}TraData_;

template<typename T1, typename T2, typename T3>
void fixed_size_push_back(T1 &cur_pos, T2 data, T3 length) {
    if (cur_pos.size() < length) {
        cur_pos.push_back(data);
    } else {
        cur_pos.pop_front();
        cur_pos.push_back(data);
    }
}

class ObjectTrajectory{
 public:
        int  his_max_length, fut_max_length,trk_id_;  // vel_length
        char tra_label;
        deque<TraData> his_pos;
        vector<vector<float>> fut_pos_pre;
        int time_since_update = 0;
        int hits = 0;
        Eigen::Vector3f vel_abs_estimate_;

        // record data
        float pos_next_abs_estimate[3]={0};
        deque<TraData> cur_pos;
        // record data
 private:
        bool is_predict = false;

        // kf
        int dim_x_ = 9;
        int dim_obs_  = 3;
        Eigen::MatrixXf kf_x_ = Eigen::MatrixXf::Zero(dim_x_, 1);
        Eigen::MatrixXf kf_P_ = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
        Eigen::MatrixXf kf_F_ = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
        Eigen::MatrixXf kf_Q_ = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
        Eigen::MatrixXf kf_H_ = Eigen::MatrixXf::Zero(dim_obs_, dim_x_);
        Eigen::MatrixXf kf_R_ = Eigen::MatrixXf::Identity(dim_obs_, dim_obs_);
        Eigen::MatrixXf kf_y_ = Eigen::MatrixXf::Zero(dim_obs_, 1);
        Eigen::MatrixXf kf_K_ = Eigen::MatrixXf::Zero(dim_x_, dim_obs_);
        Eigen::MatrixXf kf_I_ = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
        double kf_time_last_;

        //ransac 
        int sample_selcect_size_ = 6;
        int iter_max_times_= 50;
        int inliner_size_threshold_ = 3;
        double threshold_ = 0.5;
        cv::Mat best_model_;
        double best_error_ = 1e10;
        default_random_engine  gen_;
        uniform_int_distribution<int> dist_;
        unordered_set<int> us_;

 public:
        ObjectTrajectory(int trk_id,char tra_label_f, int his_max_length_f, int fut_max_length_f, TraData mea_data,Eigen::Vector3f& vel_begin):
        trk_id_(trk_id),
        tra_label (tra_label_f),
        his_max_length (his_max_length_f),
        fut_max_length ( fut_max_length_f),
        kf_time_last_(mea_data.time_stamp),
        vel_abs_estimate_(vel_begin)
        { 
            fut_pos_pre = vector<vector<float>>(fut_max_length_f, vector<float>(3, 0));

            // kf
            kf_x_ << mea_data.pos_3dbbox[0], mea_data.pos_3dbbox[1], mea_data.pos_3dbbox[2], 
                            vel_abs_estimate_[0], vel_abs_estimate_[1], vel_abs_estimate_[2],0,0,0;
            // kf_P_(0, 0) = 0.5;  kf_P_(1, 1) = 0.5;
            for (int i = 0;  i < dim_x_; i++)
                kf_P_(i, i) = 0.5;

            kf_Q_ = kf_Q_*10;

            kf_H_(0, 0) = 1; kf_H_(1, 1) = 1;kf_H_(2, 2) = 1;
            kf_R_ = kf_R_*1;

            //ransac
            gen_ = default_random_engine(10);

            update(mea_data);
        }
        ~ObjectTrajectory() {}

        void update(const TraData& mea_data) {
            time_since_update = 0;
            hits++;
            fixed_size_push_back(cur_pos, mea_data, 1);
            fixed_size_push_back(his_pos, mea_data, his_max_length);
            is_predict = false;
        }

        double motion_model(double t, Mat beta ) {
            return beta.at<double>(2, 0)*pow(t, 2)+beta.at<double>(1, 0)*t+beta.at<double>(0, 0);
        }

        Mat rls(const vector<double>& t, const vector<double>& data) {
            Mat A = Mat::zeros(t.size(), 3, CV_64FC1);
            for (int row = 0; row < A.rows; row++) {
                for (int col = 0; col < A.cols; col++) {
                    A.at<double>(row, col) = pow(t[row], col);
                }
            }
            // 构建B矩阵
            Mat B = Mat::zeros(t.size(), 1, CV_64FC1);
            for (int row = 0; row < B.rows; row++) {
                B.at<double>(row, 0)= data[row];
            }
            Mat X;
            solve(A, B, X, CV_SVD);
            return X;
        }

        double calError(double y_pre,double y){
            return pow(y_pre-y,2);
        }

        void ransacFit(vector<double>& t, vector<double>& data,cv::Mat& model_ret){
            if(t.size()<=sample_selcect_size_){
                model_ret = rls(t,data);
                return;
            }
            dist_ = uniform_int_distribution<int>(0,t.size()-1);
            int counter = 0;
            while(counter<iter_max_times_){
                counter++;
                vector<double> t_se,data_se,t_less,data_less;
                while(us_.size()<sample_selcect_size_){
                    int idx = dist_(gen_);
                    if(us_.count(idx)==0){
                        us_.insert(idx);
                    }
                }

                for(int i=0;i<data.size();i++){
                    if(us_.count(i)){
                        t_se.emplace_back(t[i]);
                        data_se.emplace_back(data[i]);
                    }else{
                        t_less.emplace_back(t[i]);
                        data_less.emplace_back(data[i]);
                    }
                }
                us_.clear();

                cv::Mat X = rls(t_se,data_se);
                model_ret = X;

                vector<int> inliner_idx;
                for(int i=0;i<t_less.size();i++){ 
                    double error = calError(motion_model(t_less[i],X),data_less[i]);
                    if(error<threshold_){
                        inliner_idx.emplace_back(i);
                    }
                }

                if(inliner_idx.size()>inliner_size_threshold_){
                    for(auto& idx:inliner_idx){
                        t_se.emplace_back(t[idx]);
                        data_se.emplace_back(data[idx]);
                    }
                    X = rls(t_se,data_se);

                    double mean_error = 0;
                    for(int i=0;i<t_se.size();i++){
                        mean_error = mean_error + calError(motion_model(t_se[i],X),data_se[i]);
                    }
                    mean_error = mean_error/t_se.size();

                    if(mean_error<best_error_){
                        best_error_ = mean_error;
                        best_model_ = X;
                        model_ret = best_model_;
                    }
                }
            }
        }

        void predict() {   // must be called once
            time_since_update+=1;
            if (is_predict)
                return;

            for(int i=0;i<fut_max_length;++i){
                double delta_t = 0.1+i*0.1;
                fut_pos_pre[i][0] = kf_x_(0, 0)+kf_x_(3, 0)*delta_t+kf_x_(6, 0)*delta_t*delta_t/2;
                fut_pos_pre[i][1] = kf_x_(1, 0)+kf_x_(4, 0)*delta_t+kf_x_(7, 0)*delta_t*delta_t/2;
                fut_pos_pre[i][2] = kf_x_(2, 0)+kf_x_(5, 0)*delta_t+kf_x_(8, 0)*delta_t*delta_t/2;
            }

            is_predict = true;
        }

        TraData get_state() {   // after update  before predict
            TraData ret;
            if (time_since_update == 0) {
                ret = this->cur_pos.back();
            } else {
                if (this->is_predict) {
                    TraData re_data;
                    re_data.time_stamp = this->his_pos.back().time_stamp + time_since_update*0.1;
                    re_data.pos_3dbbox[0] = this->fut_pos_pre[time_since_update - 1][0];
                    re_data.pos_3dbbox[1] = this->fut_pos_pre[time_since_update - 1][1];
                    re_data.pos_3dbbox[2] = this->fut_pos_pre[time_since_update - 1][2];
                    ret =  re_data;
                } else {
                    ret =  this->his_pos.back();
                }
            }

            if(this->hits==3){
                Eigen::Vector3f vel_tmp(Eigen::Vector3f::Zero());
                vector<double> his_time;
                his_time.push_back(0);
                for (int i = 0; i < 2; ++i) {
                    his_time.push_back(his_pos[i+1].time_stamp - his_pos[i].time_stamp + his_time[i] );
                }
                for (int xyz_index = 0; xyz_index < 3; xyz_index++) {
                    vector<double> his_pos_xyz;
                    double first_pos = this->his_pos[0].pos_3dbbox[xyz_index];
                    for (int i = 0; i < 3; i++) {
                        his_pos_xyz.push_back(this->his_pos[i].pos_3dbbox[xyz_index] - first_pos);
                    }
                    Mat beta;
                    beta = rls(his_time,his_pos_xyz);
                    vel_tmp[xyz_index] = beta.at<double>(1, 0);
                }
                if((vel_tmp - vel_abs_estimate_).norm()>0.5){
                    // vel_tmp.normalize();
                    // vel_tmp = vel_tmp*vel_abs_estimate_.norm();
                    kf_x_.block<3,1>(3,0) = vel_tmp;
                }
            }

            double delta_t = ret.time_stamp-kf_time_last_;
            kf_time_last_ = ret.time_stamp;
            kf_F_(0, 3) = delta_t;
            kf_F_(0, 6) = delta_t*delta_t/2;
            kf_F_(1, 4) = delta_t;
            kf_F_(1, 7) = delta_t*delta_t/2;
            kf_F_(2, 5) = delta_t;
            kf_F_(2, 8) = delta_t*delta_t/2;

            kf_F_(3, 6) = delta_t;
            kf_F_(4, 7) = delta_t;
            kf_F_(5, 8) = delta_t;

            kf_x_ = kf_F_ * kf_x_;
            kf_P_ = kf_F_* kf_P_ * kf_F_.transpose() + kf_Q_;
            kf_K_  = kf_P_ * kf_H_.transpose() * (kf_H_ * kf_P_ * kf_H_.transpose() + kf_R_).inverse();
            kf_y_<< ret.pos_3dbbox[0], ret.pos_3dbbox[1], ret.pos_3dbbox[2];
            kf_x_ += kf_K_ * (kf_y_ - kf_H_ * kf_x_);
            kf_P_ = (kf_I_ - kf_K_ * kf_H_) * kf_P_;
            this->pos_next_abs_estimate[0] = kf_x_(0, 0)+kf_x_(3, 0)*delta_t+kf_x_(6, 0)*delta_t*delta_t/2;
            this->pos_next_abs_estimate[1] = kf_x_(1, 0)+kf_x_(4, 0)*delta_t+kf_x_(7, 0)*delta_t*delta_t/2;
            this->pos_next_abs_estimate[2] = kf_x_(2, 0)+kf_x_(5, 0)*delta_t+kf_x_(8, 0)*delta_t*delta_t/2;
            // this->pos_next_abs_estimate[0] = kf_x_(0, 0)+kf_x_(3, 0)*delta_t;
            // this->pos_next_abs_estimate[1] = kf_x_(1, 0)+kf_x_(4, 0)*delta_t;
            // this->pos_next_abs_estimate[2] = kf_x_(2, 0)+kf_x_(5, 0)*delta_t;
            vel_abs_estimate_ = kf_x_.block<3,1>(3,0);
            return ret;
        }

        vector<float> get_tra_pre_next() {
            return vector<float>({this->pos_next_abs_estimate[0],this->pos_next_abs_estimate[1],this->pos_next_abs_estimate[2]});
        }

        vector<Eigen::Vector3f> get_his_pos_world() {
            vector<Eigen::Vector3f> pos;
            for (int i = 0; i < this->his_pos.size(); i++) {
                pos.push_back(Eigen::Vector3f(this->his_pos[i].pos_3dbbox[0], this->his_pos[i].pos_3dbbox[1], this->his_pos[i].pos_3dbbox[2]));
            }
            return pos;
        }

        vector<Eigen::Vector3f> get_pre_pos_world() {
            vector<Eigen::Vector3f> pos;
            for (int i = 0; i < this->fut_pos_pre.size(); i++) {
                pos.push_back(Eigen::Vector3f(this->fut_pos_pre[i][0], this->fut_pos_pre[i][1], this->fut_pos_pre[i][2]));
            }
            return pos;
        }
};

#endif
