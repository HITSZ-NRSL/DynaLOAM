// Copyright 2021 hsx
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf/transform_broadcaster.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include<visualization_msgs/MarkerArray.h>
#include<visualization_msgs/Marker.h>

// #include<cv_bridge/cv_bridge.h>
#include "cv_bridge/cv_bridge.h"
#include<opencv2/opencv.hpp>
#include <nav_msgs/Odometry.h>

// #include <pcl/conversions.h>

#include <map>

#include <pcl/filters/passthrough.h>

#include"api_share_ros.cpp"
#include"pc_data_loader.hpp"
#include"pc_image_det_handle.hpp"
#include"ros_show.hpp"
#include"tra_manege.hpp"
#include <chrono>
#include<mutex>
#include<thread>
#include <image_transport/image_transport.h>

using namespace Eigen;
using namespace std;
using PointType = pcl::PointXYZINormal;

// tratory maneger
TraManeger tra_maneger_(5, 3,10, 10);

// record data
std::chrono::system_clock::time_point t1_, t2_;
string record_data_path_;

deque<DetOdomData::Ptr> deq_detOdomData_;
mutex deq_detOdomData_mutex_;
image_transport::Publisher image_pub_;

void getDataThread(){
    while(true){
        while(share_men_instance_.p_share_data->pc_flag!=3||
        share_men_instance_.p_share_data->odom_flag!=3){
            ros::Duration(0.00001).sleep();
        }

        DetOdomData::Ptr p_det;
        p_det.reset(new DetOdomData(share_men_instance_.p_share_data->start_time,share_men_instance_.p_share_data->time_stamp,
        share_men_instance_.p_share_data->pc_pre_size,share_men_instance_.p_share_data->pc_pre_data,
        share_men_instance_.p_share_data->img_pre_object_size,share_men_instance_.p_share_data->img_pre_data,share_men_instance_.p_share_data->img_data,
        share_men_instance_.p_share_data->odom_array));
        {
            lock_guard<mutex> lck(deq_detOdomData_mutex_);
            deq_detOdomData_.emplace_back(p_det);
        }

        //debug
        // cout<<"tra pre time "<<p_det->time_stamp_<<endl;
        // cout<<"tra pre odom "<<p_det->p_tracker_odom_->pose.pose.orientation.x<<" "<<
        // p_det->p_tracker_odom_->pose.pose.orientation.y<<" "<<
        // p_det->p_tracker_odom_->pose.pose.orientation.z<<" "<<
        // p_det->p_tracker_odom_->pose.pose.orientation.w<<" "<<
        // p_det->p_tracker_odom_->pose.pose.position.x<<" "<<
        // p_det->p_tracker_odom_->pose.pose.position.y<<" "<<
        // p_det->p_tracker_odom_->pose.pose.position.z<<endl;
        // for(auto& c:p_det->pc_det_center_v_)
        //     cout<<"tra pre pc det "<<c.transpose()<<endl;
        // for(auto& c:p_det->img_class_v_)
        //     cout<<"tra pre img class "<<c<<endl;
        // for(auto& c:p_det->img_center_v_)
        //     cout<<"tra pre img cneter "<<c.transpose()<<endl;
        // cv::imshow("test", p_det->img_);
        // cv::waitKey(1);

        share_men_instance_.p_share_data->pc_flag = 0;
        share_men_instance_.p_share_data->odom_flag = 0;
    }
}

void trackThread() {
    while(true){
        t1_ = std::chrono::system_clock::now();
        DetOdomData::Ptr p_detOdomData;
        {
            lock_guard<mutex> lck(deq_detOdomData_mutex_);
            if(deq_detOdomData_.empty())
                continue;
            p_detOdomData = deq_detOdomData_.front();
            deq_detOdomData_.pop_front();
        }
        vector<Eigen::Vector2f>  center_uv;
        // for (const auto& c: p_detOdomData->pc_det_center_v_) {
        //     center_uv.push_back(calib_.lidar2img(c));
        //     // cout<<"pc det "<<c.transpose()<<endl;
        // }
        vector<RetData> cur_state;
        vector<RecordData>  record_data;
        vector<Eigen::Vector2f> pre_pos;
        vector<Eigen::Vector3f> pre_pos_world;
        tra_maneger_.update(p_detOdomData, center_uv, cur_state, record_data, pre_pos,pre_pos_world);

        //share memory
        while(share_men_instance_.p_share_data->dynamic_obj_flag!=1)
            ros::Duration(0.001).sleep();
        share_men_instance_.p_share_data->dynamic_obj_size = pre_pos_world.size();
        for(int i=0;i<share_men_instance_.p_share_data->dynamic_obj_size;++i){
            share_men_instance_.p_share_data->dynamic_obj[i*3] = pre_pos_world[i][0];
            share_men_instance_.p_share_data->dynamic_obj[i*3+1] = pre_pos_world[i][1];
            share_men_instance_.p_share_data->dynamic_obj[i*3+2] = pre_pos_world[i][2]; 
        }
        share_men_instance_.p_share_data->dynamic_obj_flag =0;
        //share memory

        //data record
        // ofstream OutFile(record_data_path_, ios::app); // 利用构造函数创建txt文本，并且打开该文本
        // if(!record_data.empty()){
        //     OutFile << to_string( (ros::Time::now().toSec() - record_data[0].time_stamp)*1000 );
        //     OutFile << endl;
        // }
        // OutFile.close();

        publish_result(cur_state);

        t2_ = std::chrono::system_clock::now();
        cout << "cost time   " << std::chrono::duration_cast<std::chrono::microseconds>( t2_-t1_ ).count()*0.001<<" ms" << std::endl;
        ros::Duration(0.000001).sleep();
    }
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "det_tra_pre");
    ros::NodeHandle nh;

    vector<float> p2_V;
    vector<float> trv2c_V;
    nh.param<vector<float>>("p2", p2_V, vector<float>());
    nh.param<vector<float>>("trv2c", trv2c_V, vector<float>());
    nh.param<string>("record_data_path", record_data_path_, "");
    std::remove(record_data_path_.c_str()); //删除原来保存的文件
    calib_.P2 = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(p2_V.data(), 4, 4);
    calib_.Trv2c = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(trv2c_V.data(), 4, 4);

    result_pub_ = nh.advertise<visualization_msgs::MarkerArray>("result", 1);

    thread get_data_thread(getDataThread);
    get_data_thread.detach();

    thread track_thread(trackThread);
    track_thread.detach();

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}
