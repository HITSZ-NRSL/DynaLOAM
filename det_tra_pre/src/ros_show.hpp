// Copyright 2021 hsx
#ifndef ROS_SHOW
#define ROS_SHOW

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
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include<visualization_msgs/MarkerArray.h>
#include<visualization_msgs/Marker.h>
#include"pc_data_loader.hpp"
#include"tra_manege.hpp"
using namespace std;
using namespace cv;
using PointT = pcl::PointXYZINormal;

ros::Publisher result_pub_;
float array_3dbbox_[8][3] = {1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1};
cv::Mat mat_3dbbox_(8, 3, CV_32F, reinterpret_cast<float*>(array_3dbbox_ ));

cv::Mat boxes_to_corners_3d(const float  bbox[7]) {
    cv::Mat corners = cv::Mat::zeros(8, 3, CV_32F);
    corners.colRange(0, 1) = mat_3dbbox_.colRange(0, 1) *bbox[3]/2;
    corners.colRange(1, 2) = mat_3dbbox_.colRange(1, 2) *bbox[4]/2;
    corners.colRange(2, 3) = mat_3dbbox_.colRange(2, 3) *bbox[5]/2;
    float ca = cos(bbox[6]);
    float sa = sin(bbox[6]);
    // float T_array[4][4]={ca,sa,0,0,
    //                                           -sa, ca,0,0,
    //                                           0,0,1,0,
    //                                           bbox[0],bbox[1],bbox[2],1 };
    float T_array[3][3]={ca, sa, 0,
                                            -sa, ca, 0,
                                            0, 0, 1 };
    cv::Mat T(3, 3, CV_32F, (float*)T_array );
    corners = corners*T;
    add(bbox[0],  corners.colRange(0, 1),  corners.colRange(0, 1));
    add(bbox[1],  corners.colRange(1, 2),  corners.colRange(1, 2));
    add(bbox[2],  corners.colRange(2, 3),  corners.colRange(2, 3));
    return corners;
}

void publish_result(const vector<RetData>& ret) {
    string frame_id = "camera_init";
    float life = 0.4;
    // 3d bbox
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.type = marker.SPHERE_LIST;
    marker.action = marker.ADD;
    // marker.lifetime = rospy.Duration()
    marker.lifetime = ros::Duration(life);
    marker.header.stamp = ros::Time::now();

    //  marker scale (scale y and z not used due to being linelist)
    marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3; 
    //  marker color
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    geometry_msgs::Point p;
    for (int i = 0; i < ret.size(); i++) {
        p.x = ret[i].pos_3dbbox[0];
        p.y = ret[i].pos_3dbbox[1];
        p.z = ret[i].pos_3dbbox[2];
        marker.points.push_back(p);
        marker_array.markers.push_back(marker);

        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id;
        text_marker.header.stamp =  ros::Time::now();

        text_marker.id = i + 1000;
        text_marker.action = marker.ADD;
        // text_marker.lifetime = rospy.Duration()
        text_marker.lifetime = ros::Duration(life);
        text_marker.type = marker.TEXT_VIEW_FACING;

        text_marker.pose.position.x = ret[i].pos_3dbbox[0];
        text_marker.pose.position.y =ret[i].pos_3dbbox[1];
        text_marker.pose.position.z = ret[i].pos_3dbbox[2] + 0.5;

        string name;
        if (ret[i].trk_label == 0)
            name = string("oth");
        if (ret[i].trk_label == 1)
            name = string("tar");
        if (ret[i].trk_label == 2)
            name = string("ped");
        text_marker.text = name;

        text_marker.scale.x = 1;
        text_marker.scale.y = 1;
        text_marker.scale.z = 1;

        text_marker.color.r = 1.0;
        text_marker.color.g = 0.0;
        text_marker.color.b = 0.0;
        text_marker.color.a = 1.0;
        marker_array.markers.push_back(text_marker);
    }

    int id = 0;
    for (auto& m:marker_array.markers) {
        m.id = id;
        id += 1;
    }

    for (int i = 0; i < ret.size(); i++) {
        if (ret[i].trk_label == 0) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = ros::Time::now();
            marker.action = visualization_msgs::Marker::ADD;
            // marker.lifetime = rospy.Duration()
            marker.lifetime = ros::Duration(life);
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.id = 100;
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            marker.scale.x = 0.2;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            for (auto pos:ret[i].pos_his_master) {
                p.x = pos(0);
                p.y = pos(1);
                p.z = pos(2);
                marker.points.push_back(p);
            }
            marker_array.markers.push_back(marker);
        }
        if (ret[i].trk_label == 1) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = ros::Time::now();
            marker.action = visualization_msgs::Marker::ADD;
            // marker.lifetime = rospy.Duration()
            marker.lifetime = ros::Duration(life);
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.id = 101;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            marker.scale.x = 0.2;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            for (auto pos:ret[i].pos_his_master) {
                p.x = pos(0);
                p.y = pos(1);
                p.z = pos(2);
                marker.points.push_back(p);
            }
            marker_array.markers.push_back(marker);
        }
        if (ret[i].trk_label == 2) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = ros::Time::now();
            marker.action = visualization_msgs::Marker::ADD;
            // marker.lifetime = rospy.Duration()
            marker.lifetime = ros::Duration(life);
            marker.type = visualization_msgs::Marker::LINE_STRIP;
            marker.id = 102+i;
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.8;
            marker.color.a = 1.0;
            marker.scale.x = 0.2;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            for (auto pos:ret[i].pos_his_master) {
                p.x = pos(0);
                p.y = pos(1);
                p.z = pos(2);
                marker.points.push_back(p);
                // cout<<"white his pos "<<pos(0)<<" "<<pos(1)<<" "<<pos(2)<<" "<<endl;
            }
            marker_array.markers.push_back(marker);
        }
    }

    for (int i=0; i < ret.size(); i++) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.action = visualization_msgs::Marker::ADD;
        // marker.lifetime = rospy.Duration()
        marker.lifetime = ros::Duration(life);
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.id = 1000+ i;
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;
        marker.scale.x = 0.2;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        for (auto pos:ret[i].pos_pre_master) {
            p.x = pos(0);
            p.y = pos(1);
            p.z = pos(2);
            marker.points.push_back(p);
            // cout<<"pre pos "<<pos(0)<<" "<<pos(1)<<" "<<pos(2)<<" "<<endl;
        }
        marker_array.markers.push_back(marker);
    }
    result_pub_.publish(marker_array);
}

void showDetationResult(const vector<RetData> & ret){
    if(ret.empty())
        return;
    string frame_id = "map";
    float life = 0.4;
    // 3d bbox
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.type = marker.SPHERE_LIST;
    marker.action = marker.ADD;
    // marker.lifetime = rospy.Duration()
    marker.lifetime = ros::Duration(life);
    marker.header.stamp = ros::Time::now();

    //  marker scale (scale y and z not used due to being linelist)
    marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3; 
    //  marker color
    marker.color.a = 1.0;
    marker.color.r = 0;
    marker.color.g = 1;
    marker.color.b = 0;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    geometry_msgs::Point p;
    for (int i = 0; i < ret.size(); i++) {
        p.x = ret[i].pos_3dbbox[0];
        p.y = ret[i].pos_3dbbox[1];
        p.z = ret[i].pos_3dbbox[2];
        marker.points.push_back(p);
        marker_array.markers.push_back(marker);
    }

    int id = 10000;
    for (auto& m:marker_array.markers) {
        m.id = id;
        id += 1;
    }
    result_pub_.publish(marker_array);
}
    

#endif
