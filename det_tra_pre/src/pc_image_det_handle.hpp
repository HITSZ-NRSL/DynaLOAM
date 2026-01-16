// Copyright 2021 hsx
#ifndef PC_IMAGE_DET_HANDLE
#define PC_IMAGE_DET_HANDLE

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
#include"pc_data_loader.hpp"
using namespace std;
using namespace cv;
using PointT = pcl::PointXYZINormal;

Eigen::Vector3f change2world(Eigen::Vector4f pos, const nav_msgs::OdometryConstPtr& master_traj_gt_msg) {
    Eigen::Quaternionf q_odom(master_traj_gt_msg->pose.pose.orientation.w, master_traj_gt_msg->pose.pose.orientation.x,
                              master_traj_gt_msg->pose.pose.orientation.y, master_traj_gt_msg->pose.pose.orientation.z);
    Eigen::Vector3f pos_odom(master_traj_gt_msg->pose.pose.position.x,
                             master_traj_gt_msg->pose.pose.position.y, master_traj_gt_msg->pose.pose.position.z);
    Eigen::Matrix4f T_matrix = Eigen::Matrix4f::Identity();
    T_matrix.block<3, 3>(0, 0) = q_odom.toRotationMatrix();
    T_matrix.block<3, 1>(0, 3) = pos_odom;
    pos = T_matrix*pos;
    Eigen::Vector3f pt;
    pt(0) = pos(0);
    pt(1) = pos(1);
    pt(2) = pos(2);
    return pt;
}

Eigen::Vector3f change2master(Eigen::Vector4f pos, const nav_msgs::OdometryConstPtr& master_traj_gt_msg) {
    Eigen::Quaternionf q_odom(master_traj_gt_msg->pose.pose.orientation.w, master_traj_gt_msg->pose.pose.orientation.x,
                              master_traj_gt_msg->pose.pose.orientation.y, master_traj_gt_msg->pose.pose.orientation.z);
    Eigen::Vector3f pos_odom(master_traj_gt_msg->pose.pose.position.x,
                             master_traj_gt_msg->pose.pose.position.y, master_traj_gt_msg->pose.pose.position.z);
    Eigen::Matrix4f T_matrix = Eigen::Matrix4f::Identity();
    T_matrix.block<3, 3>(0, 0) = q_odom.toRotationMatrix();
    T_matrix.block<3, 1>(0, 3) = pos_odom;
    pos = T_matrix.inverse()*pos;
    Eigen::Vector3f pt;
    pt(0) = pos(0);
    pt(1) = pos(1);
    pt(2) = pos(2);
    return pt;
}

#endif
