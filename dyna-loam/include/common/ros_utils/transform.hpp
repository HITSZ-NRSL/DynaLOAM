#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Transform.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace ros_utils {
    geometry_msgs::Transform OdomToTransform(nav_msgs::Odometry& odom) {
        geometry_msgs::Transform t;
        t.translation.x = odom.pose.pose.position.x;
        t.translation.y = odom.pose.pose.position.y;
        t.translation.z = odom.pose.pose.position.z;
        t.rotation = odom.pose.pose.orientation;
        return t;
    }

    Eigen::Matrix4f TransformToMatrix(geometry_msgs::Transform &transform) {
        Eigen::Quaternionf q(transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m.block<3, 1>(0, 3) = Eigen::Vector3f(transform.translation.x, transform.translation.y, transform.translation.z);
        return m;
    }
}

#endif