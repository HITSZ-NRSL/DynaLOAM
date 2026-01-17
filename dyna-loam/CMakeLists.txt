cmake_minimum_required(VERSION 3.0.2)
project(dynamic_removal)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  roslib
  cv_bridge
  rosbag
  # pcl library
  pcl_conversions
  # msgs
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  visualization_msgs
  image_transport
  message_filters
)

find_package(OpenMP REQUIRED)

add_message_files(
  DIRECTORY msg
  FILES
  keyScan.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  nav_msgs
)

catkin_package()


find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(octomap REQUIRED)
find_package(Threads REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_subdirectory(pcl_utils)
add_subdirectory(offline)
add_subdirectory(test)

