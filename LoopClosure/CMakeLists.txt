cmake_minimum_required(VERSION 3.0.2)
project(loop_closure)

add_compile_options(-std=c++14 -O3)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  roslib
  rospy  
  std_msgs  
)

find_package(Eigen3 REQUIRED)
find_package(PCL)
find_package(OpenCV)
find_package(yaml-cpp REQUIRED)

catkin_package()
include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/bow3d")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/lidar_iris")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/scancontext")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/evaluation")




