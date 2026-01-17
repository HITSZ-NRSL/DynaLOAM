cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-g -O3 -pthread")

add_executable(octotree octotree.cpp)
target_link_directories(octotree PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_include_directories(octotree PRIVATE ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
target_link_libraries(octotree ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} pcl_utils)

add_executable(octomap_slam octomap_slam.cpp)
target_link_directories(octomap_slam PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_include_directories(octomap_slam PRIVATE ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
target_link_libraries(octomap_slam ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} pcl_utils)

