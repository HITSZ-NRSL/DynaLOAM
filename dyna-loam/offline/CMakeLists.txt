cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-g -O3 -pthread")
add_definitions(-DJSON_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../config/dynamicFilter16.json")

add_library(visibilityMethod visibilityMethod.cpp)
target_include_directories(visibilityMethod PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_options(visibilityMethod PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(visibilityMethod ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} pcl_utils)

add_library(dynamicFilter dynamicFilter.cpp)
target_include_directories(dynamicFilter PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_options(dynamicFilter PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(dynamicFilter ${PCL_LIBRARIES} ${catkin_LIBRARIES} pcl_utils visibilityMethod)

add_executable(dynamic_filter_offline dynamic_filter_offline.cpp)
target_link_libraries(dynamic_filter_offline dynamicFilter)

add_executable(visibilitySegment visibilitySegment.cpp)
target_link_libraries(visibilitySegment dynamicFilter)

add_executable(dynamic_filter_online dynamic_filter_online.cpp)
target_link_libraries(dynamic_filter_online dynamicFilter ${catkin_LIBRARIES})

add_executable(dynamic_filter_kitti dynamic_filter_kitti.cpp)
target_link_libraries(dynamic_filter_kitti dynamicFilter ${catkin_LIBRARIES})


