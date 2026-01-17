cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-g -O3 -pthread")
add_definitions(-DJSON_FILE="${CMAKE_CURRENT_SOURCE_DIR}/../config/dynamicFilter16.json")

add_executable(bagRemoval bagRemoval.cpp)
target_include_directories(bagRemoval PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(bagRemoval ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(bagRecord bagRecord.cpp)
target_include_directories(bagRecord PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_directories(bagRecord PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_link_libraries(bagRecord ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} Boost::filesystem pcl_utils)

add_executable(bagParse bagParse.cpp)
target_include_directories(bagParse PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_directories(bagParse PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_link_libraries(bagParse ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} Boost::filesystem pcl_utils)

add_executable(submapGen submapGen.cpp)
target_link_directories(submapGen PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_include_directories(submapGen PRIVATE ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
target_compile_options(submapGen PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(submapGen ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} pcl_utils visibilityMethod)

add_executable(segment segment.cpp)
target_link_directories(segment PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_include_directories(segment PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(segment ${PCL_LIBRARIES} pcl_utils)


