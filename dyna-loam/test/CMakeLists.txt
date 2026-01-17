cmake_minimum_required(VERSION 3.0.2)

add_executable(pubPCDCloud pubPCDCloud.cpp)
target_include_directories(pubPCDCloud PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_directories(pubPCDCloud PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_link_libraries(pubPCDCloud ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} Boost::filesystem pcl_utils)

add_executable(pubBinCloud pubBinCloud.cpp)
target_include_directories(pubBinCloud PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_directories(pubBinCloud PUBLIC /home/eric/a_ros_ws/lio_sam_lrc/devel/lib)
target_link_libraries(pubBinCloud ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} Boost::filesystem pcl_utils)


add_executable(testPCA testPCA.cpp)
target_include_directories(testPCA PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(testPCA ${PCL_LIBRARIES})

add_executable(generateGT generateGT.cpp)
target_include_directories(generateGT PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(generateGT ${PCL_LIBRARIES})

add_executable(segmentRes segmentRes.cpp)
target_include_directories(segmentRes PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(segmentRes ${PCL_LIBRARIES})