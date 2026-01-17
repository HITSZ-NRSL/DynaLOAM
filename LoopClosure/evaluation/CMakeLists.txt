include_directories(
    include
    ../bow3d/include
    ../scancontext/include
    ../lidar_iris/LidarIris
    # ${PCL_INCLUDE_DIRS}
    # ${EIGEN3_INCLUDE_DIRS}
    # ${Boost_INCLUDE_DIR}
    # ${catkin_INCLUDE_DIRS}  
    # ${CHOLMOD_INCLUDE_DIR}
)


add_executable(kitti_lc kitti_lc.cpp)
target_link_libraries(kitti_lc bow3d_lib sc_lib iris_lib
${catkin_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
