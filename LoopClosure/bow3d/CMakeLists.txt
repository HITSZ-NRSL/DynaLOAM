find_package(catkin REQUIRED COMPONENTS  

)


include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}  
  ${CHOLMOD_INCLUDE_DIR}
)
    
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS pcl_ros roscpp roslib rospy std_msgs 
    DEPENDS EIGEN3 PCL
)

add_executable(bow3d 
      src/Frame.cpp       
      src/LinK3D_Extractor.cpp 
      src/BoW3D.cpp 
      Example.cpp
      )

target_link_libraries(bow3d ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${G2O_LIBS} ${Sophus_LIBRARIES})

add_library(bow3d_lib src/Frame.cpp       
            src/LinK3D_Extractor.cpp 
            src/BoW3D.cpp)
target_link_libraries(bow3d_lib ${CERES_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
 

