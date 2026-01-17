SET(CMAKE_BUILD_TYPE Release)
# set(OpenCV_DIR "/home/eric/programefiles/opencv3.4/opencv-3.4.15/build")


include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories("./LidarIris/")

add_executable(lidar_iris "demo.cpp" "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
target_link_libraries(lidar_iris ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_library(iris_lib "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
target_link_libraries(iris_lib ${PCL_LIBRARIES} ${OpenCV_LIBS})

# add_executable(front "front_view.cpp")
# target_link_libraries(front ${PCL_LIBRARIES} ${OpenCV_LIBS})


# add_executable(test_icp "test_icp.cpp" "LidarIris/LidarIris.cpp" "LidarIris/LidarIris.h" "LidarIris/fftm/fftm.cpp" "LidarIris/fftm/fftm.hpp")
# target_link_libraries(test_icp ${PCL_LIBRARIES} ${OpenCV_LIBS})

