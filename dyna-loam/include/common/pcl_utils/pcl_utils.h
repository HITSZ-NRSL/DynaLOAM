#ifndef __PCL_UTILS_H__
#define __PCL_UTILS_H__

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <fstream>  


namespace pcl_utils {
    using namespace std;

    void readPCDFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath);

    void savePCDFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath);

    void readBinFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath);

    void saveBinFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath);
}

#endif