// segment res to true static and preserved dynamic

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <string>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>

using namespace std;
using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointXYZI>;

vector<int> DYNAMIC_CLASSES = {252, 253, 254, 255, 256, 257, 258, 259};

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "no input map file" << endl;
        return -1;
    }
    std::string map_file = argv[1];
    std::string map_file_dir = argv[2];
    PointCloud::Ptr map_cloud(new PointCloud());
    pcl::io::loadPCDFile(map_file, *map_cloud);
    
    PointCloud::Ptr preserved_static(new PointCloud());
    PointCloud::Ptr preserved_dynamic(new PointCloud());

    for (auto& p : map_cloud->points) {
        uint32_t float2int      = static_cast<uint32_t>(p.intensity);
        uint32_t semantic_label = float2int & 0xFFFF;
        uint32_t inst_label     = float2int >> 16;
        bool     is_static      = true;
        for (int class_num: DYNAMIC_CLASSES) {
            if (semantic_label == class_num) { 
                is_static = false;
            }
        }
        if (is_static) {
            preserved_static->push_back(p);
        } else {
            preserved_dynamic->push_back(p);
        }
    }

    // save cloud
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    pcl::io::savePCDFileBinary(map_file_dir + "/preservedStaticMap.pcd", *preserved_static);
    pcl::io::savePCDFileBinary(map_file_dir + "/preservedDynamic.pcd", *preserved_dynamic);
    cout << "done" << endl;
    cout << "****************************************************" << endl;
}