#include "common/pcl_utils/pcl_utils.h"

namespace pcl_utils {
    void readPCDFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath){
        cloudInptr->clear();
        if (pcl::io::loadPCDFile<pcl::PointXYZI> (filePath, *cloudInptr) == -1) //* load the file
        {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        }
        return;
    }

    void savePCDFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath) {
        if(cloudInptr->empty()) {
            cerr << "cloud is empty" << endl;
            return;
        }
        else {
            pcl::io::savePCDFile<pcl::PointXYZI>(filePath, *cloudInptr);
        }
    }

    void readBinFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath){
        cloudInptr->clear();
        int32_t num = 10000000;
        float *data = (float*)malloc(num * sizeof(float));
        float *px = data + 0;
        float *py = data + 1;
        float *pz = data + 2;
        float *pr = data + 3;//反射强度
        fstream input(filePath.c_str(), ios::in | ios::binary);
        if(!input.good()){
        cerr << "Couldn't read in_file: " << filePath << endl;
        }
        for (int i = 0; input.good() && !input.eof(); i++) {
            pcl::PointXYZI point;
            input.read((char*) &point.x, 3*sizeof(float));
            input.read((char*) &point.intensity, sizeof(float));
            cloudInptr->push_back(point);
        }
        input.close();
    }

    void saveBinFile(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInptr, string& filePath) {
        if(cloudInptr->empty()) {
            cerr << "cloud is empty" << endl;
            return;
        }
        std::ofstream myFile(filePath.c_str(), std::ios::out | std::ios::binary);
        for (int i = 0; i < cloudInptr->size(); i++) {
            myFile.write((char*)& cloudInptr->at(i).x, sizeof(cloudInptr->at(i).x));
            myFile.write((char*)& cloudInptr->at(i).y, sizeof(cloudInptr->at(i).y));
            myFile.write((char*)& cloudInptr->at(i).z, sizeof(cloudInptr->at(i).z));
            myFile.write((char*)& cloudInptr->at(i).intensity, sizeof(cloudInptr->at(i).intensity));
        }
        myFile.close();
    }

}

