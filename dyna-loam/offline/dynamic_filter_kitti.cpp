#include "dynamic_removal/dataLoader.h"
#include "dynamic_removal/tictoc.hpp"
#include "dynamic_removal/dynamicFilter.h"

using namespace std;
using PointXYZI = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<pcl::PointXYZI>;


std::shared_ptr<spdlog::logger> logger;

int main(int argc, char **argv)
{
    logger = spdlog::stdout_color_mt("console_main");
    string jsonFile = "/home/eric/a_ros_ws/dyna_loam_ws/src/dyna-loam/config/dynamicFilter_offline.json";
    fstream f(jsonFile);
    json data = json::parse(f);
    string cloudFileDir   = data["cloudFileDir"].get<string>();
    string poseFile       = data["poseFile"].get<string>();
    string saveFileDir    = data["saveFileDir"].get<string>();
    string configFile = data["configFile"].get<string>();
    string calibration_file = data["calibration_file"].get<string>();
    bool saveCloudFlag    = data["saveCloudFlag"].get<bool>(); 
    bool isKittiFormat = data["isKittiFormat"].get<bool>();
    bool evaluation = data["evaluation"].get<bool>();
    bool visRawCloud    = data["visRawCloud"].get<bool>();
    bool visGroundSeg   = data["visGroundSeg"].get<bool>();
    bool visVisibilitySeg = data["visVisibilitySeg"].get<bool>();
    bool visClusterSeg  = data["visClusterSeg"].get<bool>(); 
    bool visDynamic     = data["visDynamic"].get<bool>();
    bool visStatic      = data["visStatic"].get<bool>();
    bool visEvaluation  = data["visEvaluation"].get<bool>();
    int startPose      = data["startPose"].get<int>();
    int endPose        = data["endPose"].get<int>();
    int frameInterval  = data["frameInterval"].get<int>();
    int submapMaxSize  = data["submapMaxSize"].get<int>();
    double leafSize       = data["leafSize"].get<double>();

    DataLoaderBase::Ptr loader;
    loader.reset(new KittiFormatLoader());
    std::dynamic_pointer_cast<KittiFormatLoader>(loader)->loadKittiCalibration(calibration_file);
    loader->loadFrameInfo(cloudFileDir, poseFile, startPose, endPose);
    logger->info("load frames: {}", loader->getSize());

    for (int i = 0; i < loader->getSize(); ++i) {
        DataLoaderBase::Frame frame = loader->loadFrame(i);
        logger->info("frame {} size: {}", i, frame.frame->size());
    }
}