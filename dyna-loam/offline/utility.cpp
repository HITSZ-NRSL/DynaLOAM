#include "dynamic_removal/utility.h"

namespace removert{

SphericalPoint cart2sph(const PointXYZI & _cp){
    SphericalPoint sph_point {
        std::atan2(_cp.y, _cp.x), 
        std::atan2(_cp.z, std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y)),
        std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y + _cp.z*_cp.z)
};    
return sph_point;
}

std::set<int> convertIntVecToSet(const std::vector<int> & v) 
{ 
    std::set<int> s; 
    for (int x : v) { 
        s.insert(x); 
    } 
    return s; 
} 

void pubRangeImg(cv::Mat& _rimg, 
                sensor_msgs::ImagePtr& _msg,
                image_transport::Publisher& _publiser,
                std::pair<float, float> _caxis)
{
    cv::Mat scan_rimg_viz = convertColorMappedImg(_rimg, _caxis);
    _msg = cvmat2msg(scan_rimg_viz);
    _publiser.publish(_msg);    
} // pubRangeImg

sensor_msgs::ImagePtr cvmat2msg(const cv::Mat &_img)
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _img).toImageMsg();
  return msg;
}

}