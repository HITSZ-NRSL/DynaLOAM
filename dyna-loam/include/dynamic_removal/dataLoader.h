#ifndef DYNAMICREMOVAL_DATALOADER_H_
#define DYNAMICREMOVAL_DATALOADER_H_

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>

class DataLoaderBase
{
public:
  typedef std::shared_ptr<DataLoaderBase> Ptr;
  typedef std::shared_ptr<const DataLoaderBase> ConstPtr;

  struct Frame
  {
    size_t idx = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr frame = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Vector3f t = Eigen::Vector3f::Identity();
    Eigen::Quaternionf r = Eigen::Quaternionf::Identity();
  };
  
  struct FrameInfo
  {
    size_t idx = 0;
    std::string pcd_filename = "";
    Eigen::Vector3f t = Eigen::Vector3f::Identity();
    Eigen::Quaternionf r = Eigen::Quaternionf::Identity();
  };

protected:
  std::vector<FrameInfo> frame_info_buf_;

public:
  size_t getSize() const { return frame_info_buf_.size(); }
  FrameInfo getFrameInfo(const size_t idx) const
  {
    if(idx < frame_info_buf_.size())
      return frame_info_buf_[idx];
    return FrameInfo();
  }
  
  virtual size_t loadFrameInfo(const std::string &pcds_dir, const std::string &pose_file, const int start = 0, const int end = -1) = 0;
  virtual Frame loadFrame(const size_t idx) = 0;
};

class KittiFormatLoader : public DataLoaderBase
{
  Eigen::Affine3f calib_ = Eigen::Affine3f::Identity();
  Eigen::Affine3f calib_inv_ = Eigen::Affine3f::Identity();
  
  Eigen::Affine3f parseLine(const std::string &str)
  {
    int num = 0;
    std::string parse_str;
    std::istringstream i_stream(str);
    
    Eigen::Affine3f res = Eigen::Affine3f::Identity();
    while (getline(i_stream, parse_str, ' '))
    {
      res(num / 4, num %  4) = std::stof(parse_str);
      num++;
    }
    return res;
  }
  
public:
  bool loadKittiCalibration(const std::string &calibration_file)
  {
    std::ifstream ifs(calibration_file);
    std::string csv_line;
    while (getline(ifs, csv_line))
    {
      if(csv_line.find("Tr:") != std::string::npos){
        calib_ = parseLine(csv_line.substr(4));
        calib_inv_ = calib_.inverse();
        return true;
      }
    }
    return false;
  }

  size_t loadFrameInfo(const std::string &pcds_dir, const std::string &pose_file, const int start, const int end) override
  {
    frame_info_buf_.clear();

    int line_num = 0;
    std::ifstream ifs(pose_file);
    std::string csv_line;
    while (getline(ifs, csv_line))
    {
      int tmp = line_num;
      if(tmp >= start){
        if(tmp >= end && start < end)
          break;

        FrameInfo frame_info;
        frame_info.idx = line_num;

        Eigen::Affine3f lidar_pose = calib_inv_ * parseLine(csv_line) * calib_;
        frame_info.t = lidar_pose.translation();
        frame_info.r = Eigen::Quaternionf(lidar_pose.rotation());

        std::stringstream frame_filename_str;
        frame_filename_str << std::setfill('0') << std::right << std::setw(6) << line_num << ".bin";
        frame_info.pcd_filename = pcds_dir + frame_filename_str.str();
    
        frame_info_buf_.push_back(frame_info);
      }

      line_num++;
    }

    return frame_info_buf_.size();
  }

  void loadKittiCloud(const std::string &file_name, pcl::PointCloud<pcl::PointXYZI> &cloud)
  {
    cloud.clear();
    cloud.reserve(200000);
    std::fstream in(file_name, std::ios::in | std::ios::binary);
    while(in.good() && !in.eof()){
      pcl::PointXYZI p;
		  in.read((char *)&p.x, sizeof(float));
		  in.read((char *)&p.y, sizeof(float));
		  in.read((char *)&p.z, sizeof(float));
		  in.read((char *)&p.intensity, sizeof(float));
		  cloud.push_back(p);
  	}
	  in.close();
  }
  
  Frame loadFrame(const size_t idx) override
  {
    if(idx >= frame_info_buf_.size()){
      return Frame();
    }

    Frame frame;
    loadKittiCloud(frame_info_buf_[idx].pcd_filename, *frame.frame);
    frame.idx = frame_info_buf_[idx].idx;
    frame.t = frame_info_buf_[idx].t;
    frame.r = frame_info_buf_[idx].r;
    
    return frame;
  };
};

#endif