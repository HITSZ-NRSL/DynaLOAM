# [Autonomous Robots 2025] DynaLOAM: Robust LiDAR Odometry and Mapping in Dynamic Environments.
## Affiliation: Networked Robotics and Systems Lab, HITSZ

## Introduction 
Simultaneous localization and mapping (SLAM) based on LiDAR in dynamic environments remains a challenging problem due to unreliable data association and residual ghost tracks in the map. In recent years, some related works have attempted to utilize semantic information or geometric constraints between consecutive frames to reject dynamic objects as outliers. However, challenges persist, including poor real-time performance, heavy reliance on meticulously annotated datasets, and susceptibility to misclassifying static points as dynamic. This paper presents a novel dynamic LiDAR SLAM framework called DynaLOAM, in which a complementary dynamic interference suppression scheme is exploited. For accurate relative pose estimation, a lightweight detector is proposed to rapidly respond to pre-defined dynamic object classes in the LiDAR FOV and eliminate correspondences from dynamic landmarks. Then, an online submap cleaning method based on visibility and clustering is proposed for real-time dynamic object removal in submap, which is further utilized for pose optimization and global static map construction. By integrating the complementary characteristics of prior appearance detection and online visibility check, DynaLOAM can finally achieve accurate pose estimation and static map construction in dynamic environments. Extensive experiments are conducted on the KITTI dataset and three real scenarios. The results show that our approach achieves promising performance compared to state-of-the-art methods.

## Usage
### Object detection
Refer to the README in RobDet3D to compile the model engine.
Run src/RobDet3D/tools/deploy/pc_det_thread.py for detection. You need to modify the paths of sharelib and model in it.
### LiDAR Odometry

# start ia-ssd Detector
conda activate pt_iassd  &&
python ../pc_det_thread.py

# start Lidar Odometry
roslaunch fast_lio mapping_horizon.launch

# Datasets

## Reference
If you think this work is meaningful, please cite:
```bash
@article{wang2025dynaloam,
  title={DynaLOAM: robust LiDAR odometry and mapping in dynamic environments},
  author={Wang, Yu and Lyu, Ruichen and Ouyang, Junyuan and Wang, Zhihao and Xie, Xiaochen and Chen, Haoyao},
  journal={Autonomous Robots},
  volume={49},
  number={4},
  pages={29},
  year={2025},
  publisher={Springer}
}

@ARTICLE{9939009,
  author={Ouyang, Junyuan and Chen, Haoyao},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Det6D: A Ground-Aware Full-Pose 3-D Object Detector for Improving Terrain Robustness}, 
  year={2022},
  volume={71},
  number={},
  pages={1-9},
  keywords={Three-dimensional displays;Feature extraction;Task analysis;Point cloud compression;Object detection;Pose estimation;Detectors;3-D object detection;autonomous driving;complex terrain;point cloud},
  doi={10.1109/TIM.2022.3219469}}




