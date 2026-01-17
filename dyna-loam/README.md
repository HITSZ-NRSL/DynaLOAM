# DynaLOAM

## 1. 输入
```yaml
lidar_topic: /cloud_registered_body # lidar scan in lidar frame
initial_odom: /Odometry # initial odom info
```
## 2. 输出
```yaml
/lio_sam/dynamic_cloud # dynamic submap in map frame 
/lio_sam/static_cloud # static submap in map frame
/lio_sam/odomOptimized # odom optimized by scan-to-map
/lio_sam/scanFiltered # scan filtered by knn
```
## 3. 运行
```
roslaunch dynamic_removal dynamic_filter_online.launch
```
