#! /usr/bin/python

# save traj in tum format
import rospy
import tf
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

path_odom = '/media/eric/Elements SE/data/KITTI_raw/00/aloam_filter_odom.txt'
path_map = '/media/eric/Elements SE/data/KITTI_raw/05/aloam_map.txt'

# first lidar stamp
t0 = 0

def mapCallback(msg: Path):
    global t0
    if t0 == 0:
        print("no ready")
        return
    print("receive msg")
    with open(path_map, 'w') as file:
        for i in msg.poses:
            file.write(str(rospy.Time.to_sec(i.header.stamp) - t0) + ' ')
            file.write(str(i.pose.position.x) + ' ')
            file.write(str(i.pose.position.y) + ' ')
            file.write(str(i.pose.position.z) + ' ')
            file.write(str(i.pose.orientation.x) + ' ')
            file.write(str(i.pose.orientation.y) + ' ')
            file.write(str(i.pose.orientation.z) + ' ')
            file.write(str(i.pose.orientation.w) + '\n')       

def odomCallback(msg: Path):
    global t0
    if t0 == 0:
        print("no ready")
        return
    print("receive msg")
    with open(path_odom, 'w') as file:
        for i in msg.poses:
            file.write(str(rospy.Time.to_sec(i.header.stamp) - t0) + ' ')
            file.write(str(i.pose.position.x) + ' ')
            file.write(str(i.pose.position.y) + ' ')
            file.write(str(i.pose.position.z) + ' ')
            file.write(str(i.pose.orientation.x) + ' ')
            file.write(str(i.pose.orientation.y) + ' ')
            file.write(str(i.pose.orientation.z) + ' ')
            file.write(str(i.pose.orientation.w) + '\n')  

def cloudCallback(msg: PointCloud2):
    global t0
    if t0 == 0:
        t0 = rospy.Time.to_sec(msg.header.stamp)
    else:
        return

if __name__ == "__main__":
    rospy.init_node("save_traj")
    print("save traj node start")
    print("save path: ")
    print(path_odom)
    # print(path_map)
    # sub = rospy.Subscriber("/aft_mapped_path", Path, mapCallback, queue_size=100)
    sub = rospy.Subscriber("/laser_odom_path", Path, odomCallback, queue_size=100)
    subLidar = rospy.Subscriber("/points_raw", PointCloud2, cloudCallback, queue_size=100)
    rospy.spin()
