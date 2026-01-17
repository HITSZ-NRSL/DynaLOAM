import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import math

from functools import reduce
 
def str2int(s):
    def fn(x,y):
        return x*10+y
    def char2num(s):
        return {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]
    return reduce(fn,map(char2num,s))

sys.path.append("..")

dataset = "kitti"
seq = "05"
N = 0

#read kitti pose
pose_dir = "/media/eric/Elements SE/data/kitti_odom/dataset/poses/"

traj = np.loadtxt(pose_dir + seq + ".txt")

bow3d_res = np.loadtxt("./" + dataset + "/bow3d/" +seq+".txt")

gt = {}

cur_len = 0

for line in open("./" + dataset + "/gt/" +seq+".txt", "r"):
    if cur_len > 1500:
        break
    cur_len += 1
    if line.strip():
        sl = line.split()
        if len(sl) >= 2:
            if str2int(sl[0]) - str2int(sl[1]) > 300:
                gt[sl[0]] = 1
                N = N+1
            else:
                gt[sl[0]] = 0
        else:
            gt[sl[0]] = 0
print(N)

pre = 1
rec = len(bow3d_res) / N
F1 = 2 * (pre *rec) / (pre + rec)

print("pre: ", pre)
print("rec: ", rec)
print("F1: ", F1)


x_cord = traj[:,3]
z_cord = traj[:,11]

fig2 = plt.figure(2)
plt.title('trajectory',fontsize=20)# give plot a title
plt.xlabel('x', fontsize=20)# make axis labels
plt.ylabel('z',fontsize=20)
plt.tick_params(labelsize=18)
plt.plot(x_cord, z_cord,  "k", linewidth=1.0)

for i in range(len(bow3d_res[:,0])):
    if gt[str(int(bow3d_res[i][0]))]:
        index = int(int(bow3d_res[i][0]))
        # plt.scatter(x_cord[index], z_cord[index], c="g",alpha=0.2)
    if bow3d_res[i][2] == 1:
        index = int(bow3d_res[i][0])
        plt.scatter(x_cord[index], z_cord[index], c="r")
    if bow3d_res[i][2] == 0:
        index = int(bow3d_res[i][0])
        plt.scatter(x_cord[index], z_cord[index], c="b")

plt.show()
