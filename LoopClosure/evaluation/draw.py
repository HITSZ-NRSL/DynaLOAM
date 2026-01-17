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
method = "iris"
filtered = False
N = 0

#read kitti pose
pose_dir = "/media/eric/Elements SE/data/kitti_odom/dataset/poses/"

traj = np.loadtxt(pose_dir + seq + ".txt")

if filtered :
    lc_res = np.loadtxt("./" + dataset + "/" + method +  "/" + seq + "_filtered.txt")
else:
    lc_res = np.loadtxt("./" + dataset + "/" + method +  "/"  +seq + ".txt")
gt = {}
for line in open("./" + dataset + "/gt/" +seq+".txt", "r"):
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

x_cord = traj[:,3]
z_cord = traj[:,11]

# #dist 0
th0 = []
pre0 = []
rec0 = []
lc_min = lc_res[:, 2].min()
lc_max = lc_res[:, 2].max()

print(lc_min, lc_max)
for i in np.arange(lc_min, lc_max + (lc_max-lc_min) * 1.0 /50, (lc_max-lc_min) * 1.0 /50):
    print(i)
    tp = 0
    p = 0
    for j in range(0, lc_res.shape[0]):
        if lc_res[j][2] <= i:
            p = p+1
            if lc_res[j][3] == 1.0:
                tp = tp+1
    
    re = tp * 1.0 / N
    pr = tp * 1.0 / p
    th0.append(i)
    rec0.append(re)
    pre0.append(pr)

thres = 100
F1 = 0.0
pre = 0
rec = 0
for i in range(len(th0)-1):
    print([th0[i],pre0[i],rec0[i]])
    if pre0[i]==1.0 and pre0[i+1]!=1.0:
        thres = th0[i]
        pre = pre0[i]
        rec = rec0[i]
        F1 = 2 * (pre0[i] * rec0[i]) / (pre0[i] + rec0[i])
        break 
if thres == 100:
    pre = pre0[-1]
    rec = rec0[-1]
    F1 = 2 * (pre * rec) / (pre + rec)

##draw p-r curve
#coding:utf-8
fig1 = plt.figure(1) # create figure 1
plt.title('Precision/Recall Curve',fontsize=20)# give plot a title
plt.xlabel('Recall', fontsize=20)# make axis labels
plt.ylabel('Precision',fontsize=20)
plt.tick_params(labelsize=18)
plt.plot(rec0, pre0,  "r", label = "LiDAR lc", linewidth=3.0)
plt.legend(loc="lower left", fontsize=20)


fig2 = plt.figure(2)
plt.title(method + " result",fontsize=10)# give plot a title
plt.xlabel('x', fontsize=20)# make axis labels
plt.ylabel('z',fontsize=20)
plt.tick_params(labelsize=18)
plt.plot(x_cord, z_cord,  "k", linewidth=1.0)


for i in range(len(lc_res[:,0])):
    if gt[str(int(lc_res[i][0]))]:
        index = int(int(lc_res[i][0]))
        # plt.scatter(x_cord[index], z_cord[index], c="g",alpha=0.2)
    if lc_res[i][2] <= thres and lc_res[i][3] == 1:
        index = int(lc_res[i][0])
        plt.scatter(x_cord[index], z_cord[index], c="r")
    # if lc_res[i][2] <= thres and lc_res[i][3] == 0:
    #     index = int(lc_res[i][0])
    #     plt.scatter(x_cord[index], z_cord[index], c="b")

plt.show()
print("pre: " , pre)
print("rec: ", rec)
print("F1: ", F1)
