#! /usr/bin/python

f1 = open("/media/eric/Elements SE/data/KITTI_raw/09/aloam_odom.txt", "r")
f2 = open("/media/eric/Elements SE/data/KITTI_raw/09/times.txt", "w")

line = f1.readline()
count = 1
while line:
    line1 = line[:-1].split(' ')
    if count >= 1101:
        f2.write(line1[0])
        f2.write("\n")
    line = f1.readline()
    count += 1
f1.close()
f2.close()