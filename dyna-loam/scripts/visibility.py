#! /usr/bin/python
import cv2
img = cv2.imread('/home/eric/a_ros_ws/dyna_loam_ws/src/dyna-loam/img/map.jpg')

# cv2.imshow('map', mat)
print('waitting ........')
print(img.shape)
height = img.shape[0]        
width = img.shape[1]
channels = img.shape[2]

x = 0
y = 32
w = 900
h = 96
cropped_image = img[y:y+h, x:x+w]
cv2.imshow('img', cropped_image)
cv2.imwrite('/home/eric/a_ros_ws/dyna_loam_ws/src/dyna-loam/img/map_vis.jpg', cropped_image)
cv2.waitKey(0)


