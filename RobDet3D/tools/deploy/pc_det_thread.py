# !/home/st/ubuntu_data/software/miniconda3/envs/ciassd/bin/python
#coding=utf-8 
import ctypes
import os
import sys
import time
from collections import deque
from logging import raiseExceptions

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pycuda.driver as cuda
import pycuda.autoinit 
import numpy as np
import numpy.ctypeslib as npct
import torch

import tensorrt as trt 
from utils import load_plugins 
from sensor_msgs.point_cloud2 import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header 
import rospy

# from det_tra_pre.utils import change2master, change2world, distanceCal 

libLoad = ctypes.cdll.LoadLibrary
sharelib = libLoad("/home/ou/disk/workspace/pc_dynamic_slam-main-src/devel/lib/libpython_share_memory.so")
model = "/home/ou/disk/workspace/oyjy/pc_dynamic_slam-main-src/src/point_detection/config/iassd_hvcsx2_gqx2_4x2_80e_peds(export_fp16).engine"

ros_topic_out = "objects"
# score_threshold=0.6
score_threshold=0.4

pc_flag_ = ctypes.c_int(100)
p_pc_flag_ = ctypes.pointer(pc_flag_)

pc_array_ = np.zeros(shape=(1000*200,4), dtype=np.float32)
pc_size_ = np.zeros(shape=(1), dtype=np.int32)
sharelib.get_pc_array.argtypes = [
    npct.ndpointer(dtype=np.float32, ndim=2, shape =pc_array_.shape),
    npct.ndpointer(dtype=np.int32, ndim=1)
]
 
class MarkerPublisher:
    def __init__(self):
        self.pub = rospy.Publisher(ros_topic_out, MarkerArray, queue_size=1)
    
    def boxes_to_corners_3d(self,boxes3d):
        """
            7 -------- 4
        /|         /|
        6 -------- 5 .
        | |        | |
        . 3 -------- 0
        |/         |/
        2 -------- 1
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        Returns:
        """
        def check_numpy_to_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float(), True
            return x, False
        def rotate_points_along_z(points, angle):
            """
            Args:
                points: (B, N, 3 + C)
                angle: (B), angle along z-axis, angle increases x ==> y
            Returns:

            """
            points, is_numpy = check_numpy_to_torch(points)
            angle, _ = check_numpy_to_torch(angle)

            cosa = torch.cos(angle)
            sina = torch.sin(angle)
            zeros = angle.new_zeros(points.shape[0])
            ones = angle.new_ones(points.shape[0])
            rot_matrix = torch.stack((
                cosa,  sina, zeros,
                -sina, cosa, zeros,
                zeros, zeros, ones
            ), dim=1).view(-1, 3, 3).float()
            points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
            points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
            return points_rot.numpy() if is_numpy else points_rot
            
        boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]

        return corners3d.numpy() if is_numpy else corners3d

    def boxes2markers(self, header, boxes):
        def get_default_marker(action, ns):
            marker = Marker()
            marker.header = header
            marker.type = marker.LINE_LIST
            marker.action = action

            marker.ns = ns
            # marker scale (scale y and z not used due to being linelist)
            marker.scale.x = 0.08
            # marker color
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0

            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.points = []
            return marker

        def get_boxes_corners_line_list():
            corners3d = self.boxes_to_corners_3d(boxes[:, :7])  # (N,8,3)
            corner_for_box_list = [0, 1, 0, 3, 2, 3, 2, 1, 4, 5, 4, 7, 6, 7, 6, 5, 3, 7, 0, 4, 1, 5, 2, 6, 0, 5, 1, 4]
            boxes_corners_line_list = corners3d[:, corner_for_box_list, :3]
            return boxes_corners_line_list

        markers = MarkerArray()
        corners = get_boxes_corners_line_list()
        for i, box in enumerate(boxes):
            marker = get_default_marker(Marker.ADD, 'Pedestrian')
            marker.id = i
            markers.markers.append(marker)
            for box_corner in corners[i]:
                marker.points.append(Point(box_corner[0], box_corner[1], box_corner[2]))

        if not hasattr(self, "last_markers"):
            self.last_markers = []
        if len(markers.markers) < len(self.last_markers):
            for last_marker in self.last_markers[len(markers.markers):]:
                last_marker.action = Marker.DELETE
                markers.markers.append(last_marker)
        self.last_markers = markers.markers
        return markers
    
    def pub_boxes(self, boxes):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "os_sensor"
        markers = self.boxes2markers(header, boxes)
        self.pub.publish(markers)

pub = MarkerPublisher()
class Processor_ROS:
    def __init__(self, model_path):
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
        with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.ctx.set_optimization_profile_async(0, self.stream.handle)
        self.input_shape = self.ctx.get_binding_shape(self.engine.get_binding_index("points"))
        self.input_shape[0] = 1
        self.ctx.set_binding_shape(self.engine.get_binding_index("points"), self.input_shape)
        self.output_shape = self.ctx.get_binding_shape(self.engine.get_binding_index("boxes"))[1]
        self.pc =  np.zeros([i for i in self.input_shape],dtype=np.float32)
        self.h_inputs = {'points': self.pc}
        self.d_inputs = {}
        self.h_outputs = {}
        self.d_outputs = {} 
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.d_inputs[binding] = cuda.mem_alloc(self.h_inputs[binding].nbytes)
            else:
                output_shape = self.ctx.get_binding_shape(self.engine.get_binding_index(binding))
                size = trt.volume(output_shape)
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                self.h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                self.d_outputs[binding] = cuda.mem_alloc(self.h_outputs[binding].nbytes)
        assert self.ctx.all_binding_shapes_specified
        

    def run_detection(self, pc):
        self.pc[:]=0
        try:
            sampled_indices = np.random.choice(np.arange(0, len(pc), dtype=np.int32), self.pc.shape[1], replace=False)
        except: 
            sampled_indices = np.random.choice(np.arange(0, len(pc), dtype=np.int32), self.pc.shape[1], replace=True)
        self.pc[0,:] = pc[sampled_indices]
        # min_size = min(self.pc.shape[1], pc.shape[0])
        # self.pc[0,:min_size]=pc[:min_size]
        
        self.h_inputs = {'points': self.pc}

        for key in self.h_inputs:
            cuda.memcpy_htod_async(self.d_inputs[key], self.h_inputs[key], self.stream)
            
        self.ctx.execute_async_v2(
            bindings=[int(self.d_inputs[k]) for k in self.d_inputs] + [int(self.d_outputs[k]) for k in self.d_outputs],
            stream_handle=self.stream.handle
        )

        for key in self.h_outputs:
            cuda.memcpy_dtoh_async(self.h_outputs[key], self.d_outputs[key], self.stream)
        self.stream.synchronize()
        num = self.h_outputs['nums'].reshape(-1)[0]
        boxes = self.h_outputs['boxes'].reshape(self.output_shape, -1)
        scores = self.h_outputs['scores'].reshape(self.output_shape)

        mask = scores[:num]>score_threshold
        boxes = boxes[:num,:7][mask]
        center = np.asarray(boxes[:,:3], dtype=np.float32)
        pub.pub_boxes(boxes)
        # return boxes
        return center

def get_pc_array():
    sharelib.get_pc_array(pc_array_,pc_size_)
    return pc_array_[0:pc_size_[0],:]

def get_pc_flag():  ##使用return int 会出错
    sharelib.get_pc_flag(p_pc_flag_)
    return pc_flag_.value

def send_pc_pre_array(pc_pre):
    if not pc_pre.flags['C_CONTIGUOUS']:
        pc_pre = np.ascontiguousarray(pc_pre, dtype=pc_pre.dtype)  # 如果不是C连续的内存，必须强制转换
    p_pc_pre = ctypes.cast(pc_pre.ctypes.data, ctypes.POINTER(ctypes.c_float))   #转换为ctypes，这里转换后的可以直接利用ctypes转换为c语言中的int*，然后在c中使用
    sharelib.send_pc_pre_array(p_pc_pre, pc_pre.shape[0])

def send_pc_box_array(pc_pre):
    if not pc_pre.flags['C_CONTIGUOUS']:
        pc_pre = np.ascontiguousarray(pc_pre, dtype=pc_pre.dtype)  # 如果不是C连续的内存，必须强制转换
    p_pc_pre = ctypes.cast(pc_pre.ctypes.data, ctypes.POINTER(ctypes.c_float))   #转换为ctypes，这里转换后的可以直接利用ctypes转换为c语言中的int*，然后在c中使用
    sharelib.send_pc_box_array(p_pc_pre, pc_pre.shape[0])

epoch_times = []
if __name__ == "__main__":
    rospy.init_node('iassd')
    proc = Processor_ROS(model)
    print("cia-ssd launch!")
    epoch=0
    while not rospy.is_shutdown():
        time.sleep(0.001)
        if get_pc_flag()==1 :
            t0 = time.time()
            pc = get_pc_array()
            # print("data ",proc.run_detection(pc))
            # box_part = proc.run_detection(pc)
            # center_part = np.asarray(box_part[:,:3], dtype=np.float32)
            send_pc_pre_array(proc.run_detection(pc))
            # send_pc_box_array(box_part)
            torch.cuda.empty_cache()
            t1= time.time() - t0
            print("time ", t1)
            epoch=epoch+1
            if epoch > 5 :
                epoch_times.append(t1)  # add the epoch time to the list
                avrage_time = sum(epoch_times) / len(epoch_times)  # calculate the average epoch time
                print("avrage time ", avrage_time)