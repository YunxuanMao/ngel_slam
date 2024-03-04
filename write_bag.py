#! /usr/bin/env python
# coding=UTF-8

import os
import glob
from tqdm import tqdm
import numpy as np
import rospy
import rosbag
from sensor_msgs.msg import Image, CameraInfo, Imu
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray
from tf.msg import tfMessage
import cv2
from cv_bridge import CvBridge

def get_camera_info():
    H, W, fx, fy, cx, cy = 480 , 640, 577.590698, 578.729797, 318.905426, 242.683609
    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = fx
    camera_intrinsics[1, 1] = fy
    camera_intrinsics[0, 2] = cx
    camera_intrinsics[1, 2] = cy

    camera_info = CameraInfo()
    camera_info.height = H
    camera_info.width = W

    camera_info.distortion_model = "plumb_bob"
    camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    camera_info.R = np.ndarray.flatten(np.identity(3))
    camera_info.K = np.ndarray.flatten(camera_intrinsics[:, :3])
    camera_info.P = np.ndarray.flatten(camera_intrinsics)

    return camera_info

def get_images(color_path, depth_path):
    color = Image()
    depth = Image()
    
    color_data = cv2.imread(color_path)
    depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    H, W = depth_data.shape
    color_data = cv2.resize(color_data, (W, H))
    depth_data = depth_data.astype(np.float32)

    cvbridge = CvBridge()

    bgr_msg = cvbridge.cv2_to_imgmsg(color_data, "bgr8")
    depth_msg = cvbridge.cv2_to_imgmsg(depth_data, "32FC1")

    return bgr_msg, depth_msg
    
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description='Arguments for running.'
    )
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--frame_id', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')

    args = parser.parse_args()

    input_folder = args.input_folder

    rospy.init_node("w_bag_p", anonymous=True)  # 初始化节点

    bag = rosbag.Bag(args.output, 'w') # 创建rosbag对象并打开文件流

    color_paths = sorted(glob.glob(os.path.join(
            input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
    depth_paths = sorted(glob.glob(os.path.join(
            input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
    n_img = len(color_paths)
    print color_paths
    # print depth_paths

    camera_info = get_camera_info()

    header = Header(frame_id = args.frame_id)

    for i in tqdm(range(n_img)):
        #print(color_paths[i])
        timestamp = rospy.Time(i+0.5)
        header.stamp = timestamp
        color, depth = get_images(color_paths[i], depth_paths[i])
        color.header = header
        depth.header = header

        # 写数据
        bag.write("/camera/depth/camera_info", camera_info, timestamp)  # 话题
        bag.write("/camera/depth/image", depth, timestamp)
        bag.write("/camera/rgb/camera_info", camera_info, timestamp)
        bag.write("/camera/rgb/image_color", color, timestamp)

    # 关闭流
    bag.close()