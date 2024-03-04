import os
import yaml
import json
import time
import copy
import random
from multiprocessing import Manager

import numpy as np
import torch

from colorama import Fore, Style

from src.utils.datasets import get_dataset
from src.models.decoder import OCCDecoder, RGBDecoder
from orb_slam3_ros.msg import BA_info
import rospy



def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
    
def read_json_to_dict(json_path: str, ):
    with open(json_path, 'r') as f:
        dic = json.load(f)
        return dic


class KeyframeManager():

    def __init__(self, cfg, args, ssnerf):

        self.cfg = cfg
        self.args = args
        self.device = cfg['device']
        self.verbose = cfg['verbose']
        self.freq = cfg['freq']
        torch.cuda.set_device(self.device)

        # init dataloader
        self.frame_reader = ssnerf.frame_reader
        self.n_img = ssnerf.n_img

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = ssnerf.H, ssnerf.W, ssnerf.fx, ssnerf.fy, ssnerf.cx, ssnerf.cy

        self.input_folder = os.path.join(ssnerf.input_folder, 'keyframes')

        # global keyframe containers
        self.keyframe_list = ssnerf.shared_keyframe_list
        self.keyframe_poses = ssnerf.shared_keyframe_poses
        self.LBA_flag = ssnerf.shared_LBA_flag
        self.GBA_flag = ssnerf.shared_GBA_flag
        self.Loop_flag = ssnerf.shared_Loop_flag
        self.time_stamp = ssnerf.shared_time_stamp
        self.share_color_images = ssnerf.share_color_images
        self.share_depth_images = ssnerf.share_depth_images

        # local keyframe containers
        self.anchor_frames = ssnerf.shared_anchor_frames
        self.local_keyframe_list = ssnerf.shared_local_keyframe_list

        # BA informations
        self.countLBA = 0
        self.countGBA = 0
        self.countLoop = 0

        # sparse volume config
        self.map_length = ssnerf.map_length
        self.overlap_threshold = ssnerf.overlap_threshold

        # keys for mappers to update
        self.local_anchor = ssnerf.shared_local_anchor
        self.global_anchor = ssnerf.shared_global_anchor

        # images for mapper
        self.local_optimized_keyframes = ssnerf.shared_local_optimized_keyframes
        self.global_optimized_keyframes = ssnerf.shared_global_optimized_keyframes

        self.init_mapping = ssnerf.init_mapping
        self.final = ssnerf.final

        self.init = True
        self.in_loop = False

        self.self.new_volume_flag = ssnerf.shared_new_volume_flag
        self.end_local_map_flag = ssnerf.shared_end_local_map_flag
        self.local_volume_init = ssnerf.shared_local_volume_init
        self.local_mapping_flag = ssnerf.shared_local_mapping_flag
        self.global_mapping_flag = ssnerf.shared_global_mapping_flag
        self.gloal_mapping = ssnerf.shared_gloal_mapping
        self.init_decoder_flag = ssnerf.shared_init_decoder_flag

        self.ref_kf = {}

        # param for new local map
        self.new_volume_flag = False
        self.new_map = 0
        # param for loop close
        self.loop_close_flag = False

        self.manager = Manager()

        self.kf_sub = rospy.Subscriber("/orb_slam3/BA_info", BA_info, self.callback)

    


    def read_info(self, info):
        

        time_stamp = info['time']
        countLBA = info['countLBA']
        countGBA = info['countGBA']
        countLoop = info['countLoop']
        kf_list = info['KFlist']
        kf_poses = info['KFposes']

        LBA_flag = countLBA != self.countLBA
        GBA_flag = countGBA != self.countGBA
        Loop_flag = countLoop != self.countLoop

        self.LBA_flag.value = LBA_flag or GBA_flag      
        self.GBA_flag.value = GBA_flag        

        optimized_kf = []
        new_kf = []

        if self.init:
            print(f"Init with {kf_list}")
            new_kf = kf_list
            self.keyframe_list[:] = kf_list
            for kf in new_kf:
                self.keyframe_poses[kf] = kf_poses[str(kf)]
                ret = self.frame_reader[kf]
                self.share_color_images[kf] = ret['color']
                self.share_depth_images[kf] = ret['depth']

        if LBA_flag or GBA_flag:

            # wait for mapping
            if self.global_mapping_flag.value or self.local_mapping_flag.value:
                while True:
                    if not self.global_mapping_flag.value and not self.local_mapping_flag.value:
                        break
            # update keyframes

            # 1. get optimized keyframes and new keyframes
            for kf in kf_list:
                if random.random() <= 1 or kf_list.index(kf) == 0:
                    # optimized keyframes
                    if kf in self.keyframe_poses:
                        if self.keyframe_poses[kf] != kf_poses[str(kf)]:
                            optimized_kf.append(kf)
                    # new keyframes
                    else:
                        new_kf.append(kf)
                        ret = self.frame_reader[kf]
                        self.share_color_images[kf] = ret['color']
                        self.share_depth_images[kf] = ret['depth']
            
            # 2. update keyframe information
            self.keyframe_list.extend(new_kf)
            self.keyframe_poses.clear()
            for k,v in kf_poses.items():
                self.keyframe_poses[int(k)] = v

            # 3. update BA information
            self.countLBA = countLBA
            self.countGBA = countGBA
            self.countLoop = countLoop

            # 4. cull local keyframes
            self.local_kf_culling(kf_list)

        if self.verbose:
            if LBA_flag:
                # print(Fore.MAGENTA)
                print(f"\033[95mLocal BA #{countLBA} Done! Time: {time_stamp}\033[0m")
                print(f"\033[95mOptimized keyframes #{len(optimized_kf)}: {optimized_kf}\033[0m")
                print(f"\033[95mNew keyframes #{len(new_kf)}: {new_kf}\033[0m")
                # print(Style.RESET_ALL)
            if GBA_flag:
                # print(Fore.MAGENTA)
                print(f"\033[95mGlobal BA #{countGBA} Done! Time: {time_stamp}\033[0m")
                print(f"\033[95mOptimized keyframes #{len(optimized_kf)}: {optimized_kf}\033[0m")
                print(f"\033[95mNew keyframes #{len(new_kf)}: {new_kf}\033[0m")
                # print(Style.RESET_ALL)
            if Loop_flag:
                # print(Fore.MAGENTA)
                print(f'\033[95mLoop detect at {time_stamp}!\033[0m')
                # print(Style.RESET_ALL)

        return LBA_flag, GBA_flag, Loop_flag, optimized_kf, new_kf
    

    def local_kf_culling(self, kf_list):
        # delete redundent keyframes in local keyframe information
        for i in range(len(self.anchor_frames)):
            anchor = self.anchor_frames[i]

            local_keyframes = self.local_keyframe_list[anchor]
            for kf in local_keyframes:
                if kf not in kf_list:
                    local_keyframes.remove(kf)
            
            # if anchor was deleted, the first local keyframe will be the new anchor
            if anchor not in kf_list:

                del self.local_keyframe_list[anchor]

                anchor = local_keyframes[0]
                self.anchor_frames[i] = anchor
                self.anchor_frames.append(anchor)
                
            self.local_keyframe_list[anchor] = local_keyframes
    
    def callback(self, data):

        info = {}
        kf_list = []
        
        time_stamp = data.stamp.secs
        info['time'] = time_stamp
        info['countLBA'] = data.countLBA
        info['countGBA'] = data.countGBA
        info['countLoop'] = data.countLoop
        info['KFposes'] = {}

        kfs = data.KFposes
        for kf in kfs:
            frame_id = int(float(kf.header.frame_id))
            # print(time_stamp)

            px = kf.pose.position.x
            py = kf.pose.position.y
            pz = kf.pose.position.z

            ox = kf.pose.orientation.x
            oy = kf.pose.orientation.y
            oz = kf.pose.orientation.z
            ow = kf.pose.orientation.w

            kf_info = [px, py, pz, ox, oy, oz, ow]
            info['KFposes'][frame_id] = kf_info
            kf_list.append(frame_id)
            
        info['KFlist'] = kf_list
        LBA_flag, GBA_flag, Loop_flag, optimized_kf, new_kf = self.read_kf_json(info)
        
        
        if self.init:
            init_frame = new_kf[0]
            anchor_frame = init_frame
            last_ref_kf = init_frame
            
            self.local_anchor.value = init_frame
            self.new_volume_flag = True
            self.self.new_volume_flag.value = True
            self.local_volume_init.value = True
            self.init_mapping.value = True
            while True:
                if self.init_decoder_flag.value:
                    break


        if LBA_flag or GBA_flag: 

            # not in loop, update new local volume
            if not self.in_loop:
            
                # if a new local volume needed
                if len(optimized_kf) > 0:
                    if optimized_kf[0] - last_ref_kf > self.map_length:
                        if self.new_map == 0:
                            new_ref_kf = optimized_kf[0]
                        if optimized_kf[0] == new_ref_kf:
                            self.new_map += 1
                        else:
                            new_ref_kf = optimized_kf[0]
                            self.new_map = 1
                        # self.new_map += 1
                    else:
                        self.new_map = 0

                    if self.new_map == 3 and self.keyframe_list[-1] - anchor_frame >= self.map_length:
                        new_ref_kf = optimized_kf[0]
                        if self.verbose:
                            print('Local map stop updating!')
                        
                        self.end_local_map_flag.value = True
                        self.LBA_flag.value = False
                        self.GBA_flag.value = False 
                        

                        self.new_map = 0
                    

                        if not self.loop_close_flag:
                            last_ref_kf = new_ref_kf
                            self.ref_kf[anchor_frame].append(new_ref_kf)
                            self.new_volume_flag = True
                            self.self.new_volume_flag.value = True
                            # wait for mapping
                            self.local_volume_init.value = True
                            if len(new_kf) > 0:
                                anchor_frame = new_kf[0]
                            else:
                                anchor_frame = self.keyframe_list[-1]
                        
                        else:
                            self.in_loop = True
                            
                        
                        # wait for model saving
                        while True:
                            if not self.end_local_map_flag.value:
                                break
                        

                # don't need a new volume
                if not self.new_volume_flag:
                    if len(optimized_kf) > 0:
                        if len(self.ref_kf) > 1:
                            local_ref_kf = optimized_kf[0]
                            ref = False
                            for k, v in self.ref_kf.items():
                                if len(v) > 1:
                                    if local_ref_kf >= v[0] and local_ref_kf < v[1]:
                                        self.local_anchor.value = k
                                        ref = True
                            
                            if not ref:
                                self.local_anchor.value = self.anchor_frames[-1]


                    self.local_keyframe_list[self.local_anchor.value].extend(new_kf)
                    local_optimized_kf = sorted(list(set(self.local_keyframe_list[self.local_anchor.value]) & set(optimized_kf)))
                    if len(local_optimized_kf) > 0 or len(new_kf) > 0:
                        self.local_optimized_keyframes[:] = local_optimized_kf + new_kf

                    anchors = []
                    optimized_kf_anchors = []
                    num_optimized_kf_anchors = []
                    if len(self.anchor_frames) > 1:
                        for anchor in self.anchor_frames:
                            if anchor != self.local_anchor.value:
                                optimized_kf_anchor = sorted(list(set(self.local_keyframe_list[anchor]) & set(optimized_kf)))
                                if len(optimized_kf_anchor) > 0:
                                    anchors.append(anchor)
                                    optimized_kf_anchors.append(optimized_kf_anchor)
                                    num_optimized_kf_anchors.append(len(optimized_kf_anchor))
                                
                            if len(anchors) > 0:
                                anchor = random.choices(anchors, weights = num_optimized_kf_anchors, k=1)[0]
                                anchor_idx = anchors.index(anchor)
                                self.global_anchor.value = anchor
                                self.global_optimized_keyframes[:] = optimized_kf_anchors[anchor_idx]
                                self.gloal_mapping.value = True
                    
                
                # if self.verbose:
                #     print(Fore.MAGENTA)
                #     print(f"\033[95mLocal optimized keyframe: {self.local_optimized_keyframes}.\033[0m")
                    # print(Style.RESET_ALL)

        
            else:
                ref = False
                if len(self.ref_kf) > 1:
                    if len(optimized_kf) > 0:
                        local_ref_kf = optimized_kf[0]
                        ref = False
                        for k, v in self.ref_kf.items():
                            if len(v) > 1:
                                if local_ref_kf >= v[0] and local_ref_kf < v[1]:
                                    self.local_anchor.value = k
                                    ref = True
                    # else:
                    #     ref = True

                if ref:
                    self.local_keyframe_list[self.local_anchor.value].extend(new_kf)
                    local_optimized_kf = sorted(list(set(self.local_keyframe_list[self.local_anchor.value]) & set(optimized_kf)))
                    if len(local_optimized_kf) > 0 or len(new_kf) > 0:
                        self.local_optimized_keyframes[:] = local_optimized_kf + new_kf
                    
                    
                else:
                    optimized_kf_anchors = []
                    num_optimized_kf_anchors = []
                    for anchor in self.anchor_frames:
                        optimized_kf_anchor = sorted(list(set(self.local_keyframe_list[anchor]) & set(optimized_kf)))
                        optimized_kf_anchors.append(optimized_kf_anchor)
                        num_optimized_kf_anchors.append(len(optimized_kf_anchor))
                    if np.max(num_optimized_kf_anchors) > 0:
                        anchor_idxs = np.argsort(num_optimized_kf_anchors)
                        # local mapping
                        anchor_frame = self.anchor_frames[anchor_idxs[-1]]
                        optimized_kf_anchor = optimized_kf_anchors[anchor_idxs[-1]]
                        # optimized_kf_anchor = np.array(optimized_kf_anchors[anchor_idxs[-1]])
                        # optimized_kf_anchor = list(optimized_kf_anchor[optimized_kf_anchor >= last_ref_kf])
                        if len(optimized_kf_anchor + new_kf) > 0:
                            self.local_optimized_keyframes[:] = optimized_kf_anchor + new_kf
                        self.local_anchor.value = anchor_frame
                        self.local_keyframe_list[anchor_frame].extend(new_kf)
                        # global mapping
                        # if len(optimized_kf_anchors[anchor_idxs[-2]]) > 0:
                        #     anchor_frame = self.anchor_frames[anchor_idxs[-2]]
                        #     optimized_kf_anchor = optimized_kf_anchors[anchor_idxs[-2]]
                        #     self.global_optimized_keyframes[:] = optimized_kf_anchor
                        #     self.global_anchor.value = anchor_frame
                        
                        if np.max(num_optimized_kf_anchors) < self.overlap_threshold and np.max(num_optimized_kf_anchors) < len(optimized_kf):
                            self.end_local_map_flag.value = True
                            self.LBA_flag.value = False

                            self.GBA_flag.value = False 
                            self.new_volume_flag = True
                            self.self.new_volume_flag.value = True
                            # wait for mapping
                            self.local_volume_init.value = True
                            if len(new_kf)>0:
                                anchor_frame = new_kf[0]
                            else:
                                anchor_frame = self.keyframe_list[-1]
                            last_ref_kf = anchor_frame
                            self.in_loop = False
                            while True:
                                if not self.end_local_map_flag.value:
                                    break
            
            

        self.LBA_flag.value = False
        # self.GBA_flag.value = False 
        if not self.in_loop:      
            if Loop_flag:
                self.in_loop = True
                last_loop_close = self.keyframe_list[-1]
                self.loop_close_flag = True
                self.ref_kf[anchor_frame].append(last_loop_close)

        

        if self.new_volume_flag:
            print(f'New local map start! #{len(self.anchor_frames)}')
                            
            self.local_anchor.value = anchor_frame
            self.anchor_frames.append(anchor_frame)
            self.ref_kf[anchor_frame] = [last_ref_kf]
            if len(new_kf) > 0:
                self.local_keyframe_list[anchor_frame] = self.manager.list(new_kf)
                self.local_optimized_keyframes[:] = self.manager.list(new_kf)
            else:
                self.local_keyframe_list[anchor_frame] = self.manager.list([anchor_frame])
                self.local_optimized_keyframes[:] = self.manager.list([anchor_frame])
            self.new_volume_flag = False
            self.self.new_volume_flag.value = False
            
            if self.init:
                while True:
                    if not self.init_mapping.value:
                        if self.verbose:
                            print(Fore.MAGENTA)
                            print(f"Global initializing done!")
                            print(Style.RESET_ALL)
                        break
                self.init = False
                
            
            
            while True:
                if not self.local_volume_init.value:
                    if self.verbose:
                        print(Fore.MAGENTA)
                        print(f"Local initializing done!")
                        print(Style.RESET_ALL)
                    break

            
        # sleep
        time.sleep(1/self.freq)

        self.time_stamp.value += 1
            
    




