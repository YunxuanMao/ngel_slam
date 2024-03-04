import os
from multiprocessing import Manager
import time
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from src.models.decoder import OCCDecoder, RGBDecoder
from src.models.renderer import Renderer
from src.utils.visualizer import Visualizer
from src.keyframeManager import KeyframeManager
from src.mapper import Mapper
from src.global_mapper import Global_Mapper
from src.utils.datasets import get_dataset
import rospy

LOSS_TYPE = ['depth_occ', 'rgb']

class NGEL_SLAM():

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args
        self.device = cfg['device']
        self.verbose = cfg['verbose']
        torch.cuda.set_device(self.device)
        self.dilate = cfg['dilate']

        # param of octree grid
        self.feature_dim = cfg['models']['octree']['feature_dim']
        self.base_lod = cfg['models']['octree']['base_lod'] 
        self.num_lods = cfg['models']['octree']['num_lods'] 
        self.interpolation_type = cfg['models']['octree']['interpolation_type']
        self.multiscale_type = cfg['models']['octree']['multiscale_type']
        self.feature_std = cfg['models']['octree']['feature_std']
        self.feature_bias = cfg['models']['octree']['feature_bias']

        # param of decoders
        self.occ_dim = cfg['models']['occ_dim']
        self.rgb_dim = cfg['models']['color_dim']
        self.rgb_start = cfg['models']['color_start']

        # init dataloader
        self.frame_reader = get_dataset(cfg, args, 1, self.device)
        self.n_img = len(self.frame_reader)

        # volume bound
        self.bbox = cfg['mapping']['bound']
        origin = (np.array(self.bbox)[:,1] + np.array(self.bbox)[:,0]) / 2
        scale = np.ceil((np.array(self.bbox)[:,1] - np.array(self.bbox)[:,0]).max()/2)
        self.origin = torch.tensor(origin, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)

        # sparse volume config
        self.map_length = cfg['map_length']
        self.overlap_threshold = cfg['overlap_threshold']
        self.smooth = cfg['mapping']['smooth']

        self.pos_embed = cfg['pos_embed']

        # camera intrinsic
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        # data loading and saving
        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder
        self.output_folder = os.path.join(cfg['data']['output'] + '_ORB_cl', f'{self.base_lod}_{self.num_lods}_{self.map_length}_{self.overlap_threshold}')
        
        if self.smooth:
            self.output_folder += f'_smooth_{cfg["mapping"]["smooth_length"]}'
        os.makedirs(self.output_folder, exist_ok=True)
        print(f'Output folder: {self.output_folder}')
        
        self.ckpt_output_folder = os.path.join(self.output_folder, 'ckpt')
        os.makedirs(self.ckpt_output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder,'config.yaml'), 'w') as file:
            file.write(yaml.dump(self.cfg, allow_unicode=True))

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # shared data python
        manager = Manager()

        # global keyframes infos
        self.shared_keyframe_list = manager.list()
        self.shared_keyframe_poses = manager.dict()
        self.shared_LBA_flag = manager.Value('b', False)
        self.shared_GBA_flag = manager.Value('b', False)
        self.shared_Loop_flag = manager.Value('b', False)
        self.shared_new_volume_flag = manager.Value('b', False)
        self.shared_end_local_map_flag = manager.Value('b', False)
        self.shared_global_mapping_flag = manager.Value('b', False)
        self.shared_local_mapping_flag = manager.Value('b', False)
        self.shared_gloal_mapping = manager.Value('b', False)
        self.shared_local_volume_init = manager.Value('b', False)
        self.shared_init_decoder_flag = manager.Value('b', False)
        self.shared_time_stamp = manager.Value('i', 0)
        self.init_mapping = manager.Value('b', False)
        self.final = manager.Value('b', False)
        # local keyframes infos
        self.shared_anchor_frames = manager.list()
        self.shared_local_keyframe_list = manager.dict()
        # keys for mapper
        self.shared_local_anchor = manager.Value('i', 0)
        self.shared_global_anchor = manager.Value('i', 0)
        # images for mapper
        self.shared_local_optimized_keyframes = manager.list()
        self.shared_global_optimized_keyframes = manager.list()
        # keyframes
        self.shared_keyframe_rays = manager.dict()

        self.share_color_images = manager.dict()
        self.share_depth_images = manager.dict()

        # shared data torch
        # models
        # self.init_decoders()
        # self.occ_decoder.share_memory()
        # self.rgb_decoder.share_memory()
        
        

        

        self.local_mapper = Mapper(cfg, args, self)

        self.global_mapper = Global_Mapper(cfg, args, self)


    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here]
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def init_decoders(self):
        
        self.occ_decoder = OCCDecoder(self.cfg['models']["occ_decoder"], occ_feat_dim=self.occ_dim).to(self.device)
        self.rgb_decoder = RGBDecoder(self.cfg['models']["rgb_decoder"], self.rgb_dim).to(self.device) 
    

    def tracking(self, rank):
        rospy.init_node('kf_listener', anonymous=True)

        KeyframeManager(self.cfg, self.args, self)

        rospy.spin()

    def local_mapping(self, rank):

        self.local_mapper.run()

    def global_mapping(self, rank):

        self.global_mapper.run()

    def run(self):
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))

            if rank == 1:
                p = mp.Process(target=self.local_mapping, args=(rank, ))

            if rank == 2:
                p = mp.Process(target=self.global_mapping, args=(rank, ))


            p.start()
            processes.append(p)

        for p in processes:
            p.join()


    def load_model(self, models_path):
        # p = mp.Process(target=self.zombie, args=(models_path, ))
        # p.start()
        # p.join()
        models = torch.load(models_path)
        if self.pos_embed:
            self.embedders = models['embedder']
        else:
            self.embedders = None
        self.anchor_frames, self.world_std_c2w, self.grids, self.occ_decoders, self.rgb_decoders, self.anchor_std_c2w = models['anchors'], models['world_std_c2w'], models['grids'], models['occ_decoders'], models['rgb_decoders'], models['anchor_std_c2w']
        
        self.renderer = Renderer(self.cfg, self.args, self)
        self.visualizer = Visualizer(self.cfg, self.args, self)

        self.visualizer.world_std_c2w = self.world_std_c2w
        self.visualizer.anchor_std_c2w = self.anchor_std_c2w

    def render_img(self, frame_idx, pose_file_idx, vis = True, vis_every = False, fuse_method = 'max', gt_pose = True):
        return self.visualizer.render_img(frame_idx, self.anchor_frames, pose_file_idx, self.grids, self.embedders, self.occ_decoders, self.rgb_decoders, vis = vis, vis_every = vis_every, fuse_method=fuse_method, gt_pose=gt_pose)
    

    def extract_mesh(self, idx, pose_file_idx, reso = 512,vis_every = True, clean = True):

        return self.visualizer.extract_mesh(idx, self.anchor_frames, pose_file_idx, reso, self.grids, self.embedders, self.occ_decoders, self.rgb_decoders, vis_every=vis_every, clean = clean)