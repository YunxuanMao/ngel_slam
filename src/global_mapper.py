import os
import random
import time
import copy
import threading
import torch.multiprocessing as mp

import numpy as np

import torch
from torch.autograd import Variable

from wisp.core import Rays
from wisp.models.grids import OctreeGrid
from wisp.models.embedders import get_positional_embedder
from wisp.models.decoders import BasicDecoder

from colorama import Fore, Style

from src.common import get_samples, get_rays_all, get_pc_from_depth, qp2tm, crop_pc, get_points_from_depth

from src.models.decoder_nice import MLP
from src.models.decoder import OCCDecoder, RGBDecoder
from src.models.renderer import Renderer
from src.utils.visualizer import Visualizer

LOSS_TYPE = ['depth_occ', 'rgb', 'smooth', 'unc']

class Global_Mapper():

    def __init__(self, cfg, args, ssnerf):

        self.cfg = cfg
        self.args = args
        self.device = cfg['device_global']
        self.verbose = cfg['verbose']


        torch.cuda.set_device(self.device)
        self.dilate = cfg['dilate']
        self.bbox = cfg['mapping']['bound']

        # param of octree grid
        self.feature_dim = cfg['models_global']['octree']['feature_dim']
        self.base_lod = cfg['models_global']['octree']['base_lod'] 
        self.num_lods = cfg['models_global']['octree']['num_lods'] 
        self.interpolation_type = cfg['models_global']['octree']['interpolation_type']
        self.multiscale_type = cfg['models_global']['octree']['multiscale_type']
        self.feature_std = cfg['models_global']['octree']['feature_std']
        self.feature_bias = cfg['models_global']['octree']['feature_bias']

        # param of decoders
        self.occ_dim = cfg['models_global']['occ_dim']
        self.rgb_dim = cfg['models_global']['color_dim']
        self.rgb_start = cfg['models_global']['color_start']

        # mapping config
        self.pix_per_frame = cfg['mapping']['ray_sample'] # for training
        self.pix_per_keyframe = cfg['mapping']['ray_sample_keyframe_select'] # for keyframe selection
        self.keyframe_num = cfg['mapping']['keyframe_num']
        self.iters = cfg['mapping']['iters']
        self.init_iters = cfg['mapping']['init_iters']

        # learning rate
        self.grid_lr = cfg['mapping']['grid_lr']
        self.occ_lr = cfg['mapping']['occ_lr']
        self.rgb_lr = cfg['mapping']['rgb_lr']

        # visualize frequency
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.vis_freq = cfg['mapping']['vis_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq'] 

        self.pos_embed = cfg['pos_embed']
        self.pos_multires = cfg['embedder']['freq']

        # keyframes config
        self.ray_option = cfg['keyframes']['option']
        self.num_rays_to_save = cfg['keyframes']['num_rays_to_save']
        self.depth_trunc = cfg["keyframes"]["depth_trunc"]

        # mesh config
        self.mesh_reso = cfg['meshing']['reso']
        self.clean_mesh = cfg['meshing']['clean']

        # global mapper config
        self.iters_out = cfg['global_mapper']['iters_out']
        self.iters_in = cfg['global_mapper']['iters_in']

        self.lamudas = {}
        for t in LOSS_TYPE:
            self.lamudas[t] = cfg['mapping']['lamudas'][t]
        
        self.input_folder = ssnerf.input_folder
        self.output_folder = ssnerf.output_folder
        
        self.ckpt_output_folder = os.path.join(self.output_folder, 'ckpt')
        os.makedirs(self.ckpt_output_folder, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = ssnerf.H, ssnerf.W, ssnerf.fx, ssnerf.fy, ssnerf.cx, ssnerf.cy
        self.origin = ssnerf.origin
        self.scale = ssnerf.scale

        

        # init dataloader
        self.frame_reader = ssnerf.frame_reader
        self.n_img = ssnerf.n_img

        self.grids = {}
        self.embedders = {}
        self.occ_decoders = {}
        self.rgb_decoders = {}

        self.LBA_flag = ssnerf.shared_LBA_flag
        self.GBA_flag = ssnerf.shared_GBA_flag
        
        self.keyframe_list = ssnerf.shared_keyframe_list
        self.keyframe_poses = ssnerf.shared_keyframe_poses

        self.anchor_frames = ssnerf.shared_anchor_frames
        self.local_keyframe_list = ssnerf.shared_local_keyframe_list

        self.share_color_images = ssnerf.share_color_images
        self.share_depth_images = ssnerf.share_depth_images

        # keys for mappers to update
        self.local_anchor = ssnerf.shared_local_anchor
        self.global_anchor = ssnerf.shared_global_anchor

        # images for mapper
        self.local_optimized_keyframes = ssnerf.shared_local_optimized_keyframes
        self.global_optimized_keyframes = ssnerf.shared_global_optimized_keyframes

        self.init_mapping = ssnerf.init_mapping
        self.final = ssnerf.final
        self.init = True
        self.i = 0
        self.time_stamp = ssnerf.shared_time_stamp

        self.keyframe_rays = ssnerf.shared_keyframe_rays
        # self.global_keyframe_images = {}

        self.anchor_std_c2w = {}

        self.keyframeManager = ssnerf.keyframeManager

        self.new_volume_flag = ssnerf.shared_new_volume_flag
        self.end_local_map_flag = ssnerf.shared_end_local_map_flag
        self.local_mapping_flag = ssnerf.shared_local_mapping_flag
        self.global_mapping_flag = ssnerf.shared_global_mapping_flag
        self.gloal_mapping = ssnerf.shared_gloal_mapping
        self.local_volume_init = ssnerf.shared_local_volume_init
        self.init_decoder_flag = ssnerf.shared_init_decoder_flag

        self.meshing_flag = False
        self.init = True

        self.renderer = Renderer(cfg, args, self)
        self.visualizer = Visualizer(cfg, args, self)

    def init_model(self):
        world_points = []
        keyframe_list = copy.deepcopy(self.keyframe_list)
        keyframe_rays = copy.deepcopy(self.keyframe_rays)
        keyframe_poses = copy.deepcopy(self.keyframe_poses)
        rays_all = []
        for frame in keyframe_list:
            rays = keyframe_rays[frame]
            depth = rays[:, -1]

            pose = keyframe_poses[frame]
            trans = np.array(pose[:3])
            quat = np.array(pose[3:])
            c2w = qp2tm(quat, trans)
            c2w_i = self.world_std_c2w.to(c2w) @ c2w

            points_i = get_points_from_depth(depth, c2w_i, self.fx, self.fy, self.cx, self.cy, dmax = self.depth_trunc)
            world_points.append(points_i)

            rays = self.get_rays(rays, c2w_i)
            rays_all.append(rays)
        
        self.rays_all = torch.cat(rays_all).to(self.device)

        points = torch.cat(world_points)
        points = crop_pc(points, self.bbox)
        points = ((points - self.origin) / self.scale).to(self.device)

        self.grid = OctreeGrid.from_pointcloud(
                    pointcloud=points,
                    feature_dim=self.feature_dim,
                    base_lod=self.base_lod, 
                    num_lods=self.num_lods,
                    interpolation_type=self.interpolation_type, 
                    multiscale_type=self.multiscale_type,
                    feature_std=self.feature_std,
                    feature_bias=self.feature_bias,
                    dilate = self.dilate
                    ).to(self.device)
        
        self.occ_decoder, self.rgb_decoder = self.init_decoders()

        return keyframe_list, keyframe_rays, keyframe_poses
        

    def init_decoders(self):
        
        # occ_decoder = OCCDecoder(self.cfg['models_global']["occ_decoder"], occ_feat_dim=self.occ_dim).to(self.device)
        # rgb_decoder = RGBDecoder(self.cfg['models_global']["rgb_decoder"], self.rgb_dim).to(self.device) 

        # occ_decoder = MLP('occ', dim = 3, c_dim = self.occ_dim).to(self.device) 
        # rgb_decoder = MLP('color', dim = 3, c_dim=self.rgb_dim, color = True).to(self.device) 
        
        
        occ_decoder = OCCDecoder(self.cfg['models_global']["occ_decoder"], occ_feat_dim=self.occ_dim).to(self.device)
        rgb_decoder = RGBDecoder(self.cfg['models_global']["rgb_decoder"], self.rgb_dim).to(self.device) 
        
        return occ_decoder, rgb_decoder
    
    def get_params_optimized(self, grid_lr, occ_decoder_lr, rgb_decoder_lr, grid, decoders): 
        params = []

        for i in range(grid.num_lods):
            grid.features[i] = Variable(grid.features[i].to(self.device), requires_grad = True)
            params.append({'params':grid.features[i], 'lr': grid_lr[i]})

        params.append({'params': decoders['rgb'].parameters(), 'lr':rgb_decoder_lr}) 
        params.append({'params': decoders['occ'].parameters(), 'lr': occ_decoder_lr})

        return params
    

    
    def get_rays(self, rays, c2w):
        color = rays[:, 3:6]
        depth = rays[:, -1]
        dirs = rays[:, :3]

        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1).type(torch.float32)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        rays_o = (rays_o - self.origin) / self.scale

        return torch.cat([rays_o, rays_d, color, depth], -1)


    def update_grid(self, optimizer, iters):
        # Init this epoch
        loss_dict_total = {'loss': 0., 'unc': 0.}
        for t in LOSS_TYPE:
            loss_dict_total[t] = 0.
        
        n_rays = len(self.rays_all)
        idxs = random.sample(range(0, n_rays), self.bs)
        rays = self.rays_all[idxs]

        batch_rays_o = rays[:, :3]
        batch_rays_d = rays[:, 3:6]
        batch_gt_color = rays[:, 6:9]
        batch_gt_depth = rays[:, -1]

        batch_rays = Rays(batch_rays_o, batch_rays_d)

        decoders = {'occ': self.occ_decoder, 'rgb': self.rgb_decoder}

        for i in range(iters):

            loss_dict = self.renderer.get_hit(batch_rays, batch_gt_color, batch_gt_depth, self.grid, decoders)

            loss = 0.
            for k, v in loss_dict.items():
                loss += self.lamudas[k] * loss_dict[k]
                loss_dict_total[k] += loss_dict[k].cpu().item()

            loss.backward(retain_graph = True)
            optimizer.step()
            optimizer.zero_grad()       

            loss_dict_total['loss']+= loss.cpu().item()

            self.i += 1

            if self.verbose:
                edesc = f'Iter {self.i} train info: '
                for k, v in loss_dict_total.items():
                    loss_dict_total[k] = v
                    if v > 0:
                        edesc += k + '=' + "{:.5f} ".format(v)
                
                print(edesc) 

        return loss_dict_total
    
    
    def run(self):

        while (True):

            if self.GBA_flag.value:
                self.anchor_frame = self.keyframe_list[0]
                break
                
            time.sleep(0.05)

        self.init_model()

        decoders = {'occ': self.occ_decoder, 'rgb': self.rgb_decoder}
        params = self.get_params_optimized(self.grid_lr, self.occ_lr, self.rgb_lr, self.grid, decoders)
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.99))

        for i in range(self.iters_out):
            self.update_grid(optimizer, self.iters_in)

            self.visualizer.extract_mesh(f'global_{i}', [self.anchor_frame], self.anchor_frame, self.mesh_reso, [self.grid], None, [self.occ_decoder], [self.rgb_decoder], False, self.clean_mesh)