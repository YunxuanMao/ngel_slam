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

from src.common import get_samples, get_rays_all, get_pc_from_depth, qp2tm, crop_pc

from src.models.decoder_nice import MLP
from src.models.decoder import OCCDecoder, RGBDecoder
from src.models.renderer import Renderer
from src.utils.visualizer import Visualizer

LOSS_TYPE = ['depth_occ', 'rgb', 'smooth', 'unc']

dmax = 300

class Mapper():

    def __init__(self, cfg, args, ssnerf):

        self.cfg = cfg
        self.args = args
        self.device = cfg['device']
        self.verbose = cfg['verbose']


        torch.cuda.set_device(self.device)
        self.dilate = cfg['dilate']
        self.bbox = cfg['mapping']['bound']

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


        # mesh config
        self.mesh_reso = cfg['meshing']['reso']
        self.clean_mesh = cfg['meshing']['clean']

        self.lamudas = {}
        for t in LOSS_TYPE:
            self.lamudas[t] = cfg['mapping']['lamudas'][t]
        
        self.input_folder = ssnerf.input_folder
        self.output_folder = ssnerf.output_folder
        
        self.ckpt_output_folder = os.path.join(self.output_folder, 'ckpt')
        os.makedirs(self.ckpt_output_folder, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = ssnerf.H, ssnerf.W, ssnerf.fx, ssnerf.fy, ssnerf.cx, ssnerf.cy
        self.origin = ssnerf.origin.to(self.device)
        self.scale = ssnerf.scale.to(self.device)

        

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

        self.keyframe_images = {}
        self.global_keyframe_images = {}

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

    def save_models(self, idx):
        print('Model saving ...')
        models_path = os.path.join(self.ckpt_output_folder, f'models_{idx}.pth')
        models = {'anchors': list(self.anchor_frames), 'world_std_c2w': self.world_std_c2w, 'anchor_std_c2w': self.anchor_std_c2w, 'grids': self.grids, 'occ_decoders': self.occ_decoders, 'rgb_decoders': self.rgb_decoders} # {'anchors': list(self.anchor_frames), 'kf_anchor': dict(self.local_keyframe_list), 
        if self.pos_embed:
            models['embedder'] = self.embedders
        
        torch.save(models, models_path)
        print(f'Models saved in {models_path}.')



    def init_decoders(self):
        
        # occ_decoder = OCCDecoder(self.cfg['models']["occ_decoder"], occ_feat_dim=self.occ_dim).to(self.device)
        # rgb_decoder = RGBDecoder(self.cfg['models']["rgb_decoder"], self.rgb_dim).to(self.device) 

        # occ_decoder = MLP('occ', dim = 3, c_dim = self.occ_dim).to(self.device) 
        # rgb_decoder = MLP('color', dim = 3, c_dim=self.rgb_dim, color = True).to(self.device) 
        
        pos_embedder, pos_embed_dim = get_positional_embedder(frequencies=self.pos_multires)
        pos_embedder.to(self.device)

        if not self.pos_embed:
            pos_embed_dim = 0
        
        occ_decoder = OCCDecoder(self.cfg['models']["occ_decoder"], occ_feat_dim=self.occ_dim + pos_embed_dim).to(self.device)
        rgb_decoder = RGBDecoder(self.cfg['models']["rgb_decoder"], self.rgb_dim + pos_embed_dim).to(self.device) 
        
        return pos_embedder, occ_decoder, rgb_decoder

    def get_rays(self, gt_color, gt_depth, c2w, pixs_per_image = None):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        
        if pixs_per_image is None:
            batch_rays_o, batch_rays_d = get_rays_all(
                        H, W, fx, fy, cx, cy, c2w, self.device)

            batch_rays_o =  batch_rays_o.reshape(-1, 3)
            batch_rays_d =  batch_rays_d.reshape(-1, 3)
            batch_gt_depth = gt_depth.reshape(-1)
            batch_gt_color = gt_color.reshape(-1, 3)

        
        else:
            batch_rays_o, batch_rays_d, indices = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, self.device)
        
            batch_gt_depth = gt_depth.reshape(-1)[indices]
            batch_gt_color = gt_color.reshape(-1, 3)[indices]

            

        batch_rays_o = (batch_rays_o - self.origin) / self.scale

        return batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color
    
    

    def update_grid(self, pixs_per_image, grid, decoders, optimizer, std_c2w, selected_keyframes, mapper = 'local'):
        '''
        Train model
        '''
        # Init this epoch
        loss_dict_total = {'loss': 0., 'unc': 0.}
        for t in LOSS_TYPE:
            loss_dict_total[t] = 0.
        
        
        num_kfs = len(selected_keyframes)

        pixs_per_kf = pixs_per_image // (num_kfs + 1)


        batch_rays_o_list, batch_rays_d_list, batch_gt_color_list, batch_gt_depth_list = [], [], [], []


        # get rays from keyframes
        for idx in selected_keyframes:
            if mapper == 'local':
                if idx in self.keyframe_poses and idx in self.keyframe_images:
                    pose = self.keyframe_poses[idx]
                    gt_color_i, gt_depth_i = self.keyframe_images[idx]['color'], self.keyframe_images[idx]['depth']
                    trans = np.array(pose[:3])
                    quat = np.array(pose[3:])
                    c2w = qp2tm(quat, trans).to(self.device)
                    
                    gt_c2w_i = std_c2w @ c2w

                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = self.get_rays(gt_color_i, gt_depth_i, gt_c2w_i, pixs_per_image = pixs_per_kf)

                    
                    batch_rays_o_list.append(batch_rays_o)
                    batch_rays_d_list.append(batch_rays_d)
                    batch_gt_color_list.append(batch_gt_color)
                    batch_gt_depth_list.append(batch_gt_depth)
            else:
                if idx in self.keyframe_poses and idx in self.global_keyframe_images:
                    pose = self.keyframe_poses[idx]
                    gt_color_i, gt_depth_i = self.global_keyframe_images[idx]['color'], self.global_keyframe_images[idx]['depth']
            
                    trans = np.array(pose[:3])
                    quat = np.array(pose[3:])
                    c2w = qp2tm(quat, trans).to(self.device)
                    
                    gt_c2w_i = std_c2w @ c2w

                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = self.get_rays(gt_color_i, gt_depth_i, gt_c2w_i, pixs_per_image = pixs_per_kf)

                    
                    batch_rays_o_list.append(batch_rays_o)
                    batch_rays_d_list.append(batch_rays_d)
                    batch_gt_color_list.append(batch_gt_color)
                    batch_gt_depth_list.append(batch_gt_depth)

        batch_rays_o = torch.cat(batch_rays_o_list)
        batch_rays_d = torch.cat(batch_rays_d_list)
        batch_gt_depth = torch.cat(batch_gt_depth_list)
        batch_gt_color = torch.cat(batch_gt_color_list)


        batch_rays = Rays(batch_rays_o, batch_rays_d)

        loss_dict = self.renderer.get_hit(batch_rays, batch_gt_color, batch_gt_depth, grid, decoders, dmax = dmax)

        loss = 0.
        for k, v in loss_dict.items():
            loss += self.lamudas[k] * loss_dict[k]
            loss_dict_total[k] += loss_dict[k].cpu().item()

        loss.backward(retain_graph = True)
        optimizer.step()
        optimizer.zero_grad()       

        loss_dict_total['loss']+= loss.cpu().item()


        if self.verbose:
            edesc = f'Iter {self.i} train info: '
            for k, v in loss_dict_total.items():
                loss_dict_total[k] = v
                if v > 0:
                    edesc += k + '=' + "{:.5f} ".format(v)
            
            print(edesc) 

        return loss_dict_total
    
    
    def grid_increment(self, std_c2w, change_frame_idx, grid = None, mapper = 'local'):
        gt_depth = []
        c2w = []
        
        for idx in change_frame_idx:
            if mapper == 'local':
                if idx in self.keyframe_images and idx in self.keyframe_poses:
                    gt_depth.append(self.keyframe_images[idx]['depth'])
                    pose = self.keyframe_poses[idx]
                    trans = np.array(pose[:3])
                    quat = np.array(pose[3:])
                    c2w_i = qp2tm(quat, trans).to(self.device)
                    c2w.append(std_c2w @ c2w_i)
            else:
                if idx in self.global_keyframe_images and idx in self.keyframe_poses:
                    gt_depth.append(self.global_keyframe_images[idx]['depth'])
                    pose = self.keyframe_poses[idx]
                    trans = np.array(pose[:3])
                    quat = np.array(pose[3:])
                    c2w_i = qp2tm(quat, trans).to(self.device)
                    c2w.append(std_c2w @ c2w_i)
        
        points = get_pc_from_depth(gt_depth, c2w, self.fx, self.fy, self.cx, self.cy, self.device, dmax=dmax)
        points.reshape(-1, 3)
        points = crop_pc(points, self.bbox)
        points = (points - self.origin) / self.scale

        # init
        if grid is None:
            grid = OctreeGrid.from_pointcloud(
                        pointcloud=points,
                        feature_dim=self.feature_dim,
                        base_lod=self.base_lod, 
                        num_lods=self.num_lods,
                        interpolation_type=self.interpolation_type, 
                        multiscale_type=self.multiscale_type,
                        feature_std=self.feature_std,
                        feature_bias=self.feature_bias,
                        dilate = self.dilate
                        )
            
        else:
            grid.update(points, self.device, dilate = self.dilate)

        
        return grid
    
    def get_new_std_c2w(self, anchor_frame):
        # TODO: quaternion to transform matrix
        pose = self.keyframe_poses[anchor_frame]
        trans = np.array(pose[:3])
        quat = np.array(pose[3:])
        c2w = qp2tm(quat, trans).to(self.device)

        std_c2w = self.anchor_std_c2w[anchor_frame] @ c2w.inverse()
        
        return std_c2w
    
    def local_load_images(self, frame_list):
        LBA_flag = False
        s = time.time()
        for frame in frame_list:
            if frame not in self.keyframe_images:
                # ret = self.frame_reader[frame]
                # images = {'color': ret['color'], 'depth': ret['depth']}
                color = self.share_color_images[frame].to(self.device)
                depth = self.share_depth_images[frame].to(self.device)
                self.keyframe_images[frame] = {'color': color, 'depth': depth}
                LBA_flag = True
        
        for frame in list(self.keyframe_images):
            if frame not in frame_list:
                del self.keyframe_images[frame]
        e = time.time()
        if LBA_flag:
            print(f'load image use {1000*(e-s)}ms')
        
        return LBA_flag

    def global_load_images(self, frame_list):
        LBA_flag = False
        s = time.time()
        for frame in frame_list:
            if frame not in self.global_keyframe_images:
                # ret = self.frame_reader[frame]
                # images = {'color': ret['color'], 'depth': ret['depth']}
                color = self.share_color_images[frame].to(self.device)
                depth = self.share_depth_images[frame].to(self.device)
                self.global_keyframe_images[frame] = {'color': color, 'depth': depth}
                LBA_flag = True
        
        for frame in list(self.global_keyframe_images):
            if frame not in frame_list:
                del self.global_keyframe_images[frame]
        e = time.time()
        if LBA_flag:
            print(f'load image use {1000*(e-s)}ms')
        
        return LBA_flag
    
    def load_images(self):
        LBA_flag = False
        s = time.time()
        for frame in self.keyframe_list:
            if frame not in self.keyframe_images:
                # ret = self.frame_reader[frame]
                # images = {'color': ret['color'], 'depth': ret['depth']}
                color = self.share_color_images[frame].to(self.device)
                depth = self.share_depth_images[frame].to(self.device)
                self.keyframe_images[frame] = {'color': color, 'depth': depth}
                LBA_flag = True
        
        for frame in list(self.keyframe_images):
            if frame not in self.keyframe_list:
                del self.keyframe_images[frame]
        e = time.time()
        if LBA_flag:
            print(f'load image use {1000*(e-s)}ms')
        
        return LBA_flag
    


    def get_params_optimized(self, grid_lr, occ_decoder_lr, rgb_decoder_lr, grid, decoders): 
        params = []

        for i in range(grid.num_lods):
            grid.features[i] = Variable(grid.features[i].to(self.device), requires_grad = True)
            params.append({'params':grid.features[i], 'lr': grid_lr[i]})

        params.append({'params': decoders['rgb'].parameters(), 'lr':rgb_decoder_lr}) 
        if self.init:
            params.append({'params': decoders['occ'].parameters(), 'lr': occ_decoder_lr})

        return params
    

    def update_volume(self, anchor_frame, local_ba_flag, active_kf, std_c2w, vol_init, mapper = 'local'):
        
        if vol_init:
            self.grids[anchor_frame] = self.grid_increment(std_c2w, active_kf, None, mapper)
            local_ba_flag = False
            self.local_volume_init.value = False

        grid = self.grids[anchor_frame]
        embedder = self.embedders[anchor_frame]
        occ_decoder = self.occ_decoders[anchor_frame]
        rgb_decoder = self.rgb_decoders[anchor_frame]
        decoders = {'occ': occ_decoder, 'rgb': rgb_decoder, 'embedder': embedder}


        if local_ba_flag:

            grid = self.grid_increment(std_c2w, active_kf, grid, mapper)


        params = self.get_params_optimized(self.grid_lr, self.occ_lr, self.rgb_lr, grid, decoders)

        optimizer = torch.optim.Adam(params)

        # selected_keyframes = self.keyframe_selection(gt_color, gt_depth, c2w, keyframe_list, self.keyframe_info, active_kf, grid, std_c2w)  
        selected_keyframes = random.choices(list(active_kf), list(active_kf), k=min(len(active_kf), self.keyframe_num))
        if active_kf[-1] not in selected_keyframes:
            selected_keyframes.append(active_kf[-1])

        # if self.verbose:
        #     print(Fore.GREEN)
        #     print(f"Epoch {self.i} mapping Frame Update: {selected_keyframes}" )
        #     print(Style.RESET_ALL)

        if self.init:
            depth_loss = []
            rgb_loss = []
            for i in range(self.init_iters):
                loss = self.update_grid(self.pix_per_frame, grid, decoders, optimizer, std_c2w, selected_keyframes, mapper)
                depth_loss.append(loss['depth_occ'].cpu().item())
                rgb_loss.append(loss['rgb'].cpu().item())
            np.savetxt('depth.txt', depth_loss)
            np.savetxt('rgb.txt', rgb_loss)


            self.init = False
            self.init_mapping.value = False
            self.i += 1
            print(Fore.GREEN)
            print(f'Mapping initializing done!')
            print(Style.RESET_ALL)

        else:
            for i in range(self.iters):
                self.update_grid(self.pix_per_frame, grid, decoders, optimizer, std_c2w, selected_keyframes, mapper)
            self.i += 1
        
        self.grids[anchor_frame] = grid
        self.embedders[anchor_frame] = decoders['embedder']
        self.occ_decoders[anchor_frame] = decoders['occ']
        self.rgb_decoders[anchor_frame] = decoders['rgb']
        
        torch.cuda.empty_cache()

        return grid, decoders
    

    def run(self):
    
        
        t1 = threading.Thread(target = self.local_mapping, name = 't1')
        t2 = threading.Thread(target = self.global_mapping, name = 't2')
        t1.start()
        t2.start()
        t1.join()
        t2.join()
            

    
    def local_mapping(self):
        torch.cuda.set_device(self.device)
        i = 0
        while True:
            if self.final.value:
                # for anchor in self.anchor_frames:
                #     std_c2w = self.get_new_std_c2w(anchor)
                #     local_optimized_keyframes = self.local_keyframe_list[anchor]
                #     self.update_volume(anchor, True, local_optimized_keyframes, std_c2w, False)
                #     for i in range(300):
                #         self.update_volume(anchor, False, local_optimized_keyframes, std_c2w, False)
                self.save_models('final')
                self.visualizer.extract_mesh('final', self.anchor_frames, self.n_img-1, self.mesh_reso, self.grids, self.embedders, self.occ_decoders, self.rgb_decoders, True, self.clean_mesh)
                time.sleep(1)
                self.final.value = False
                break

            vol_init = False
            if self.init:
                while True:
                    if self.init_mapping.value:
                        print(Fore.GREEN)
                        print(f'Mapping start initializing ...')
                        print(Style.RESET_ALL)
                        break
                
                ret = self.frame_reader[self.local_anchor.value]
                self.world_std_c2w = ret['pose']
                self.visualizer.world_std_c2w = self.world_std_c2w
                embedder, occ_decoder, rgb_decoder = self.init_decoders()
                self.init_decoder_flag.value = True

            if self.end_local_map_flag.value:
                embedder = copy.deepcopy(self.embedders[self.local_anchor.value])
                occ_decoder = copy.deepcopy(self.occ_decoders[self.local_anchor.value])
                rgb_decoder = copy.deepcopy(self.rgb_decoders[self.local_anchor.value])

                self.meshing_flag = True
                self.visualizer.extract_mesh(self.time_stamp.value, self.anchor_frames, self.time_stamp.value, self.mesh_reso, self.grids, self.embedders, self.occ_decoders, self.rgb_decoders, True, self.clean_mesh)
                self.meshing_flag = False
                self.end_local_map_flag.value = False

            if self.new_volume_flag.value:
                while True:
                    if not self.new_volume_flag.value:
                        break
                
                self.embedders[self.local_anchor.value] = embedder
                self.occ_decoders[self.local_anchor.value] = occ_decoder
                self.rgb_decoders[self.local_anchor.value] = rgb_decoder
                
                pose = self.keyframe_poses[self.local_anchor.value]
                trans = np.array(pose[:3])
                quat = np.array(pose[3:])
                c2w = qp2tm(quat, trans).to(self.device)
                self.anchor_std_c2w[self.local_anchor.value] = self.world_std_c2w.to(c2w) @ c2w
                self.visualizer.anchor_std_c2w = self.anchor_std_c2w
                vol_init = True

            if self.LBA_flag.value:
                print(Fore.GREEN)
                print(f'Stop mapping. Waiting for local ba...')
                print(Style.RESET_ALL)
                while True:
                    if not self.LBA_flag.value:
                        print(Fore.GREEN)
                        print(f'Local ba done. Continue mapping.')
                        print(Style.RESET_ALL)
                        break
                print(Fore.GREEN)
                print(f'New optimized frames # {len(self.local_optimized_keyframes)}: {self.local_optimized_keyframes}')
                print(Style.RESET_ALL)

            self.local_mapping_flag.value = True
            # self.local_volume_init.value = False
            # self.init_mapping.value = False

            if len(self.local_optimized_keyframes) > 50:
                local_optimized_keyframes = random.choices(self.local_optimized_keyframes, self.local_optimized_keyframes, k=50 )
            else:
                local_optimized_keyframes = self.local_optimized_keyframes
            # LBA_flag = self.local_load_images(local_optimized_keyframes)
            LBA_flag = self.load_images()
            
            std_c2w = self.get_new_std_c2w(self.local_anchor.value)
            
            self.update_volume(self.local_anchor.value, LBA_flag, local_optimized_keyframes, std_c2w, vol_init)
            
            if i % self.mesh_freq == 0:
                self.meshing_flag = True
                if self.global_mapping_flag.value:
                    while True:
                        if not self.global_mapping_flag.value:
                            break
                self.visualizer.extract_mesh(self.time_stamp.value, self.anchor_frames, self.time_stamp.value, self.mesh_reso, self.grids, self.embedders, self.occ_decoders, self.rgb_decoders, False, self.clean_mesh)
                self.meshing_flag = False
            
            if i % self.vis_freq == 0:
                if i == 0:
                    self.visualizer.render_img(self.anchor_frames[0], self.anchor_frames, self.time_stamp.value, self.grids, self.embedders, self.occ_decoders, self.rgb_decoders, vis=True)
                else:
                    self.visualizer.render_img(self.time_stamp.value, self.anchor_frames, self.time_stamp.value, self.grids, self.embedders, self.occ_decoders, self.rgb_decoders, vis=True)

            if i % self.ckpt_freq == 0:
                self.save_models(self.time_stamp.value)
            
            i += 1

            self.local_mapping_flag.value = False
        
    def global_mapping(self):
        torch.cuda.set_device(self.device)
        GBA = False

        while True:
            if self.gloal_mapping.value:
                break
            time.sleep(1)
            if self.final.value:
                break

        while True:
            
            # if self.GBA_flag.value:
            if self.final.value:
                break
            
            while True:
                global_anchor = random.choice(self.anchor_frames)
                if global_anchor != self.local_anchor.value:
                    print(Fore.RED)
                    print(f'Global mapping anchor {global_anchor}. kf_list: {self.local_keyframe_list[global_anchor]}')
                    print(Style.RESET_ALL)
                    break
            
            self.global_load_images(self.local_keyframe_list[global_anchor])
        
            for i in range(1000):
                if self.LBA_flag.value:
                    print(Fore.RED)
                    print(f'Stop mapping. Waiting for local ba...')
                    print(Style.RESET_ALL)
                    while True:
                        if not self.LBA_flag.value:
                            print(Fore.RED)
                            print(f'Local ba done. Continue mapping.')
                            print(Style.RESET_ALL)
                            break

                if self.meshing_flag:
                    while True:
                        if not self.meshing_flag:
                            break
                        time.sleep(0.005)

                if global_anchor == self.local_anchor.value:
                    break
                
                self.global_mapping_flag.value = True

                kf_list = random.choices(self.local_keyframe_list[global_anchor], self.local_keyframe_list[global_anchor],  k=20)
                # kf_list = self.local_keyframe_list[global_anchor]
                
                std_c2w = self.get_new_std_c2w(global_anchor)
                    
                self.update_volume(global_anchor, False, kf_list, std_c2w, False, 'global')
                
                self.global_mapping_flag.value = False

                if self.final.value:
                    break

                # time.sleep(0.005)
            

              




                                    


