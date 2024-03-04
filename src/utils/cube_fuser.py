import os
import time
import sys
import math
import yaml
from packaging import version

import numpy as np
import torch
import skimage
import trimesh
import torch.nn.functional as F
import pytorch_ssim
import lpips

from tqdm import trange

import open3d as o3d

from wisp.core import Rays
from wisp.models.grids import OctreeGrid
from wisp.accelstructs import OctreeAS

from scipy.spatial.transform import Rotation as R


from src.models.sdf_grid import SDFGrid
from src.utils.datasets import get_dataset
from src.utils.visualizer import batchify_vol
from src.common import crop_pc, get_pc_from_depth, get_rays_all

import matplotlib.pyplot as plt

def cal_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

class CubeFuser():

    def __init__(self, cfg, args, first_frames):
        self.cfg = cfg
        self.args = args
        self.device = cfg['device']
        self.points_batch_size = 50000
        torch.cuda.set_device(self.device)
        
        self.position_encoding = cfg['models']['position_encoding']
        self.use_occ = cfg['models']['use_occ']
        self.use_sdf = cfg['models']['use_sdf']
        self.use_color = cfg['models']['use_color']
        self.use_sem = cfg['models']['use_sem'] if 'use_sem' in cfg['models'] else False
        self.use_ins = cfg['models']['use_ins'] if 'use_ins' in cfg['models'] else False
        assert (self.use_occ or self.use_sdf), 'At least one geometry expression! '

        self.multiscale_type = cfg['models']['octree']['multiscale_type']
        self.no_decoder = cfg['no_decoder']

        n_cat = 1
        if self.multiscale_type == 'cat':
            n_cat = self.num_lods
        if self.use_occ:
            self.occ_dim = cfg['models']['occ_dim'] * n_cat
        
        if self.use_sdf:
            self.sdf_dim = cfg['models']['sdf_dim'] * n_cat
        
        if self.use_color:
            self.rgb_dim = cfg['models']['color_dim'] * n_cat
            self.rgb_start = cfg['models']['color_start'] * n_cat
            # assert (self.rgb_dim + self.rgb_start) <= self.feature_dim

        if self.use_sem:
            self.sem_dim = cfg['models']['sem_dim'] * n_cat
            self.sem_start = (cfg['models']['sem_start'] + 1) * n_cat - 1
            # assert (self.sem_dim + self.sem_start) <= self.feature_dim
        
        if self.use_ins:
            self.ins_dim = cfg['models']['ins_dim'] * n_cat
            self.ins_start = (cfg['models']['ins_start'] + 1) * n_cat - 1

        self.output_folder = args.output_folder


        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.update_cam()
        
        self.frame_reader = get_dataset(cfg, args, 1, self.device)
        self.n_img = len(self.frame_reader)

        self.bbox = cfg['mapping']['bound']
        origin = (np.array(self.bbox)[:,1] + np.array(self.bbox)[:,0]) / 2
        scale = np.ceil((np.array(self.bbox)[:,1] - np.array(self.bbox)[:,0]).max()/2)
        self.origin = torch.tensor(origin, dtype=torch.float32).to(self.device)
        self.scale = torch.tensor(scale, dtype=torch.float32).to(self.device)

        self.img_output_folder = os.path.join(self.output_folder, 'vis')
        os.makedirs(self.img_output_folder, exist_ok=True)

        self.mesh_output_folder = os.path.join(self.output_folder, 'mesh')
        os.makedirs(self.mesh_output_folder, exist_ok=True)

        self.first_frames = first_frames
        if len(self.first_frames.shape) == 0:
            self.first_frames = [self.first_frames]
        self.init_frames = self.first_frames[0]
            
        models_paths = []
        for first_frame in self.first_frames:
            models_paths.append(os.path.join(self.output_folder, str(int(first_frame)), 'models.pth'))

        
        self.load_models(models_paths)

        self.input_folder = args.input_folder

        with open(os.path.join(self.input_folder, 'keyframes', 'map_info.yaml'), 'r') as f:
            data = yaml.full_load(f)
        loop_closes = data['loop_close']
        self.loop_closes = loop_closes

        self.loss_fn_alex = lpips.LPIPS(net='alex')
        self.ssim_loss = pytorch_ssim.SSIM(window_size = 11)
        
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

    
    def load_models(self, models_paths):
        self.grids_all = []
        for models_path in models_paths:
            models = torch.load(models_path)
            self.shared_grid = models['grid']
            # self.shared_s = models['s']
            if self.use_sdf:
                self.shared_sdf_decoder = models['sdf_decoder']
            if self.use_occ:
                self.shared_occ_decoder = models['occ_decoder']
            if self.use_color:
                self.shared_rgb_decoder = models['rgb_decoder']
            if self.use_sem:
                self.shared_rgb_decoder = models['sem_decoder']
            grid = SDFGrid(self.cfg, self.args, self)
            self.grids_all.append(grid)

    def load_rel_c2w(self, frames, pose_file):
        ret = self.frame_reader[int(self.init_frames)]
        std_c2w = ret['pose']
        # std_c2w[:3, 1] *= -1
        # std_c2w[:3, 2] *= -1

        kf_infos = np.loadtxt(pose_file).reshape(-1, 8)

        frame_in_kf = []
        self.kf = []

        self.rel_c2w = []
        self.rel_c2w_inv = []
        for info in kf_infos:
            idx = int(info[0])
            if idx in frames:
                position = torch.tensor([info[1], info[2], info[3]]).to(self.device)
                orientation = np.array([info[4], info[5], info[6], info[7]])
                c2w = torch.zeros([4,4]).to(self.device)
                c2w[3, 3] = 1

                c2w[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix()).to(self.device)
                c2w[:3, 3] = position
                # c2w[:3, 1] *= -1
                # c2w[:3, 2] *= -1
                est_c2w_inv = std_c2w @ c2w.inverse() @ std_c2w.inverse()
                # est_c2w_inv[:3, 1] *= -1
                # est_c2w_inv[:3, 2] *= -1
                est_c2w = std_c2w @ c2w @ std_c2w.inverse()
            
                self.rel_c2w_inv.append(est_c2w_inv)
                self.rel_c2w.append(est_c2w)

                frame_in_kf.append(np.argwhere(np.array(frames) == idx).squeeze())
                self.kf.append(idx)
        self.kf.append(len(self.frame_reader)-1)
        self.grids = []
        for i in frame_in_kf:
            self.grids.append(self.grids_all[i])


    @torch.no_grad()
    def render_img(self, frame_idx, pose_file_idx, vis_every = False, vis = True):
        pose_file = os.path.join(self.input_folder, 'keyframes', f'{int(pose_file_idx)}.txt')
        self.load_rel_c2w(self.first_frames, pose_file)

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        ret = self.frame_reader[frame_idx]
        idx, gt_color, gt_depth, gt_c2w, gt_sem, gt_ins = ret['index'], ret['color'], ret['depth'], ret['pose'], None, None
        gt_color = gt_color.cpu().detach().numpy()
        gt_depth = gt_depth.cpu().detach().numpy()
        # gt_c2w[:3, 1] *= -1
        # gt_c2w[:3, 2] *= -1
        

        depths = []
        colors = []
        uncertainties = []
        depth_loss_list = []
        psnr_list = []

        i = 0
        
        for grid in self.grids:

            c2w = self.rel_c2w_inv[i] @ gt_c2w
            # c2w = gt_c2w @ self.rel_c2w_inv[i]
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = gt_c2w
            # c2w = self.frame_reader[0]['pose']

            batch_rays_o, batch_rays_d = get_rays_all(
                            H, W, fx, fy, cx, cy, c2w, self.device)
            batch_rays_o = batch_rays_o.reshape(-1, 3)
            batch_rays_d = batch_rays_d.reshape(-1, 3)
            # batch_gt_depth = gt_depth.reshape(-1)
            # batch_gt_color = gt_color.reshape(-1, 3)
            batch_rays_o = (batch_rays_o - self.origin) / self.scale

            rays = Rays(batch_rays_o, batch_rays_d)

            num_rays = len(rays)
            num_iters = num_rays // 50000 + 1
            
            batch_begin = 0
            uncertainty_list = []
            uncertainty_frame_list = []
            color_list = []
            depth_list = []
            

            for iter in range(num_iters):
                batch_end = min(num_rays, batch_begin + 50000)
                if batch_begin >= batch_end:
                    break
                ret = grid.raytrace(rays[batch_begin:batch_end], 5)

                depth_list.append(ret['depth_occ']*self.scale)
                uncertainty_list.append(ret['uncertainty_occ'])
                color_list.append(ret['color'])

                batch_begin = batch_end


            depth = torch.cat(depth_list).reshape(H, W, 1).cpu().detach().numpy()
            uncertainty = torch.cat(uncertainty_list).reshape(H, W, 1).cpu().detach().numpy()
            color = torch.cat(color_list).reshape(H, W, 3).cpu().detach().numpy()

            loss_depth = np.abs(gt_depth - depth.squeeze())[gt_depth != 0].mean()
            psnr = cal_psnr(gt_color, color)
            

            if vis_every:
                self.vis(i, frame_idx, gt_depth, depth.squeeze(), uncertainty.squeeze(), color, gt_color)

            depths.append(depth)
            uncertainties.append(uncertainty)
            colors.append(color)
            depth_loss_list.append(loss_depth)
            uncertainty_frame_list.append(uncertainty.mean())
            psnr_list.append(psnr)

            i += 1
            torch.cuda.empty_cache()
        
        # depths = np.concatenate(depths, -1).reshape(-1, i)
        uncertainties_frame = np.concatenate(uncertainties, -1).reshape(-1, i).mean(0)
        idx = np.argsort(uncertainties_frame)
        # print(idx)

        certainties_frame = 5 - uncertainties_frame

        np.savetxt(os.path.join(self.img_output_folder, f"{frame_idx}_uncertainty.txt"), uncertainties_frame)
        np.savetxt(os.path.join(self.img_output_folder, f"{frame_idx}_depth_l1.txt"), depth_loss_list)
        np.savetxt(os.path.join(self.img_output_folder, f"{frame_idx}_psnr.txt"), psnr_list)


        # mean fuse
        # w0 = certainties_frame[idx[0]] / (certainties_frame[idx[0]] + certainties_frame[idx[1]])
        # w1 = certainties_frame[idx[1]] / (certainties_frame[idx[0]] + certainties_frame[idx[1]])

        # max fuse
        # depth_fuse = np.zeros_like(gt_depth)
        # color_fuse = np.zeros_like(gt_color)

        # w1 = uncertainties[idx[0]] > uncertainties[idx[1]]
        # w0 = uncertainties[idx[0]] < uncertainties[idx[1]]

        # color_fuse = (colors[idx[1]] * w1 + colors[idx[0]] * w0).reshape(H, W, 3)
        # depth_fuse = (depths[idx[1]] * w1 + depths[idx[0]] * w0).reshape(H, W)
        # uncertainty_fuse = (uncertainties[idx[1]] * w1 + uncertainties[idx[0]] * w0).reshape(H, W)

        depth_fuse = depths[idx[0]]
        color_fuse = colors[idx[0]]
        uncertainty_fuse = uncertainties[idx[0]]

        loss_depth = np.abs(gt_depth - depth_fuse.squeeze())[gt_depth != 0].mean()
        psnr = cal_psnr(gt_color, color_fuse)
        
        ssim1 = self.ssim_loss(torch.tensor(gt_color).permute(2,1,0).unsqueeze(0).float().cuda(), torch.tensor(color_fuse).permute(2,1,0).unsqueeze(0).float().cuda())
        
        lpips1 = self.loss_fn_alex(torch.tensor(gt_color).permute(2,1,0).unsqueeze(0).float(), torch.tensor(color_fuse).permute(2,1,0).unsqueeze(0).float())
        
        if vis:
            self.vis(f'fuse_{pose_file_idx}', frame_idx, gt_depth, depth_fuse.squeeze(), uncertainty_fuse.squeeze(), color_fuse, gt_color)
        
        print(f'Depth loss: {depth_loss_list}')
        print(f'PSNR: {psnr_list}')
        print(f'Uncertainties: {uncertainties_frame}')

        print(f'Depth loss fuse: {loss_depth}')
        print(f'PSNR fuse: {psnr}, SSIM: {ssim1}, LPIPS: {lpips1}')

        return loss_depth, psnr, ssim1, lpips1

    
    @torch.no_grad()
    def extract_mesh(self, reso, pose_file_idx, vis_every = False, clean = True):
        pose_file = os.path.join(self.input_folder, 'keyframes', f'{int(pose_file_idx)}.txt')
        self.load_rel_c2w(self.first_frames, pose_file)

        print('Extracting meshes ...')
        device = self.device

        nx = ny = nz = reso
        x = torch.linspace(-1, 1, nx)
        y = torch.linspace(-1, 1, ny)
        z = torch.linspace(-1, 1, nz)

        grid = torch.stack(torch.meshgrid(x,y,z)).to(device).permute(1, 2, 3, 0)

        i = 0
        meshes = []
        for sdfgrid in self.grids:
            occ = batchify_vol(sdfgrid.query_occ, 1)(grid).cpu().numpy()
            vertices, faces, _, _ = skimage.measure.marching_cubes(occ, 0., spacing = (x[2] - x[1], y[2] - y[1], z[2] - z[1]))
            vertices += np.array([x[0], y[0], z[0]])

            if clean:
                mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=faces,
                                        process=False)
                
                components = mesh.split(only_watertight=False)
                new_components = []
                for comp in components:
                    if comp.area > 0.05:
                        new_components.append(comp)
                    mesh = trimesh.util.concatenate(new_components)
                vertices = mesh.vertices
                faces = mesh.faces
            
            

            vertex_colors = sdfgrid.query_rgb(torch.tensor(vertices).float().cuda()).cpu().numpy()
            vertex_colors = np.clip(vertex_colors, 0, 1) * 255
            vertex_colors = vertex_colors.astype(np.uint8)

            vertices = vertices * self.scale.cpu().numpy() + self.origin.cpu().numpy()
            
            ones = np.ones_like(
                        vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate([vertices, ones], 1).reshape(
                        -1, 4, 1).astype(np.float32)
            c2w = self.rel_c2w[i]
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1

            vertices = (c2w.cpu().numpy() @ homo_vertices)[:, :3]
            


            mesh = trimesh.Trimesh(vertices.squeeze(), faces, vertex_colors=vertex_colors)
            meshes.append(mesh)
            if vis_every:
                mesh_out_file = os.path.join(self.mesh_output_folder, f'mesh_{i}.ply')
                mesh.export(mesh_out_file)
            
            i += 1
        

        mesh = trimesh.util.concatenate(meshes)
        mesh_out_file = os.path.join(self.mesh_output_folder, f'mesh_fuse_{reso}_{pose_file_idx}.ply')
        mesh.export(mesh_out_file)
        print(f'Extract Done! File path: {mesh_out_file}')
            

    def vis(self, iter, idx, depth_gt, depth_occ, uncertainty_occ, color, color_gt):

        max_depth = np.max(depth_gt)

        depth_residual_occ = np.abs(depth_gt - depth_occ)
        depth_residual_occ[depth_gt == 0.0] = 0.0
        fig, axs = plt.subplots(1, 4)
        fig.tight_layout()
        
        axs[0].imshow(depth_gt, cmap="plasma",
                            vmin=0, vmax=max_depth)
        axs[0].set_title('Input Depth')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].imshow(depth_occ, cmap="plasma",
                            vmin=0, vmax=max_depth)
        axs[1].set_title('Generated Depth')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[2].imshow(depth_residual_occ, cmap="plasma",
                            vmin=0, vmax=max_depth)
        axs[2].set_title('Depth Residual')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[3].imshow(uncertainty_occ, cmap="plasma",
                            vmin=0, vmax=1)
        axs[3].set_title('Uncertainty')
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0.3)
        plt.savefig(
            os.path.join(self.img_output_folder, f"{idx}_{iter}_occ.jpg"), bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        color_residual = np.abs(color_gt - color)
        color_gt = np.clip(color_gt, 0, 1)
        color = np.clip(color, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        fig, axs = plt.subplots(1, 3)
        fig.tight_layout()
        axs[0].imshow(color_gt, cmap="plasma")
        axs[0].set_title('Input RGB')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].imshow(color, cmap="plasma")
        axs[1].set_title('Generated RGB')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[2].imshow(color_residual, cmap="plasma")
        axs[2].set_title('RGB Residual')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
    
        plt.subplots_adjust(wspace=0, hspace=0.3)
        plt.savefig(
            os.path.join(self.img_output_folder, f"{idx}_{iter}_color.jpg"), bbox_inches='tight', pad_inches=0.2)
        plt.close()

        fig, axs = plt.subplots(1, 1)
        axs.imshow(depth_occ, cmap="plasma",
                            vmin=0, vmax=max_depth)
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{iter}_depth_only.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(depth_gt, cmap="plasma",
                            vmin=0, vmax=max_depth)
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{iter}_depth_gt_only.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(color, cmap="plasma")
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{iter}_color_only.jpg", dpi=500, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(color_gt, cmap="plasma")
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{iter}_color_gt_only.jpg", dpi=500, bbox_inches='tight', pad_inches = -0.1)
        plt.close()

        
    