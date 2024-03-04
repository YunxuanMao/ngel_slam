import os
import yaml
import json
import numpy as np
import math
import open3d as o3d
import skimage
import torch
import torch.nn.functional as F
from packaging import version
import trimesh
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from wisp.core import Rays

import pytorch_ssim
import lpips
import cv2

from src.common import get_rays_all

def cal_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value

def read_json_to_dict(json_path: str, ):
    with open(json_path, 'r') as f:
        dic = json.load(f)
        return dic

def depth2normal(depth_map):
    rows, cols = depth_map.shape

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('normal.png', normal_bgr)

    return normal


class Visualizer(object):
    """
    Mesher thread. Update the NeRF model

    """

    def __init__(self, cfg, args, pipeline, points_batch_size=500000):
        self.cfg = cfg
        self.args = args
        self.device = pipeline.device

        self.origin = pipeline.origin.cpu()
        self.scale = pipeline.scale.cpu()
        self.renderer = pipeline.renderer

        self.reso = cfg['meshing']['reso']
        self.marching_cubes_bound = cfg['mapping']['marching_cubes_bound']
        self.bound = torch.from_numpy(np.array(cfg['mapping']['marching_cubes_bound']))
        self.points_batch_size = points_batch_size
        self.frame_reader = pipeline.frame_reader
        self.samples_per_vox_test = cfg['mapping']['samples_per_vox_test']
        self.test_size = cfg['mapping']['test_chunk']

        self.pos_embed = cfg['pos_embed']

        self.input_folder = pipeline.input_folder
        self.output_folder = pipeline.output_folder
        
        self.img_output_folder = os.path.join(self.output_folder, 'vis')
        os.makedirs(self.img_output_folder, exist_ok=True)
        
        self.mesh_output_folder = os.path.join(self.output_folder, 'mesh')
        os.makedirs(self.mesh_output_folder, exist_ok=True)

        self.loss_fn_alex = lpips.LPIPS(net='alex')
        self.ssim_loss = pytorch_ssim.SSIM(window_size = 11)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = pipeline.H, pipeline.W, pipeline.fx, pipeline.fy, pipeline.cx, pipeline.cy

    
    def load_rel_c2w(self, anchor_frames, pose_file):
        

        info = read_json_to_dict(pose_file)
        kf_poses = info['KFposes']

        frames_in_kf = []
        rel_c2w = {}
        rel_c2w_inv = {}

        for k,v in kf_poses.items():
            idx = int(k)
            if idx in anchor_frames:
                std_c2w = self.anchor_std_c2w[idx].cpu()
                position = torch.tensor(v[:3])
                orientation = np.array(v[3:])
                c2w = torch.zeros([4,4])
                c2w[3, 3] = 1

                c2w[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix())
                c2w[:3, 3] = position

                # est_c2w_inv = self.world_std_c2w @ c2w.inverse() @ self.world_std_c2w.inverse()
                # est_c2w = self.world_std_c2w @ c2w @ self.world_std_c2w.inverse()
                est_c2w_inv = std_c2w @ c2w.inverse() @ self.world_std_c2w.inverse()
                est_c2w = self.world_std_c2w @ c2w @ std_c2w.inverse()
            
                rel_c2w_inv[idx] = est_c2w_inv
                rel_c2w[idx] = est_c2w

                frames_in_kf.append(idx)
        

        return rel_c2w, rel_c2w_inv, frames_in_kf


    @torch.no_grad()
    def extract_mesh(self, frame_idx, anchor_frames, pose_file_idx, reso, grids, embedders, occ_decoders, rgb_decoders, vis_every = False, clean = True):
        pose_file = os.path.join(self.input_folder, 'keyframes', f'{int(pose_file_idx)}.json')
        rel_c2w, _, frames_in_kf = self.load_rel_c2w(anchor_frames, pose_file)

        print('Extracting meshes ...')
        device = self.device

        nx = ny = nz = reso
        
        xmin, xmax, ymin, ymax, zmin, zmax = self.marching_cubes_bound[0][0],self.marching_cubes_bound[0][1],self.marching_cubes_bound[1][0],self.marching_cubes_bound[1][1],self.marching_cubes_bound[2][0],self.marching_cubes_bound[2][1]
        # xmin, xmax, ymin, ymax, zmin, zmax = -1, 1, -1, 1, -1, 1
        x = torch.linspace(xmin, xmax, nx)
        y = torch.linspace(ymin, ymax, ny)
        z = torch.linspace(zmin, zmax, nz)

        mesh_grid = torch.stack(torch.meshgrid(x,y,z)).permute(1, 2, 3, 0)
        mesh_grid = (mesh_grid - self.origin)/self.scale

        meshes = []
        frames_in_kf.sort()
        # frames_in_kf.pop(-1)
        for i in range(len(frames_in_kf)):
            idx = frames_in_kf[i]
            
            coords = mesh_grid
            chunk = 16
            full_val = torch.empty(list(mesh_grid.shape[0:3]), device=self.device)
            for c in range(0, coords.shape[0], chunk):
                query_coords = coords[c:c+chunk,:,:,:].contiguous()
                if self.pos_embed:
                    val = self.renderer.query_occ(query_coords.to(device), grids[idx], embedders[idx], occ_decoders[idx])
                else:
                    val = self.renderer.query_occ(query_coords.to(device), grids[idx], None, occ_decoders[idx])
                full_val[c:c+chunk,:,:] = val[...,0]

            occ = full_val.cpu().numpy()

            if (occ > 0).any():
                vertices, faces, _, _ = skimage.measure.marching_cubes(occ, 0., spacing = (x[2] - x[1], y[2] - y[1], z[2] - z[1]))
                vertices += np.array([x[0], y[0], z[0]])
                vertices = (vertices - self.origin.cpu().numpy()) / self.scale.cpu().numpy()

                if clean:
                    mesh = trimesh.Trimesh(vertices=vertices,
                                            faces=faces,
                                            process=False)
                    
                    components = mesh.split(only_watertight=False)
                    new_components = []
                    for comp in components:
                        if comp.area > 0.02:
                            new_components.append(comp)
                        mesh = trimesh.util.concatenate(new_components)
                    if not mesh == []:
                        vertices = mesh.vertices
                        faces = mesh.faces
                
                if self.pos_embed:
                    vertex_colors = self.renderer.query_rgb(torch.tensor(vertices).float().to(self.device), grids[idx], embedders[idx], rgb_decoders[idx]).cpu().numpy()
                else:
                    vertex_colors = self.renderer.query_rgb(torch.tensor(vertices).float().to(self.device), grids[idx], None, rgb_decoders[idx]).cpu().numpy()
                vertex_colors = np.clip(vertex_colors, 0, 1) * 255
                vertex_colors = vertex_colors.astype(np.uint8)

                vertices = vertices * self.scale.cpu().numpy() + self.origin.cpu().numpy()
                
                ones = np.ones_like(
                            vertices[:, 0]).reshape(-1, 1)
                homo_vertices = np.concatenate([vertices, ones], 1).reshape(
                            -1, 4, 1).astype(np.float32)
                
                c2w = rel_c2w[idx]
                vertices = (c2w.cpu().numpy() @ homo_vertices)[:, :3]
                


                mesh = trimesh.Trimesh(vertices.squeeze(), faces, vertex_colors=vertex_colors)
                meshes.append(mesh)
                if vis_every:
                    mesh_out_file = os.path.join(self.mesh_output_folder, f'mesh_{frame_idx}_{idx}.ply')
                    # mesh = trimesh.util.concatenate(meshes)
                    mesh.export(mesh_out_file)
                
                i += 1

        mesh = trimesh.util.concatenate(meshes)
        mesh_out_file = os.path.join(self.mesh_output_folder, f'mesh_{frame_idx}_{pose_file_idx}.ply')
        mesh.export(mesh_out_file)
        print(f'Mesh extract Done! File path: {mesh_out_file}')


    @torch.no_grad()
    def extract_mesh_one(self, frame_idx, anchor_frames, pose_file_idx, reso, grids, embedders, occ_decoders, rgb_decoders, vis_every = False, clean = True):
        pose_file = os.path.join(self.input_folder, 'keyframes', f'{int(pose_file_idx)}.json')
        rel_c2w, _, frames_in_kf = self.load_rel_c2w(anchor_frames, pose_file)

        print('Extracting meshes ...')
        device = self.device

        nx = ny = nz = reso
        
        xmin, xmax, ymin, ymax, zmin, zmax = self.marching_cubes_bound[0][0],self.marching_cubes_bound[0][1],self.marching_cubes_bound[1][0],self.marching_cubes_bound[1][1],self.marching_cubes_bound[2][0],self.marching_cubes_bound[2][1]
        # xmin, xmax, ymin, ymax, zmin, zmax = -1, 1, -1, 1, -1, 1
        x = torch.linspace(xmin, xmax, nx)
        y = torch.linspace(ymin, ymax, ny)
        z = torch.linspace(zmin, zmax, nz)

        mesh_grid = torch.stack(torch.meshgrid(x,y,z)).to(device).permute(1, 2, 3, 0)
        mesh_grid = (mesh_grid - self.origin)/self.scale

        meshes = []
        frames_in_kf.sort()
        # frames_in_kf.pop(-1)
        occ_list = []
        for i in range(len(frames_in_kf)):
            idx = frames_in_kf[i]
            
            coords = mesh_grid
            chunk = 16
            full_val = torch.empty(list(mesh_grid.shape[0:3]), device=self.device)
            for c in range(0, coords.shape[0], chunk):
                query_coords = coords[c:c+chunk,:,:,:].contiguous()
                shape = query_coords.shape
                query_coords.reshape(-1, 3)
                query_coords = query_coords * self.scale.to(query_coords) + self.origin.to(query_coords)
                ones = torch.ones_like(query_coords[:, 0])
                homo_vertices = torch.cat([query_coords, ones], -1).reshape(-1, 4, 1)
                
                c2w = rel_c2w[idx]
                query_coords = (c2w @ homo_vertices)[:, :3].reshape(*shape)

                if self.pos_embed:
                    val = self.renderer.query_occ(query_coords, grids[idx], embedders[idx], occ_decoders[idx])
                else:
                    val = self.renderer.query_occ(query_coords, grids[idx], None, occ_decoders[idx])
                full_val[c:c+chunk,:,:] = val[...,0]

            occ_list.append(full_val.cpu().numpy())

        occ = np.concatenate(occ_list)
        occ = np.max(occ, axis = 0)
        

        if (occ > 0).any():
            vertices, faces, _, _ = skimage.measure.marching_cubes(occ, 0., spacing = (x[2] - x[1], y[2] - y[1], z[2] - z[1]))
            mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=faces,
                                        process=False)
        
            # mesh = trimesh.util.concatenate(meshes)
            mesh_out_file = os.path.join(self.mesh_output_folder, f'mesh_{frame_idx}_{pose_file_idx}.ply')
            mesh.export(mesh_out_file)
            print(f'Mesh extract Done! File path: {mesh_out_file}')

        #     if clean:
        #         mesh = trimesh.Trimesh(vertices=vertices,
        #                                 faces=faces,
        #                                 process=False)
                
        #         components = mesh.split(only_watertight=False)
        #         new_components = []
        #         for comp in components:
        #             if comp.area > 0.02:
        #                 new_components.append(comp)
        #             mesh = trimesh.util.concatenate(new_components)
        #         if not mesh == []:
        #             vertices = mesh.vertices
        #             faces = mesh.faces
                
        #         if self.pos_embed:
        #             vertex_colors = self.renderer.query_rgb(torch.tensor(vertices).float().to(self.device), grids[idx], embedders[idx], rgb_decoders[idx]).cpu().numpy()
        #         else:
        #             vertex_colors = self.renderer.query_rgb(torch.tensor(vertices).float().to(self.device), grids[idx], None, rgb_decoders[idx]).cpu().numpy()
        #         vertex_colors = np.clip(vertex_colors, 0, 1) * 255
        #         vertex_colors = vertex_colors.astype(np.uint8)

        #         vertices = vertices * self.scale.cpu().numpy() + self.origin.cpu().numpy()
                
        #         ones = np.ones_like(
        #                     vertices[:, 0]).reshape(-1, 1)
        #         homo_vertices = np.concatenate([vertices, ones], 1).reshape(
        #                     -1, 4, 1).astype(np.float32)
                
        #         c2w = rel_c2w[idx]
        #         vertices = (c2w.cpu().numpy() @ homo_vertices)[:, :3]
                


        #         mesh = trimesh.Trimesh(vertices.squeeze(), faces, vertex_colors=vertex_colors)
        #         meshes.append(mesh)
        #         if vis_every:
        #             mesh_out_file = os.path.join(self.mesh_output_folder, f'mesh_{frame_idx}_{idx}.ply')
        #             # mesh = trimesh.util.concatenate(meshes)
        #             mesh.export(mesh_out_file)
                
        #         i += 1

        # mesh = trimesh.util.concatenate(meshes)
        # mesh_out_file = os.path.join(self.mesh_output_folder, f'mesh_{frame_idx}_{pose_file_idx}.ply')
        # mesh.export(mesh_out_file)
        # print(f'Mesh extract Done! File path: {mesh_out_file}')

    @torch.no_grad()
    def render_img(self, frame_idx, anchor_frames, pose_file_idx, grids, embedders, occ_decoders, rgb_decoders, vis = True, vis_every = False, fuse_method = 'max', gt_pose = True):
        pose_file = os.path.join(self.input_folder, 'keyframes', f'{int(pose_file_idx)}.json')
        _, rel_c2w_inv, frames_in_kf = self.load_rel_c2w(anchor_frames, pose_file)

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        ret = self.frame_reader[frame_idx]
        gt_color, gt_depth, gt_c2w = ret['color'], ret['depth'], ret['pose']
        if gt_pose:
            c2w = gt_c2w
        else:
            info = read_json_to_dict(pose_file)
            kf_poses = info['KFposes']
            # v = kf_poses[str(frame_idx)]
            v = [-1.866490841, 1.700594902, -3.820405245, -0.018881891, 0.772576213, 0.366389930, 0.518196762]
            position = torch.tensor(v[:3]).to(self.device)
            orientation = np.array(v[3:])
            c2w = torch.zeros([4,4]).to(self.device)
            c2w[3, 3] = 1

            c2w[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix()).to(self.device)
            c2w[:3, 3] = position
            c2w = self.world_std_c2w @ c2w

        gt_color = gt_color.cpu().detach().numpy()
        gt_depth = gt_depth.cpu().detach().numpy()

        depths = []
        colors = []
        uncertainties = []
        depth_loss_list = []
        psnr_list = []

        # print(f'Rendering frame {frame_idx}...')
        for i in range(len(frames_in_kf)):
            idx = frames_in_kf[i]
            c2w = rel_c2w_inv[idx] @ c2w
            decoders = {'occ': occ_decoders[idx], 'rgb': rgb_decoders[idx]}
            if self.pos_embed:
                decoders['embedder'] = embedders[idx]

            batch_rays_o, batch_rays_d = get_rays_all(
                            H, W, fx, fy, cx, cy, c2w, self.device)
            batch_rays_o = batch_rays_o.reshape(-1, 3)
            batch_rays_d = batch_rays_d.reshape(-1, 3)
            batch_rays_o = (batch_rays_o - self.origin) / self.scale

            rays = Rays(batch_rays_o, batch_rays_d).to(self.device)

            num_rays = len(rays)
            num_iters = num_rays // self.test_size + 1


            batch_begin = 0
            uncertainty_list = []
            uncertainty_frame_list = []
            color_list = []
            depth_list = []
            

            for iter in range(num_iters):
                batch_end = min(num_rays, batch_begin + self.test_size)
                if batch_begin >= batch_end:
                    break
                ret = self.renderer.raytrace(rays[batch_begin:batch_end], self.samples_per_vox_test, grids[idx], decoders, lod_idx = -1, test = True)

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
            torch.cuda.empty_cache()

        uncertainties_frame = np.concatenate(uncertainties, -1).reshape(-1, len(frames_in_kf)).mean(0)
        
        if fuse_method == 'max':
            idx = np.argsort(uncertainties_frame)

            depth_fuse = depths[idx[0]]
            color_fuse = colors[idx[0]]
            uncertainty_fuse = uncertainties[idx[0]]

        if fuse_method == 'sum':
            idx = np.argsort(uncertainties_frame)
            depth_fuse = np.zeros_like(depth).reshape(-1, 1)
            color_fuse = np.zeros_like(color).reshape(-1, 3)
            uncertainty = np.ones_like(depth).reshape(-1)
            for i in idx:
                depth_fuse[uncertainty == 1] = depths[i].reshape(-1, 1)[uncertainty == 1]
                color_fuse[uncertainty == 1] = colors[i].reshape(-1, 3)[uncertainty == 1]
                uncertainty[uncertainty == 1] = uncertainties[i].reshape(-1)[uncertainty == 1]
                if (uncertainty ==1).any():
                    continue
                else:
                    break
            
            depth_fuse = depth_fuse.reshape(H, W, 1)
            color_fuse = color_fuse.reshape(H, W, 3)
            uncertainty_fuse = uncertainty.reshape(H, W, 1)


            


        loss_depth = np.abs(gt_depth - depth_fuse.squeeze())[gt_depth != 0].mean()
        psnr = cal_psnr(gt_color, color_fuse)
        ssim1 = self.ssim_loss(torch.tensor(gt_color).permute(2,1,0).unsqueeze(0).float().cuda(), torch.tensor(color_fuse).permute(2,1,0).unsqueeze(0).float().cuda())
        lpips1 = self.loss_fn_alex(torch.tensor(gt_color).permute(2,1,0).unsqueeze(0).float(), torch.tensor(color_fuse).permute(2,1,0).unsqueeze(0).float())

        # print('Rendering Done!')
        # print(f'Depth loss: {depth_loss_list}')
        # print(f'PSNR: {psnr_list}')
        # print(f'Uncertainties: {uncertainties_frame}')

        # print(f'Depth loss fuse: {loss_depth}')
        # print(f'PSNR fuse: {psnr}, SSIM: {ssim1}, LPIPS: {lpips1}')
        if vis:
            self.vis(f'{pose_file_idx}', frame_idx, gt_depth, depth_fuse.squeeze(), uncertainty_fuse.squeeze(), color_fuse, gt_color)
            print(f'Images saves at {self.img_output_folder}.')

        return loss_depth, psnr, ssim1, lpips1
            
    
    
    
    def vis(self, file_idx, idx, depth_gt, depth_occ, uncertainty_occ, color, color_gt):

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
                            vmin=0, vmax=depth_residual_occ.max())
        axs[2].set_title('Depth Residual')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[3].imshow(uncertainty_occ, cmap="plasma",
                            vmin=0, vmax=0.25)
        axs[3].set_title('Uncertainty')
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0.3)
        plt.savefig(
            os.path.join(self.img_output_folder, f"{idx}_{file_idx}_occ.jpg"), bbox_inches='tight', pad_inches=0.2)
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
            os.path.join(self.img_output_folder, f"{idx}_{file_idx}_color.jpg"), bbox_inches='tight', pad_inches=0.2)
        plt.close()

        fig, axs = plt.subplots(1, 1)
        axs.imshow(depth_occ, cmap="plasma",
                            vmin=0, vmax=max_depth)
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_depth_only.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(depth_residual_occ, cmap="plasma",
                            vmin=0, vmax=depth_residual_occ.max())
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_depth_res.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(uncertainty_occ, cmap="viridis",
                            vmin=0, vmax=uncertainty_occ.max())
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_depth_unc.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(depth_gt, cmap="plasma",
                            vmin=0, vmax=max_depth)
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_depth_gt_only.jpg", dpi=100, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(color, cmap="plasma")
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_color_only.jpg", dpi=500, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(color_gt, cmap="plasma")
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_color_gt_only.jpg", dpi=500, bbox_inches='tight', pad_inches = -0.1)
        plt.close()

        normal = depth2normal(depth_occ)
        normal_gt = depth2normal(depth_gt)

        fig, axs = plt.subplots(1, 1)
        axs.imshow(normal_gt)
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_normal_gt_only.jpg", dpi=500, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(normal)
        axs.set_xticks([])
        axs.set_yticks([])
        plt.savefig(f"{self.img_output_folder}/{idx}_{file_idx}_normal_only.jpg", dpi=500, bbox_inches='tight', pad_inches = -0.1)
        plt.close()
