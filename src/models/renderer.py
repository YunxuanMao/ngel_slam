import torch
import torch.nn.functional as F
import numpy as np

import kaolin.render.spc as spc_render

from wisp.core import Rays
from wisp.ops.differential import autodiff_gradient

class Renderer:
    def __init__(self, cfg, args, pipeline):
        self.args = args
        self.cfg = cfg
        self.device = cfg['device']
        torch.cuda.set_device(self.device)

        self.samples_per_vox = cfg['mapping']['samples_per_vox']
        self.samples_per_vox_test = cfg['mapping']['samples_per_vox_test']

        self.occ_dim = cfg['models']['occ_dim']
        self.rgb_dim = cfg['models']['color_dim']
        self.rgb_start = cfg['models']['color_start']
        self.smooth_length = cfg['mapping']['smooth_length']
        self.smooth = cfg['mapping']['smooth']
        self.pos_embed = cfg['pos_embed']

        self.origin = pipeline.origin
        self.scale = pipeline.scale

    
    def raytrace(self, rays, num_samples, grid, decoders, lod_idx = None, test = False):
        N = rays.origins.shape[0]
        if lod_idx is None:
            lod_idx = -1
        # if test:
        #     lod_idx = 1
            # num_samples = int(num_samples*2)

        # raymarch_results = grid.raymarch(rays, 'voxel', num_samples, level = grid.active_lods[lod_idx])
        raymarch_results = grid.raymarch(rays, 'voxel', num_samples, level = grid.active_lods[lod_idx])

        ridx = raymarch_results.ridx
        samples = raymarch_results.samples
        deltas = raymarch_results.deltas
        boundary = raymarch_results.boundary
        depths = raymarch_results.depth_samples

        # Get the indices of the ray tensor which correspond to hits
        ridx_hit = ridx[boundary]
        # Compute the color and density for each ray and their samples
        hit_ray_d = rays.dirs.index_select(0, ridx)

        num_samples = samples.shape[0]

        ret = self.nef(samples, grid, decoders)

        results = {}

        occ_value = ret['occ']
        alpha = torch.sigmoid(occ_value.contiguous())
        transmittance = spc_render.cumprod((1.-alpha + 1e-10).contiguous(), boundary.contiguous(), exclusive=True)
        occ_weight = alpha * transmittance

        occ_depth = torch.zeros(N, 1, device=rays.origins.device)
        ray_depth = spc_render.sum_reduce(depths.reshape(num_samples, 1) * occ_weight, boundary)
        occ_depth[ridx_hit, :] = ray_depth
        results['depth_occ'] = occ_depth
        if not test and self.smooth:
            # threshold = 0.05
            # mask = ((alpha - 0.5).abs() < threshold).reshape(-1)
            # results['smooth_loss'] = torch.tensor(0., device=self.device)
            # if mask.sum() > 100:
            #     results['smooth_loss'] = self.get_smooth_loss(grid, decoders['embedder'], decoders['occ'], samples[mask])
            results['smooth_loss'] = self.get_smooth_loss(grid, samples, ret['feats'])

        
        
        if test:
            occ_unc = torch.ones(N, 1, device=rays.origins.device) * 0.5
            ray_unc = spc_render.sum_reduce((1-alpha) * alpha, boundary)
            # ray_unc = spc_render.sum_reduce(torch.log(torch.max(alpha, 1e-10*torch.ones_like(alpha))) * alpha + torch.log(torch.max(1 - alpha, 1e-10*torch.ones_like(1 - alpha))) * (1 - alpha), boundary)
            ray_p_num = spc_render.sum_reduce(torch.ones_like(alpha).to(self.device), boundary)
            
            # ray_unc /= ray_p_num
            ray_unc += 1 - spc_render.sum_reduce(occ_weight, boundary)

            occ_unc[ridx_hit, :] = ray_unc
            results['uncertainty_occ'] = occ_unc

        rgb = torch.sigmoid(ret['rgb'])
        occ_rgb = torch.zeros(N, 3, device=rays.origins.device)
        ray_rgb = spc_render.sum_reduce(rgb * occ_weight, boundary)
        occ_rgb[ridx_hit] = ray_rgb
        results['color'] = occ_rgb
        
        return results

    def get_hit(self, rays, gt_color, gt_depth, grid, decoders, test = False, dmax = 0):
        if test:
            samples_per_vox = self.samples_per_vox_test
        else:
            samples_per_vox = self.samples_per_vox
        ret = self.raytrace(rays, samples_per_vox, grid, decoders, test = test)

        mask = (gt_depth > 0)
        if dmax > 0:
            mask = mask == (gt_depth < dmax)
        loss = {}
        unc = {}

        occ_depth = ret['depth_occ']
        occ_depth = occ_depth*self.scale
        loss['depth_occ'] = (torch.abs(occ_depth.squeeze() - gt_depth) * mask).mean()

        color = ret['color']
        loss['rgb'] = F.mse_loss(color, gt_color.float())

        if self.smooth:
            loss['smooth'] = ret['smooth_loss']

        # loss['unc'] = ret['uncertainty_occ'].mean()

        return loss


    def nef(self, coords, grid, decoders, view_dirs = None, lod_idx=None):

        shape = coords.shape

        if self.pos_embed:
            embedder = decoders['embedder']
            e = embedder(coords)

        occ_decoder = decoders['occ']
        rgb_decoder = decoders['rgb']

        if lod_idx is None:
            lod_idx = grid.num_lods - 1

        if len(shape) == 3:
            coords = coords.reshape(-1, 3)

        feats = grid.interpolate(coords, lod_idx, return_coeffs = False)
        

        ret = {}

        ret['feats'] = feats

        if self.pos_embed:
            occs = occ_decoder(torch.cat([feats[:, :self.occ_dim], e], dim = -1)).float()
            rgb = rgb_decoder(torch.cat([feats[:, self.rgb_start:self.rgb_start + self.rgb_dim], e], dim = -1)).float()
        else:
            occs = occ_decoder(feats[:, :self.occ_dim]).float()
            rgb = rgb_decoder(feats[:, self.rgb_start:self.rgb_start + self.rgb_dim]).float()
        if len(shape) == 3:
            occs = occs.reshape(shape[0], shape[1], 1)
        ret['occ'] = occs


        if len(shape) == 3:
            rgb = rgb.reshape(shape[0], shape[1], 3)
        ret['rgb'] = rgb

        return ret
    
    def get_smooth_loss(self, grid, points, feats1):
        smoothness_std = self.smooth_length/self.scale

        w = torch.rand([points.shape[0], 3], device = self.device)
        w /= w.norm(dim = -1, keepdim=True)
        length = torch.rand([points.shape[0], 1], device = self.device) * smoothness_std
        points2 = points + w * length

        lod_idx = grid.num_lods - 1
        feats2 = grid.interpolate(points2, lod_idx, return_coeffs = False)
        
        smooth_loss = (feats2 - feats1).norm(dim = -1).mean()
        # query_points = points.requires_grad_(True)

        # grads = self.compute_grads(grid, embedder, occ_decoder, query_points)

        # n = F.normalize(grads, dim=-1)
        # u = F.normalize(n[...,[1,0,2]] * torch.tensor([1., -1., 0.], device=n.device), dim=-1)
        # v = torch.cross(n, u, dim=-1)
        # phi = torch.rand(list(grads.shape[:-1]) + [1], device=grads.device) * 2. * np.pi
        # w = torch.cos(phi) * u + torch.sin(phi) * v
        # points2 = points + w * smoothness_std
        # query_points2 = points2.requires_grad_(True)
        # grads2 = self.compute_grads(grid, embedder, occ_decoder, query_points2)

        # smooth_loss = (grads - grads2).norm(dim=-1).mean()
        # smooth_loss = F.mse_loss(grads, grads2)



        return smooth_loss


    def compute_grads(self, grid, embedder, occ_decoder, query_points):
        lod_idx = grid.num_lods - 1
        grad, = torch.autograd.grad([occ_decoder(torch.cat([grid.interpolate(query_points, lod_idx, return_coeffs = False)[:, :self.occ_dim], embedder(query_points)], dim = -1))], [query_points], [torch.ones([query_points.shape[0],1], device = self.device)], create_graph=True)
        return grad


    @torch.no_grad()
    def query_occ(self, coords, grid, embedder, occ_decoder):
        shape = coords.shape
        lod_idx = grid.num_lods - 1
        # lod_idx = 0
        result = grid.blas.query(coords.reshape(-1, 3), grid.active_lods[lod_idx])#, with_parents=True)
        
        mask = result.pidx<0
        
        mask = mask.reshape(shape[:-1])
        
        coords = coords.reshape(-1,3)
        
        feats = grid.interpolate(coords, lod_idx)

        if self.pos_embed:
            e = embedder(coords)

            occs = occ_decoder(torch.cat([feats[:, :self.occ_dim], e], dim = -1)).reshape(*shape[:-1], 1) #/self.scale
        else:
            occs = occ_decoder(feats[:, :self.occ_dim]).reshape(*shape[:-1], 1)
         
        occs[mask] = -1 #torch.abs(sdfs[mask])
        return occs #.squeeze().cpu().numpy()#, ~mask.cpu().numpy()   

    @torch.no_grad()
    def query_rgb(self, coords, grid, embedder, rgb_decoder):
        lod_idx = grid.num_lods - 1
        shape = coords.shape
        feats = grid.interpolate(coords, lod_idx, return_coeffs = False)
        if self.pos_embed:
            e = embedder(coords)
            rgb = rgb_decoder(torch.cat([feats[:, self.rgb_start:self.rgb_start + self.rgb_dim], e], dim = -1), None)
        else:
            rgb = rgb_decoder(feats[:, self.rgb_start:self.rgb_start + self.rgb_dim], None)
        rgb = torch.sigmoid(rgb)
        return rgb