# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import logging as log
from typing import Dict, Set, Any, Type
import torch
import torch.nn as nn
from torch.autograd import Variable
import wisp.ops.spc as wisp_spc_ops
from wisp.models.grids import BLASGrid
import kaolin.ops.spc as spc_ops
from wisp.accelstructs import BaseAS, OctreeAS, ASRaymarchResults
from wisp.ops.spc.processing import dilate_points
import copy
import time
import numpy as np

class OctreeGrid(BLASGrid):
    """This is a multiscale feature grid where the features are defined on the BLAS, the octree.
    """

    def __init__(
        self,
        pointcloud,
        feature_dim         : int,
        base_lod            : int,
        num_lods            : int          = 1,
        interpolation_type  : str          = 'linear',
        multiscale_type     : str          = 'cat',
        feature_std         : float        = 0.0,
        feature_bias        : float        = 0.0,
        dilate              : int          = 0,
    ):
        """Initialize the octree grid class.

        Args:
            accelstruct: Spatial acceleration structure which tracks the occupancy state of this grid.
                         Used to speed up spatial queries and ray tracing operations.
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid. This is the lowest LOD of the SPC octree
                            for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.
            sample_tex (bool): If True, will also sample textures and store it in the accelstruct.
            num_samples (int): The number of samples to be generated on the mesh surface.
        Returns:
            (void): Initializes the class.
        """
        max_lod = self.max_octree_lod(base_lod, num_lods)
        blas = OctreeAS.from_pointcloud(pointcloud, level=max_lod, dilate = dilate)

        super().__init__(blas)

        self.feature_dim = feature_dim
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type
        

        self.feature_std = feature_std
        self.feature_bias = feature_bias

        # List of octree levels which are optimized.
        self.active_lods = [self.base_lod + x for x in range(self.num_lods)]
        self.max_lod = OctreeGrid.max_octree_lod(self.base_lod, self.num_lods)

        log.info(f"Active LODs: {self.active_lods}")    # TODO(operel): move into trainer

        if self.num_lods > 0:
            self.init_feature_structure()

        self.morton = self.get_morton(pointcloud, dilate)

    def reset(self, pointcloud, dilate = 0):
        self.blas = OctreeAS.from_pointcloud(pointcloud, level=self.max_lod, dilate = dilate)

        self.morton = self.get_morton(pointcloud, dilate)
        self.init_feature_structure()

    @staticmethod
    def max_octree_lod(base_lod, num_lods) -> int:
        """
        Returns:
            (int): The highest level-of-detail maintaining features in this Octree grid.
        """
        return base_lod + num_lods - 1


    @classmethod
    def from_pointcloud(cls,
                        pointcloud: torch.FloatTensor,
                        feature_dim: int,
                        base_lod: int,
                        num_lods: int = 1,
                        interpolation_type: str = 'linear',
                        multiscale_type: str = 'cat',
                        feature_std: float = 0.0,
                        feature_bias: float = 0.0,
                        dilate: int = 0) -> OctreeGrid:
        """Builds the OctreeGrid, initializing it with a pointcloud.
        The cells occupancy will be determined by points occupying the octree cells.

        Args:
            pointcloud (torch.FloatTensor): 3D coordinates of shape [num_coords, 3] in normalized space [-1, 1].
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of the  octree for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            interpolation_type (str): The type of interpolation function used when querying features on the grid.
                                      'linear' - uses trilinear interpolation from nearest 8 nodes.
                                      'closest' - uses feature from nearest grid node.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.

        Returns:
            (OctreeGrid): A new instance of an OctreeGrid with occupancy initialized from the pointcloud.
        """
        max_lod = OctreeGrid.max_octree_lod(base_lod, num_lods)
        
        blas = OctreeAS.from_pointcloud(pointcloud, level=max_lod, dilate = dilate)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias)
        grid.morton = grid.get_morton(pointcloud, dilate)
        return grid


    def init_feature_structure(self):
        """ Initializes everything related to the features stored in the codebook octree structure. """

        # Assumes the occupancy structure have been initialized (the BLAS: Bottom Level Accelerated Structure).
        # Build the trinket structure
        if self.interpolation_type in ['linear']:
            self.points_dual, self.pyramid_dual, self.trinkets, self.parents = \
                wisp_spc_ops.make_trilinear_spc(self.blas.points, self.blas.pyramid)
            log.info("Built dual octree and trinkets")
        
        # Build the pyramid of features
        fpyramid = []
        for al in self.active_lods:
            if self.interpolation_type == 'linear':
                fpyramid.append(self.pyramid_dual[0,al])
            elif self.interpolation_type == 'closest':
                fpyramid.append(self.blas.pyramid[0,al])
            else:
                raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")
        self.num_feat = sum(fpyramid).long()
        log.info(f"# Feature Vectors: {self.num_feat}")

        self.features = []
        # self.features = nn.ParameterList([])
        for i in range(len(self.active_lods)):
            fts = torch.zeros(fpyramid[i], self.feature_dim[i]) + self.feature_bias
            fts += torch.randn_like(fts) * self.feature_std
            # fts = Variable(fts.cuda(), requires_grad = True)
            # self.features.append(nn.Parameter(fts).cuda())
            self.features.append(fts.cuda())
        
        self.parents_dual = spc_ops.unbatched_make_parents(self.points_dual, self.pyramid_dual)
        
        log.info(f"Pyramid:{self.blas.pyramid[0]}")
        log.info(f"Pyramid Dual: {self.pyramid_dual[0]}")

    def update(self, points, device, dilate = 0):
        # print('Updating OctreeGrid...')
        stime = time.time()
        pidx = self.query(points).pidx
        new_points = points[pidx == -1]
        level = self.max_octree_lod(self.base_lod, self.num_lods)

        # update octree
        if len(new_points) == 0:
            return
        new_morton = self.get_morton(new_points, dilate)
        self.morton, _ = torch.sort(torch.cat((self.morton, new_morton)).contiguous())
        
        points = spc_ops.morton_to_points(self.morton.contiguous())
        octree = spc_ops.unbatched_points_to_octree(points, level, sorted=True)
        # new_points = spc_ops.morton_to_points(new_morton.contiguous())
        # new_octree = spc_ops.unbatched_points_to_octree(new_points, level, sorted=True)
        # octree = torch.cat((self.blas.octree, new_octree)).to(device)

        # update blas
        # old_blas = copy.deepcopy(self.blas)
        new_blas = OctreeAS(octree)
        # self.blas = OctreeAS(octree)
        
        # self.blas = new_blas
        # update feature
        # Build the trinket structure
        if self.interpolation_type in ['linear']:
            points_dual, pyramid_dual, trinkets, parents = \
                wisp_spc_ops.make_trilinear_spc(new_blas.points, new_blas.pyramid)
            # log.info("Built dual octree and trinkets")
            # old_points_mask = points_dual.unsqueeze(1).eq(self.points_dual).all(-1).any(-1)
            
            old_points_mask_dual = torch.ones(len(points_dual), dtype=torch.bool).cuda()
            combined = torch.cat([points_dual, self.points_dual])
            unique, inverse, counts, index = unique_idx(combined, dim = 0)
            new_points_mask = index[counts == 1]
            old_points_mask_dual[new_points_mask] = False

            # old_points_mask = torch.zeros(len(points_dual), dtype=torch.bool).cuda()
            # combined = torch.cat([new_blas.points, self.blas.points])
            # unique, inverse, counts, index = unique_idx(combined, dim = 0)
            # old_points_mask_blas = index[counts > 1]
            # old_points_mask[trinkets[old_points_mask_blas].unique().long()] = True

            # new_pidx = self.query(new_points, with_parents=True).pidx.reshape(-1)
            # new_pidx_dual = trinkets.index_select(0, new_pidx).long().reshape(-1)
            # old_points_mask[new_pidx_dual] = False
            
        else:
            old_points_mask = new_blas.points.unsqueeze(1).eq(self.blas.points).all(-1).any(-1)

        # Build the pyramid of features
        fpyramid = []
        self.old_points_mask_level = []
        for al in self.active_lods:
            if self.interpolation_type == 'linear':
                fpyramid.append(pyramid_dual[0,al])
                self.old_points_mask_level.append(old_points_mask_dual[pyramid_dual[1,al]:pyramid_dual[1,al+1]])
            elif self.interpolation_type == 'closest':
                fpyramid.append(new_blas.pyramid[0,al])
                self.old_points_mask_level.append(old_points_mask[new_blas.pyramid[1,al]:new_blas.pyramid[1,al+1]])
            else:
                raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")
        num_feat = sum(fpyramid).long()
        # log.info(f"# Feature Vectors: {num_feat}")

        self.new_features = []
        self.old_features = []
        features = []
        for i in range(len(self.active_lods)):
            fts = torch.zeros(fpyramid[i], self.feature_dim[i]) + self.feature_bias
            fts += torch.randn_like(fts) * self.feature_std
            fts = fts.to(device)
            fts[self.old_points_mask_level[i]] = self.features[i]
            # features[i].append(torch.zeros(fpyramid[i], self.feature_dim))
            
            # self.features[i] = nn.Parameter(fts).cuda()
            self.features[i] = fts
            # self.features[i][self.old_points_mask_level[i]].detach()
            # new_feature = Variable(self.features[i][~self.old_points_mask_level[i]].clone(), requires_grad=True)
            # old_feature = Variable(self.features[i][self.old_points_mask_level[i]].clone(), requires_grad=True)
            # self.new_features.append(new_feature)
            # self.old_features.append(old_feature)
            
        
        self.points_dual = points_dual
        self.pyramid_dual = pyramid_dual
        self.trinkets = trinkets
        self.parents = parents
        self.blas = new_blas
        self.parents_dual = spc_ops.unbatched_make_parents(points_dual, pyramid_dual)
        # log.info(f"Pyramid:{self.blas.pyramid[0]}")
        # log.info(f"Pyramid Dual: {self.pyramid_dual[0]}")
        etime = time.time()
        # print(f' Update done! Time: {etime - stime}')
        

    def get_mask_from_rays(self, rays):
        pidx_list = []
        lod = self.max_lod
        # last lod
        result = self.raytrace(rays, lod)
        pidx = result.pidx

        pidx_dual = self.trinkets.index_select(0, pidx).long().reshape(-1)
        mask_max = torch.zeros(self.pyramid_dual[0, lod], dtype=torch.bool).cuda()
        mask_max[pidx_dual] = True
        pidx_dual += self.pyramid_dual[1, lod]

        # other lods
        mask_list = [mask_max]
        for i in range(len(self.active_lods) - 1):
            lod -= 1
            mask = torch.zeros(self.pyramid_dual[0, lod], dtype=torch.bool).cuda()
            pidx_dual = self.parents_dual.index_select(0, pidx_dual).long().reshape(-1)
            mask[pidx_dual - self.pyramid_dual[1, lod]] = True
            mask_list.insert(0, mask)
        
        return mask_list


    def freeze(self):
        """Freezes the feature grid.
        """
        for lod_idx in range(self.num_lods):
            self.features[lod_idx].requires_grad_(False)

    def _index_features(self, feats, idx):
        """Internal function. Returns the feats based on indices.

        This function exists to override in case you want to implement a different method of indexing,
        i.e. a differentiable one as in Variable Bitrate Neural Fields (VQAD).

        Args:
            feats (torch.FloatTensor): tensor of feats of shape [num_feats, feat_dim]
            idx (torch.LongTensor): indices of shape [num_indices]

        Returns:
            (torch.FloatTensor): tensor of feats of shape [num_indices, feat_dim]
        """
        return feats[idx.long()]

    def _interpolate(self, coords, feats, pidx, lod_idx):
        """Interpolates the given feature using the coordinates x. 

        This is a more low level interface for optimization.

        Inputs:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            feats (torch.FloatTensor): feats of shape [num_feats, feat_dim]
            pidx (torch.LongTensor) : point_hiearchy indices of shape [batch]
            lod_idx (int) : int specifying the index fo ``active_lods``
        Returns:
            (torch.FloatTensor): acquired features of shape [batch, num_samples, feat_dim]
        """
        batch, num_samples = coords.shape[:2]
        lod = self.active_lods[lod_idx]

        if self.interpolation_type == 'linear':
            fs = spc_ops.unbatched_interpolate_trilinear(
                coords, pidx.int(), self.blas.points, self.trinkets.int(),
                feats.half(), lod).float()
        elif self.interpolation_type == 'closest':
            fs = self._index_features(feats, pidx.long()-self.blas.pyramid[1, lod])[...,None,:]
            fs = fs.expand(batch, num_samples, feats.shape[-1])
       
        # Keep as backup
        elif self.interpolation_type == 'trilinear_old':
            corner_feats = feats[self.trinkets.index_select(0, pidx).long()]
            coeffs = spc_ops.coords_to_trilinear_coeffs(coords, self.points.index_select(0, pidx)[:,None].repeat(1, coords.shape[1], 1), lod)
            fs = (corner_feats[:, None] * coeffs[..., None]).sum(-2)

        else:
            raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")
        
        return fs

    def interpolate(self, coords, lod_idx, return_coeffs = False):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
            lod_idx  (int): int specifying the index to ``active_lods``
            features (torch.FloatTensor): features to interpolate. If ``None``, will use `self.features`.

        Returns:
            (torch.FloatTensor): interpolated features of shape
            [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        # Remember desired output shape, and inflate to (batch, num_samples, 3) format
        output_shape = coords.shape[:-1]
        if coords.ndim < 3:
            coords = coords[:, None]    # (batch, 3) -> (batch, num_samples, 3)
        
        num_coords = coords.shape[0] * coords.shape[1]

        if lod_idx == 0:
            query_results = self.blas.query(coords.reshape(-1, 3), self.active_lods[lod_idx], with_parents=False)
            pidx = query_results.pidx
            feat = self._interpolate(coords.reshape(-1, 1, 3), self.features[0], pidx, 0)
            return feat.reshape(*output_shape, feat.shape[-1])
        else:
            feats = []
            feats_color = []
            
            # In the multiscale case, the raytrace _currently_  does not return multiscale indices.
            # As such, no matter what the pidx will be recalculated to get the multiscale indices.
            num_feats = lod_idx + 1
            num_feats_color = 0
            
            # This might look unoptimal since it assumes that samples are _not_ in the same voxel.
            # This is the correct assumption here, because the point samples are from the base_lod,
            # not the highest LOD.
            query_results = self.blas.query(coords.reshape(-1, 3), self.active_lods[lod_idx], with_parents=True)
            pidx = query_results.pidx[...,self.base_lod:]
            pidx = pidx.reshape(-1, coords.shape[1], num_feats)
            pidx = torch.split(pidx, 1, dim=-1)
            
            # list of [batch, num_samples, 1]

            max_dim = np.max(self.feature_dim)

            feats = torch.zeros([num_feats, num_coords, max_dim]).cuda()

            for i in range(num_feats):
                feat = self._interpolate(
                    coords.reshape(-1, 1, 3), self.features[i], pidx[i].reshape(-1), i)[:,0]
                feats[i, :, :self.feature_dim[i]] = feat[:, :self.feature_dim[i]]
                # if feat.shape[1] > self.feature_dim[0]:
                #     feats_color.append(feat[:, self.feature_dim[0]:])
                #     num_feats_color += 1
            
            # feats = torch.cat(feats, dim=-1)
            if len(feats_color) > 0:
                feats_color = torch.cat(feats_color, dim=-1)
            n_cat = self.num_lods

            if self.multiscale_type == 'cat':
                feats = torch.cat([feats, feats_color], dim = -1)
                return feats.reshape(*output_shape, np.sum(self.feature_dim))

            if self.multiscale_type == 'sum':
                # feats = feats.reshape(*feats.shape[:-1], num_feats, self.feature_dim[0])
                # feats = feats.sum(-2)
                # feats_color = feats_color.reshape(*feats.shape[:-1], num_feats_color, self.feature_dim[-1] - self.feature_dim[0])
                # feats_color = feats_color.sum(-2)
                # feats = torch.cat([feats, feats_color], dim = -1)
                # return feats.reshape(*output_shape, self.feature_dim[-1])
                return feats.sum(0)

            # if return_coeffs:
            #     unit_coords = torch.tensor([[0.,0.,0.], [0.,0.,1.],[0.,1.,0.],[0.,1.,1.],[1.,0.,0.],[1.,0.,1.],[1.,1.,0.],[1.,1.,1.]]).cuda()
            #     lod = self.active_lods[0]
            #     pidx = pidx[0].reshape(-1)
            #     coeffs = torch.zeros(pidx.shape[0], 8).cuda()
            #     coeffs[:, -1] = 1
            #     coeffs[pidx != -1] = spc_ops.coords_to_trilinear_coeffs(coords.reshape(-1, 3)[pidx != -1], self.blas.points.index_select(0, pidx[pidx != -1]), lod)
            #     coeffs = coeffs @ unit_coords
            #     return feats.reshape(*output_shape, self.feature_dim*n_cat), coeffs.reshape(*output_shape, 3)

            return feats.reshape(*output_shape, np.sum(self.feature_dim))
        
        # relative_coords = coeffs * 

    def raymarch(self, rays, raymarch_type, num_samples, level=None) -> ASRaymarchResults:
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=level)

    def supported_blas(self) -> Set[Type[BaseAS]]:
        """ Returns a set of bottom-level acceleration structures this grid type supports """
        return {OctreeAS}

    def name(self) -> str:
        return "Octree Grid"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        parent_properties = super().public_properties()
        properties = {
            "Feature Dims": self.feature_dim,
            "Total LODs": self.max_lod,
            "Active feature LODs": [str(x) for x in self.active_lods],
            "Interpolation": self.interpolation_type,
            "Multiscale aggregation": self.multiscale_type
        }
        return {**parent_properties, **properties}

    def get_morton(self, pointcloud, dilate):
        level = self.max_octree_lod(self.base_lod, self.num_lods)
        points = spc_ops.quantize_points(pointcloud.contiguous().cuda(), level)
        points = torch.unique(points.contiguous(), dim=0)

        for i in range(dilate):
            points = dilate_points(points, level)

        unique, unique_keys, unique_counts = torch.unique(points.contiguous(), dim=0,
                                                        return_inverse=True, return_counts=True)

        morton, keys = torch.sort(spc_ops.points_to_morton(unique.contiguous()).contiguous())

        return morton

def unique_idx(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse+decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index