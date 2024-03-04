import torch
# from torch.searchsorted import searchsorted
from kornia import create_meshgrid
from scipy.spatial.transform import Rotation as R

def rays_wm2wc(rays_o, rays_d):
    # wl2wc = torch.FloatTensor([[0,-1,0],[0,0,-1],[1,0,0]])

    r4 = R.from_euler('zxz', [-60,  -90,  -30], degrees=True) #0,  -90,  -60
    wl2wc = torch.FloatTensor(r4.as_matrix()).to(rays_d.device)

    rays_o_wc = rays_o
    rays_d_wc = rays_d @ wl2wc.T
    return rays_o_wc, rays_d_wc

def get_new_grid_frame(t, w2g, grid_revolution):
    # grid_point = t @  w2g[:3,:3].T + w2g[:3,3]
    # point_idx = grid_point // (2/grid_revolution) 
    
    point_idx = t // (2/(w2g[0,0]*grid_revolution)) 

    return point_idx

def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape) # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

def get_rays_HW(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)
    return rays_o, rays_d

def get_rays_Cam(directions):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = torch.FloatTensor([0,0,0]).expand(rays_d.shape) # (H, W, 3)
    return rays_o, rays_d

def get_rays_Grid(ray, w2g):
    rays = ray.clone()
    # Rotate ray directions from camera coordinate to the world coordinate
    origins = rays[:,:,0:3]
    directions = rays[:,:,3:6]
    # normals = rays[:,:,6:9]
    rays[:,:,3:6] = directions @ w2g[:3, :3].T # (H, W, 3)
    rays[:,:,0:3] = origins @  w2g[:3,:3].T + w2g[:3,3]
    # rays[:,:,6:9] = normals @  (w2g[:3,:3]/w2g[0,0]).T
    # # rays[:,:,0:3] = (w2g @ torch.hstack((origins, torch.ones_like(origins[:,:1]))).T)[:3,:].T # (H, W, 3)
    # # return rays_o[:3,:].T, rays_d
    return rays

def get_rays_sfm(ray, scale, origin):
    rays = ray.clone()
    origins = rays[:,:,0:3]
    # directions = rays[:,:,3:6]
    # rays[:,:,3:6] = directions @ w2g[:3, :3].T # (H, W, 3)
    rays[:,:,0:3] = (origins - origin) / scale
    return rays


def get_rays_uv(derection, K):
    uv = derection @ K.T
    uv = uv[:,:2] / uv[:,2:]
    return uv

def sample_depth_points(rays_o, rays_d, near, far, N_samples):
    # Sample depth points
    N_rays = rays_o.shape[0]
    z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device) # (N_samples)
    z_vals = near * (1-z_steps) + far * z_steps
    z_vals = z_vals.expand(N_rays, N_samples)
    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    return xyz_coarse_sampled, z_vals

def sample_points_by_z(rays_o, rays_d, z_vals):
    xyz_sampled = rays_o.unsqueeze(1) + \
                  rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)
    # return torch.reshape(xyz_sampled, ((imgnum,-1,xyz_sampled.size(1),xyz_sampled.size(2))))
    return xyz_sampled

def sample_points_by_c2w(ray_directions, near, far, N_sample, c2w):
    device = c2w.device
    rays_o, rays_d = get_rays(ray_directions, c2w=c2w)
    xyz_sample, z_vals = sample_depth_points(rays_o, rays_d, near, far, N_sample)
    xyz_sample = xyz_sample.view(-1,3)
    return xyz_sample, z_vals

def sample_points_dirs_by_c2w(ray_directions, near, far, N_sample, c2w):
    device = c2w.device
    rays_o, rays_d = get_rays(ray_directions, c2w=c2w)
    xyz_sample, z_vals = sample_depth_points(rays_o, rays_d, near, far, N_sample)
    xyz_sample = xyz_sample.view(-1,3)
    rays_d = rays_d[:,None,:]
    rays_d = rays_d.expand(rays_d.shape[0], N_sample,rays_d.shape[2]).reshape(-1,3)
    return xyz_sample, z_vals, rays_d

def trans_points(points, T):
    return (T[:3,:3] @ points.t()).t() + T[:3,3]

# Trans dir do not need to add t
def trans_dirs(points, T):
    new_dir = (T[:3,:3] @ points.t()).t()
    # trans dir to unit
    unit_dir = new_dir / torch.sqrt((new_dir**2).sum(1))[:,None]
    return unit_dir

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

def get_ray_directions(H, W, fx, fy, cx, cy, device, rescale =1):
    if rescale != 1:
        H = int(H/rescale)
        W = int(W/rescale)
        
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1).to(device=device) # (H, W, 3)
    return directions