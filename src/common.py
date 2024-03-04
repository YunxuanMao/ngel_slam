import numpy as np
import torch
import scipy
import torch.nn.functional as F
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def getc2wError(c2w1, c2w2):
    P = c2w1[:3, :3]
    Q = c2w2[:3, :3]
    t1 = c2w1[:3, 3]
    t2 = c2w2[:3, 3]
    R = P @ Q.T
    theta = (torch.trace(R)-1)/2
    return torch.abs(torch.arccos(theta)*(180/torch.pi)), torch.abs(torch.norm(t1-t2))

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def get_cam_dir(H, W, fx, fy, cx, cy):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
    dirs = dirs.reshape(H, W, 1, 3)
    return dirs



def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def select_uv(i, j, n, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    # depth = depth.reshape(-1)
    # color = color.reshape(-1, 3)
    # depth = depth[indices]  # (n)
    # color = color[indices]  # (n,3)
    # if sem is not None:
    #     sem = sem.reshape(-1)
    #     sem = sem[indices]
    return i, j, indices # depth, color, sem


def get_sample_uv(H0, H1, W0, W1, n, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    # depth = depth[H0:H1, W0:W1]
    # color = color[H0:H1, W0:W1]
    # if sem is not None:
    #     sem = sem[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).cuda(), torch.linspace(H0, H1-1, H1-H0).cuda())
    i = i.t()  # transpose
    j = j.t()
    # i, j, depth, color, sem = select_uv(i, j, n, depth, color, sem, device=device)
    i, j, indices = select_uv(i, j, n, device=device)
    return i, j, indices #, depth, color, sem


def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    i, j, indices = get_sample_uv(
        H0, H1, W0, W1, n, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    
    return rays_o, rays_d, indices


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def raw2outputs_nerf_color(raw, z_vals, rays_d, occupancy=False, device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]
    if occupancy:
        raw[..., 3] = torch.sigmoid(10*raw[..., -1])
        alpha = raw[..., -1]
    else:
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
    return depth_map, depth_var, rgb_map, weights


def get_rays_all(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    # i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    # i = i.t()  # transpose
    # j = j.t()
    # dirs = torch.stack(
    #     [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
    # dirs = dirs.reshape(H, W, 1, 3)
    dirs = get_cam_dir(H, W, fx, fy, cx, cy)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p

def get_points_from_depth(depth, c2w, fx, fy, cx, cy, dmax = 100):
    intrinsic = torch.FloatTensor( [[fx, 0 , cx], [0, fy, cy], [0, 0, 1]] )
    intrinsic_inv = intrinsic.inverse()
    depth_mask = (depth > 0) == (depth < dmax)
    v_mask, u_mask = torch.where(depth_mask)
    depth_z = depth[depth_mask]
    depth_uv = torch.stack([u_mask, v_mask, torch.ones_like(u_mask)]).float()
    points = depth_z * ( intrinsic_inv @ depth_uv )
    ones = torch.ones_like(points[0, :]).unsqueeze(0)
    homo_points = torch.cat((points, ones))
    points_world = (c2w @ homo_points).permute(1,0)[:, :3]

    return points_world

def get_pc_from_depth(depth_list, c2w_list, fx, fy, cx, cy, device, dmax = 100):
    intrinsic = torch.FloatTensor( [[fx, 0 , cx], [0, fy, cy], [0, 0, 1]] ).to(device)
    intrinsic_inv = intrinsic.inverse()
    points_all = []
    for i in range(len(depth_list)):
        depth = depth_list[i]
        c2w = c2w_list[i].clone()
        # c2w[:3, 1] *= -1
        # c2w[:3, 2] *= -1
        depth_mask = (depth > 0) == (depth < dmax)
        v_mask, u_mask = torch.where(depth_mask)
        depth_z = depth[depth_mask]
        depth_uv = torch.stack([u_mask, v_mask, torch.ones_like(u_mask)]).float()
        points = depth_z * ( intrinsic_inv @ depth_uv )
        ones = torch.ones_like(points[0, :]).unsqueeze(0).to(device)
        homo_points = torch.cat((points, ones))
        points_world = (c2w @ homo_points).permute(1,0)[:, :3]
        points_all.append(points_world)
    points = torch.cat(points_all)
    # bounds = get_points_bounds(points)
    # lengths = bounds[:,1] - bounds[:,0] + margin*2
    # scale = torch.max(lengths)
    # origin = (bounds[:,1] + bounds[:,0])/2
    

    return points#, origin.to(device), scale.to(device)


def get_points_bounds(p):
    bounds = torch.tensor( [ [torch.min(p[:,0]), torch.max(p[:,0])],
                         [torch.min(p[:,1]), torch.max(p[:,1])],
                         [torch.min(p[:,2]), torch.max(p[:,2])] ])
    return bounds
    
def viridis_cmap(gray: np.ndarray):
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)

def crop_pc(points_all, bound):
    # crop by bbox
    x_crop = torch.logical_and(bound[0][0] < points_all[:, 0], points_all[:, 0] < bound[0][1])
    y_crop = torch.logical_and(bound[1][0] < points_all[:, 1], points_all[:, 1] < bound[1][1])
    z_crop = torch.logical_and(bound[2][0] < points_all[:, 2], points_all[:, 2] < bound[2][1])
    mask = torch.logical_and(torch.logical_and(x_crop, y_crop), z_crop)
    return points_all[mask]


@torch.no_grad()
def create_virtual_gt_with_linear_assignment(labels_gt, predicted_scores):
    labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]] # num_labels
    predicted_probabilities = torch.softmax(predicted_scores, dim=-1) # [n_ray, n_labels]
    cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]]) # [n_labels, n_labels]
    for lidx, label in enumerate(labels):
        cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
    assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
    new_labels = torch.zeros_like(labels_gt)
    for aidx, lidx in enumerate(assignment[0]):
        new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
    return new_labels

def count_parameters(l):
    return sum(p.float().element_size() * p.float().nelement() for p in l) / 1024 / 1024

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:f}MB'.format(all_size))
    return all_size

def qp2tm(quat, trans):
    '''
    quaternion pose to transform matrix
    '''
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor(R.from_quat(quat).as_matrix())
    c2w[:3, 3] = torch.tensor(trans)

    return c2w