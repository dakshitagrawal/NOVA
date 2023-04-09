import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from run_nerf_helpers import *
from utils.flow_utils import flow_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rgb_weights_after_flow(
    ret,
    flow_points,
    viewdirs,
    network_fn_d,
    network_query_fn_d,
    z_vals,
    rays_d,
    weights_ref,
    raw_noise_std,
    key,
    cam_order,
):
    raw_values = []
    for idx, dy_idx in enumerate(cam_order[1:]):
        raw_values.append(
            network_query_fn_d(
                flow_points[idx], viewdirs[idx + 1], network_fn_d[dy_idx - 1]
            )
        )
    raw_values = torch.stack(raw_values, dim=0)
    raw_rgba = raw_values[..., :4]
    sceneflow_b = raw_values[..., 4:7]
    sceneflow_f = raw_values[..., 7:10]
    rgb_map_d, weights_d, _, _ = raw2outputs_d(raw_rgba, z_vals, rays_d, raw_noise_std)
    ret[f"rgb_map_d{key}"] = rgb_map_d
    if key == "_b" or key == "_f":
        ret[f"acc_map_d{key}"] = torch.abs(torch.sum(weights_d - weights_ref, -1))
        ret[f"sceneflow{key}_f"] = sceneflow_f
        ret[f"sceneflow{key}_b"] = sceneflow_b
    return ret


def batchify_rays(t, chain_5frames, rays_flat, chunk=1024 * 16, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[1], chunk):
        ret = render_rays(t, chain_5frames, rays_flat[:, i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {
        k: torch.cat(all_ret[k], 0) if "full" in k else torch.cat(all_ret[k], 1)
        for k in all_ret
    }
    return all_ret


def render(
    t,
    chain_5frames,
    H,
    W,
    focal,
    focal_render=None,
    chunk=1024 * 16,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, num_obj, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [num_obj, 3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [num_obj, 3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o = []
        rays_d = []
        for obj_pose in c2w:
            if focal_render is not None:
                # Render full image using different focal length for dolly zoom. Inference only.
                ray_o, ray_d = get_rays(H, W, focal_render, obj_pose)
            else:
                ray_o, ray_d = get_rays(H, W, focal, obj_pose)
            rays_o.append(ray_o)
            rays_d.append(ray_d)
        rays_o = torch.stack(rays_o, dim=0)
        rays_d = torch.stack(rays_d, dim=0)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    num_obj = len(rays_o)
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            raise NotImplementedError
        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [num_obj, -1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o_shape = rays_o.shape
        rays_d_shape = rays_d.shape
        rays_o, rays_d = ndc_rays(H, W, focal, 1.0, rays_o, rays_d)
        assert rays_o.shape == rays_o_shape
        assert rays_d.shape == rays_d_shape

    # Create ray batch
    rays_o = torch.reshape(rays_o, [num_obj, -1, 3]).float()
    rays_d = torch.reshape(rays_d, [num_obj, -1, 3]).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    assert len(t) == len(rays)
    # assert rays.shape == torch.Size([*sh[:-1], 11 if use_viewdirs else 8])

    # Render and reshape
    all_ret = batchify_rays(t, chain_5frames, rays, chunk, **kwargs)
    for k in all_ret:
        if "full" in k:
            k_sh = list(sh[1:-1]) + list(all_ret[k].shape[1:])
        else:
            k_sh = list([-1]) + list(sh[1:-1]) + list(all_ret[k].shape[2:])

        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def render_path(
    render_poses,
    time2render,
    hwf,
    chunk,
    render_kwargs,
    savedir=None,
    flows_gt_f=None,
    flows_gt_b=None,
    focal2render=None,
):
    """Render frames.

    Args:
      render_poses: array of shape [num_frame, 3, 4]. Camera-to-world transformation matrix of each frame.
      time2render: array of shape [num_frame]. Time of each frame.
      hwf: list. [Height of image in pixels, Width of image in pixels, Focal length of pinhole camera]
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      render_kwargs: dictionary. args for the render function.
      savedir: string. Directory to save results.
      focal2render: list. Only used to perform dolly-zoom.
    Returns:
      ret_dict: dictionary. Final and intermediate results.
    """
    H, W, focal = hwf
    pretrain = render_kwargs.get("pretrain")

    ret_dict = {}
    rgbs_full = []
    rgbs_obj = []
    depths_full = []
    dynamicness = []
    flows_f = []
    flows_b = []

    grid = np.stack(
        np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing="xy",
        ),
        -1,
    )
    grid = torch.Tensor(grid)
    time_curr = time.time()
    for i, c2w in enumerate(render_poses):
        t = time2render[i]
        pose = c2w[:, :3, :4]
        print(i, time.time() - time_curr)
        time_curr = time.time()

        if focal2render is None:
            # Normal rendering.
            ret = render(
                t, False, H, W, focal, chunk=1024 * 32, c2w=pose, **render_kwargs
            )
        else:
            # Render image using different focal length.
            ret = render(
                t,
                False,
                H,
                W,
                focal,
                focal_render=focal2render[i],
                chunk=1024 * 32,
                c2w=pose,
                **render_kwargs,
            )

        rgbs_full.append(ret["rgb_map_full"].cpu().numpy())
        depths_full.append(ret["depth_map_full"].cpu().numpy())
        if not pretrain:
            rgbs_obj.append(ret["rgb_map_obj"].cpu().numpy())
            dynamicness.append(ret["dynamicness_map_obj"].cpu().numpy())

        # TODO
        # if flows_gt_f is not None:
        #     # Reconstruction. Flow is caused by both changing camera and changing time.
        #     pose_f = render_poses[min(i + 1, int(len(render_poses)) - 1), :, :3, :4]
        #     pose_b = render_poses[max(i - 1, 0), :, :3, :4]
        # else:
        #     # Non training view-time. Flow is caused by changing time (just for visualization).
        #     pose_f = render_poses[i, :, :3, :4]
        #     pose_b = render_poses[i, :, :3, :4]

        # # Sceneflow induced optical flow
        # induced_flow_f_ = induce_flow(
        #     H, W, focal, pose_f, ret["weights_obj"], ret["raw_pts_f"], grid[..., :2]
        # )
        # induced_flow_b_ = induce_flow(
        #     H, W, focal, pose_b, ret["weights_obj"], ret["raw_pts_b"], grid[..., :2]
        # )

        # if (i + 1) >= len(render_poses):
        #     induced_flow_f = np.zeros((H, W, 2))
        # else:
        #     induced_flow_f = induced_flow_f_.cpu().numpy()
        # if flows_gt_f is not None:
        #     flow_gt_f = flows_gt_f[i].cpu().numpy()
        #     induced_flow_f = np.concatenate((induced_flow_f, flow_gt_f), 0)
        # induced_flow_f_img = flow_to_image(induced_flow_f)
        # flows_f.append(induced_flow_f_img)

        # if (i - 1) < 0:
        #     induced_flow_b = np.zeros((H, W, 2))
        # else:
        #     induced_flow_b = induced_flow_b_.cpu().numpy()
        # if flows_gt_b is not None:
        #     flow_gt_b = flows_gt_b[i].cpu().numpy()
        #     induced_flow_b = np.concatenate((induced_flow_b, flow_gt_b), 0)
        # induced_flow_b_img = flow_to_image(induced_flow_b)
        # flows_b.append(induced_flow_b_img)

        # if i == 0:
        #     ret_dict["sceneflow_f_NDC"] = ret["sceneflow_f"].cpu().numpy()
        #     ret_dict["sceneflow_b_NDC"] = ret["sceneflow_b"].cpu().numpy()
        #     ret_dict["blending"] = ret["blending"].cpu().numpy()

        #     weights = np.concatenate(
        #         (
        #             ret["weights_d"][..., None].cpu().numpy(),
        #             ret["weights_s"][..., None].cpu().numpy(),
        #             ret["blending"][..., None].cpu().numpy(),
        #             ret["weights_full"][..., None].cpu().numpy(),
        #         )
        #     )
        #     ret_dict["weights"] = np.moveaxis(weights, [0, 1, 2, 3], [1, 2, 0, 3])

        if savedir is not None:
            rgb8 = to8b(rgbs_full[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

    ret_dict["rgbs_full"] = np.stack(rgbs_full, 0)
    ret_dict["depths_full"] = np.stack(depths_full, 0)

    if not pretrain:
        rgbs_obj = np.stack(rgbs_obj, 0)
        dynamicness = np.stack(dynamicness, 0)
        for idx in range(rgbs_obj.shape[1]):
            ret_dict[f"rgbs_obj_{idx}"] = rgbs_obj[:, idx]

        for idx in range(dynamicness.shape[1]):
            ret_dict[f"dynamicness_{idx}"] = dynamicness[:, idx]

    # ret_dict["flows_f"] = np.stack(flows_f, 0)
    # ret_dict["flows_b"] = np.stack(flows_b, 0)

    return ret_dict


def raw2outputs(rgba, blending, z_vals, rays_d, raw_noise_std, hard_blending=False):
    """Transforms model's predictions to semantically meaningful values.

    Args:
      rgba: [num_obj, num_rays, num_samples along ray, 4]. Prediction from all models.
      blending: [num_obj, num_rays, num_samples along ray]. Blending from all models.
      z_vals: [num_obj, num_rays, num_samples along ray]. Integration time.
      rays_d: [num_obj, num_rays, 3]. Direction of each ray.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. Inverse of depth map.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """
    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    N_obj, N_rays, N_samples, _ = rgba.shape
    if hard_blending:
        for bl_obj in range(1, N_obj):
            blending[0] = torch.where(
                blending[bl_obj] > 0.5, torch.zeros_like(blending[0]), blending[0]
            )
    blending = blending / (blending.sum(dim=0, keepdim=True) + 1e-8)

    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
    )  # [N_obj, N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_obj = torch.sigmoid(rgba[..., :3])  # [N_obj, N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(rgba[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_obj = raw2alpha(rgba[..., 3] + noise, dists)  # [N_obj, N_rays, N_samples]
    # TODO check theory, the alphas should be multiplied by their blending factor
    alphas_full = 1.0 - torch.prod(1.0 - alpha_obj, 0)  # [N_rays, N_samples]
    assert alphas_full.shape == torch.Size([N_rays, N_samples])

    T_obj = torch.cumprod(
        torch.cat(
            [torch.ones((*alpha_obj.shape[:-1], 1)), 1.0 - alpha_obj + 1e-10], -1
        ),
        -1,
    )[..., :-1]
    assert T_obj.shape == torch.Size([N_obj, N_rays, N_samples])

    T_full = torch.cumprod(
        torch.cat(
            [
                torch.ones((*alphas_full.shape[:-1], 1)),
                torch.prod(1.0 - alpha_obj * blending, dim=0) + 1e-10,
            ],
            -1,
        ),
        -1,
    )[..., :-1]
    assert T_full.shape == torch.Size([N_rays, N_samples])

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_obj = alpha_obj * T_obj
    assert weights_obj.shape == torch.Size([N_obj, N_rays, N_samples])
    # TODO check theory, instead of sum it should be product
    weights_full = torch.sum(alpha_obj * blending, dim=0) * T_full
    assert weights_full.shape == torch.Size([N_rays, N_samples])

    # Computed weighted color of each sample along each ray.
    rgb_map_obj = torch.sum(weights_obj[..., None] * rgb_obj, -2)
    rgb_map_full = torch.sum(
        (T_full[None] * alpha_obj * blending)[..., None] * rgb_obj, (0, -2)
    )
    assert rgb_map_obj.shape == torch.Size([N_obj, N_rays, 3])
    assert rgb_map_full.shape == torch.Size([N_rays, 3])

    # Estimated depth map is expected distance.
    depth_map_obj = torch.sum(weights_obj * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals[0], -1)
    assert depth_map_full.shape == torch.Size([N_rays])
    assert depth_map_obj.shape == torch.Size([N_obj, N_rays])

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_obj = torch.sum(weights_obj, -1)
    acc_map_full = torch.sum(weights_full, -1)
    assert acc_map_full.shape == torch.Size([N_rays])
    assert acc_map_obj.shape == torch.Size([N_obj, N_rays])

    # Computed dynamicness
    dynamicness_map_obj = torch.sum(weights_full[None] * blending, -1)
    assert dynamicness_map_obj.shape == torch.Size([N_obj, N_rays])

    return (
        rgb_map_full,
        depth_map_full,
        acc_map_full,
        weights_full,
        rgb_map_obj,
        depth_map_obj,
        acc_map_obj,
        weights_obj,
        dynamicness_map_obj,
        alpha_obj,
    )


def raw2outputs_d(raw_d, z_vals, rays_d, raw_noise_std):

    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_obj = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_obj = raw2alpha(raw_d[..., 3] + noise, dists)  # [N_rays, N_samples]

    T_obj = torch.cumprod(
        torch.cat(
            [torch.ones((*alpha_obj.shape[:-1], 1)), 1.0 - alpha_obj + 1e-10], -1
        ),
        -1,
    )[..., :-1]
    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_obj = alpha_obj * T_obj

    # Computed weighted color of each sample along each ray.
    rgb_map_obj = torch.sum(weights_obj[..., None] * rgb_obj, -2)

    # Estimated depth map is expected distance.
    depth_map_obj = torch.sum(weights_obj * z_vals, -1)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_obj = torch.sum(weights_obj, -1)

    return rgb_map_obj, weights_obj, depth_map_obj, acc_map_obj


def render_rays(
    t,
    chain_5frames,
    ray_batch,
    network_fn_d,
    network_fn_s,
    network_query_fn_d,
    network_query_fn_s,
    N_samples,
    num_img,
    pretrain=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    raw_noise_std=0.0,
    inference=False,
    cam_order=None,
    hard_blending=False,
):

    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn_d: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn_d: function used for passing queries to network_fn_d.
      N_samples: int. Number of different times to sample along each ray.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    # batch size
    N_obj = ray_batch.shape[0]
    N_rays = ray_batch.shape[1]

    # ray_batch: [N_obj, N_rays, 11]
    # rays_o:    [N_obj, N_rays, 0:3]
    # rays_d:    [N_obj, N_rays, 3:6]
    # near:      [N_obj, N_rays, 6:7]
    # far:       [N_obj, N_rays, 7:8]
    # viewdirs:  [N_obj, N_rays, 8:11]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[..., 0:3], ray_batch[..., 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[..., -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = ray_batch[..., 6:8].unsqueeze(-2)
    near, far = bounds[..., 0], bounds[..., 1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))
    z_vals = z_vals.expand([N_obj, N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.0:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_obj, N_rays, N_samples, 3]
    assert pts.shape == torch.Size([N_obj, N_rays, N_samples, 3])

    # Add the time dimension to xyz.
    pts_ref = torch.cat(
        [pts, torch.ones_like(pts[..., 0:1]) * torch.Tensor(t)[:, None, None, None]], -1
    )
    assert pts_ref.shape == torch.Size([N_obj, N_rays, N_samples, 4])
    for idx, t_value in enumerate(t):
        assert pts_ref[..., 3:][idx].unique() == torch.Tensor([t_value]), breakpoint()

    # First pass: we have the staticNeRF results
    raw_s = network_query_fn_s(pts_ref[..., :3][0], viewdirs[0], network_fn_s)
    raw_s_rgba = raw_s[..., :4]
    blending_s = raw_s[..., 4]
    # raw_s:          [N_rays, N_samples, 5]
    # raw_s_rgb:      [N_rays, N_samples, 0:3]
    # raw_s_a:        [N_rays, N_samples, 3:4]
    # raw_s_blending: [N_rays, N_samples, 4:5]

    if pretrain:
        rgb_map_obj, weights_obj, depth_map_obj, acc_map_obj = raw2outputs_d(
            raw_s_rgba[None], z_vals, rays_d, raw_noise_std
        )
        dynamicness_map_obj = torch.sum(weights_obj * blending_s[None], -1)
        ret = {
            "rgb_map_full": rgb_map_obj[0],
            "depth_map_full": depth_map_obj[0],
            "acc_map_full": acc_map_obj[0],
            "dynamicness_map_full": dynamicness_map_obj[0],
        }
        return ret

    # Second pass: we have the DyanmicNeRF results and the blending weight
    raw_d_values = []
    # for dy_idx, dynamic_nerf_model in enumerate(network_fn_d):
    cam_order = cam_order or np.arange(len(network_fn_d) + 1, dtype=np.int)
    for idx, dy_idx in enumerate(cam_order[1:]):
        raw_d_values.append(
            network_query_fn_d(
                pts_ref[idx + 1], viewdirs[idx + 1], network_fn_d[dy_idx - 1]
            )
        )
    raw_d_values = torch.stack(raw_d_values, dim=0)
    # raw_d:          [N_obj, N_rays, N_samples, 11]
    # raw_d_rgb:      [N_obj, N_rays, N_samples, 0:3]
    # raw_d_a:        [N_obj, N_rays, N_samples, 3:4]
    # sceneflow_b:    [N_obj, N_rays, N_samples, 4:7]
    # sceneflow_f:    [N_obj, N_rays, N_samples, 7:10]
    # raw_d_blending: [N_obj, N_rays, N_samples, 10:11]

    raw_d_rgba = raw_d_values[..., :4]
    raw_rgba = torch.cat([raw_s_rgba[None], raw_d_rgba], dim=0)

    blending_d = raw_d_values[..., 10]
    blending = torch.cat([blending_s[None], blending_d], dim=0)

    assert raw_s_rgba.shape == torch.Size([N_rays, N_samples, 4])
    assert raw_d_rgba.shape == torch.Size([N_obj - 1, N_rays, N_samples, 4])
    assert raw_rgba.shape == torch.Size([N_obj, N_rays, N_samples, 4])
    assert blending_s.shape == torch.Size([N_rays, N_samples])
    assert blending_d.shape == torch.Size([N_obj - 1, N_rays, N_samples])
    assert blending.shape == torch.Size([N_obj, N_rays, N_samples])

    # Rendering.
    (
        rgb_map_full,
        depth_map_full,
        acc_map_full,
        weights_full,
        rgb_map_obj,
        depth_map_obj,
        acc_map_obj,
        weights_obj,
        dynamicness_map_obj,
        alpha_obj,
    ) = raw2outputs(raw_rgba, blending, z_vals, rays_d, raw_noise_std, hard_blending)

    ret = {
        "rgb_map_full": rgb_map_full,
        "depth_map_full": depth_map_full,
        "acc_map_full": acc_map_full,
        "weights_full": weights_full,
        "rgb_map_obj": rgb_map_obj,
        "depth_map_obj": depth_map_obj,
        "acc_map_obj": acc_map_obj,
        "weights_obj": weights_obj,
        "dynamicness_map_obj": dynamicness_map_obj,
        "alpha_obj": alpha_obj,
        "blending": blending,
        "raw_pts": pts_ref[..., :3],
    }

    # We need the sceneflow from the dynamicNeRF.
    sceneflow_b = raw_d_values[..., 4:7]
    sceneflow_f = raw_d_values[..., 7:10]

    def get_flow_points(flow_preds, add_time):
        pts_flow = torch.cat(
            [
                pts[1:] + flow_preds,
                torch.ones_like(flow_preds[..., 0:1])
                * (torch.Tensor(t[1:])[:, None, None, None] + add_time),
            ],
            -1,
        )
        assert pts_flow.shape == torch.Size([N_obj - 1, N_rays, N_samples, 4])
        for idx, t_value in enumerate(t[1:]):
            assert (
                torch.abs(
                    pts_flow[..., 3:][idx].unique() - torch.Tensor([t_value + add_time])
                ).item()
                < 1e-6
            ), breakpoint()
        return pts_flow

    t_interval = 1.0 / num_img * 2.0
    pts_f = get_flow_points(sceneflow_f, t_interval)
    pts_b = get_flow_points(sceneflow_b, -t_interval)
    ret["sceneflow_b"] = sceneflow_b
    ret["sceneflow_f"] = sceneflow_f
    ret["raw_pts_f"] = pts_f[..., :3]
    ret["raw_pts_b"] = pts_b[..., :3]

    # Third pass: we have the DyanmicNeRF results at time t - 1
    ret = get_rgb_weights_after_flow(
        ret,
        pts_b,
        viewdirs,
        network_fn_d,
        network_query_fn_d,
        z_vals[1:],
        rays_d[1:],
        weights_obj[1:],
        raw_noise_std,
        "_b",
        cam_order,
    )

    # Fourth pass: we have the DyanmicNeRF results at time t + 1
    ret = get_rgb_weights_after_flow(
        ret,
        pts_f,
        viewdirs,
        network_fn_d,
        network_query_fn_d,
        z_vals[1:],
        rays_d[1:],
        weights_obj[1:],
        raw_noise_std,
        "_f",
        cam_order,
    )

    if inference:
        return ret

    # Also consider time t - 2 and t + 2 (Learn from NSFF)
    pts_b_b = get_flow_points(ret["sceneflow_b_b"], -t_interval * 2)
    ret["raw_pts_b_b"] = pts_b_b[..., :3]
    pts_f_f = get_flow_points(ret["sceneflow_f_f"], t_interval * 2)
    ret["raw_pts_f_f"] = pts_f_f[..., :3]

    if chain_5frames:
        # Fifth pass: we have the DyanmicNeRF results at time t - 2
        ret = get_rgb_weights_after_flow(
            ret,
            pts_b_b,
            viewdirs,
            network_fn_d,
            network_query_fn_d,
            z_vals[1:],
            rays_d[1:],
            weights_obj[1:],
            raw_noise_std,
            "_b_b",
            cam_order,
        )

        # Sixth pass: we have the DyanmicNeRF results at time t + 2
        ret = get_rgb_weights_after_flow(
            ret,
            pts_f_f,
            viewdirs,
            network_fn_d,
            network_query_fn_d,
            z_vals[1:],
            rays_d[1:],
            weights_obj[1:],
            raw_noise_std,
            "_f_f",
            cam_order,
        )

    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")
            import ipdb

            ipdb.set_trace()

    return ret
