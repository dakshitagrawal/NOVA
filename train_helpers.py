import numpy as np
import torch

from render_utils import get_rays, render


def decay_lr(args, global_step, optimizer):
    decay_rate = 0.1
    decay_steps = args.lrate_decay
    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lrate
    return new_lrate


def save_ckpt(path, global_step, render_kwargs_train, optimizer, save_dynamic=True):
    save_dict = {
        "global_step": global_step,
        "network_fn_s_state_dict": render_kwargs_train["network_fn_s"].state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if save_dynamic:
        save_dict["network_fn_d_state_dict"] = render_kwargs_train["network_fn_d"]

    torch.save(save_dict, path)
    print(f"Saved weights at {path}")


def select_batch(value, select_coords):
    return value[select_coords[:, 0], select_coords[:, 1]]


def select_batch_multiple(value, select_coords):
    return value[:, select_coords[:, 0], select_coords[:, 1]]


def run_nerf_batch(
    ids,
    poses,
    masks,
    hwf,
    N_rand,
    chunk,
    render_kwargs_train,
    chain_5frames,
    static,
    dynamic,
):
    H, W, focal = tuple(hwf)
    num_img = render_kwargs_train["num_img"]

    # First element is of static (primary) camera, the rest are of dynamic objects
    time_ids = []
    cameras_rays_o = []
    cameras_rays_d = []
    cameras_masks = []

    for obj_idx, img_idx in enumerate(ids):
        time_ids.append(img_idx / num_img * 2.0 - 1.0)  # time of the current frame
        pose = poses[int(img_idx), :3, :4]
        rays_o, rays_d = get_rays(
            H, W, focal, torch.Tensor(pose)
        )  # (H, W, 3), (H, W, 3)
        cameras_rays_o.append(rays_o)
        cameras_rays_d.append(rays_d)
        cameras_masks.append(masks[int(img_idx), obj_idx])

    cameras_rays_o = torch.stack(cameras_rays_o, dim=0)
    cameras_rays_d = torch.stack(cameras_rays_d, dim=0)
    cameras_masks = torch.stack(cameras_masks, dim=0)
    assert cameras_rays_o.shape == torch.Size([len(ids), H, W, 3])
    assert cameras_rays_d.shape == torch.Size([len(ids), H, W, 3])
    assert cameras_masks.shape == torch.Size([len(ids), H, W])

    # Select coords based on collective dynamic mask
    collective_mask = cameras_masks[0]
    coords_d = []
    for camera_mask in cameras_masks[1:]:
        coords_d.append(torch.stack((torch.where(camera_mask > 0.5)), -1))
        collective_mask[camera_mask >= 0.5] = 0
    coords_s = torch.stack((torch.where(collective_mask >= 0.5)), -1)

    select_coords = []
    total = 0

    if dynamic:
        total_cameras = len(ids) if static else len(coords_d)
        for coord_d in coords_d:
            select_ind_d = np.random.choice(
                len(coord_d),
                size=[min(len(coord_d), N_rand // total_cameras)],
                replace=False,
            )
            total += len(select_ind_d)
            select_coords.append(coord_d[select_ind_d])

    if static:
        select_inds_s = np.random.choice(
            len(coords_s), size=[max(N_rand // len(ids), N_rand - total)], replace=False
        )
        select_coords.append(coords_s[select_inds_s])

    select_coords = torch.cat(select_coords, 0)
    assert select_coords.shape == torch.Size([N_rand, 2])

    rays_o = select_batch_multiple(
        cameras_rays_o, select_coords
    )  # (N_cameras, N_rand, 3)
    rays_d = select_batch_multiple(
        cameras_rays_d, select_coords
    )  # (N_cameras, N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)
    batch_mask = select_batch_multiple(cameras_masks, select_coords)
    assert rays_o.shape == torch.Size([len(ids), N_rand, 3])
    assert rays_d.shape == torch.Size([len(ids), N_rand, 3])
    assert batch_rays.shape == torch.Size([2, len(ids), N_rand, 3])
    assert batch_mask.shape == torch.Size([len(ids), N_rand])

    #####  Core optimization loop  #####
    ret = render(
        time_ids,
        chain_5frames,
        H,
        W,
        focal,
        chunk=chunk,
        rays=batch_rays,
        **render_kwargs_train,
    )
    return ret, select_coords, batch_mask
