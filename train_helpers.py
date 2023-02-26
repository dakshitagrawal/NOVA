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
        save_dict["network_fn_d_state_dict"] = render_kwargs_train[
            "network_fn_d"
        ].state_dict()

    torch.save(save_dict, path)
    print(f"Saved weights at {path}")


def select_batch(value, select_coords):
    return value[select_coords[:, 0], select_coords[:, 1]]


def run_nerf_batch(
    img_i,
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

    t = img_i / num_img * 2.0 - 1.0  # time of the current frame
    pose = poses[img_i, :3, :4]
    mask = masks[img_i]  # Static region mask

    rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
    coords_d = torch.stack((torch.where(mask < 0.5)), -1)
    coords_s = torch.stack((torch.where(mask >= 0.5)), -1)

    if static and dynamic:
        # Evenly sample dynamic region and static region
        select_inds_d = np.random.choice(
            coords_d.shape[0], size=[min(len(coords_d), N_rand // 2)], replace=False
        )
        select_inds_s = np.random.choice(
            coords_s.shape[0], size=[N_rand // 2], replace=False
        )
        select_coords = torch.cat([coords_s[select_inds_s], coords_d[select_inds_d]], 0)
    elif static:
        select_inds_s = np.random.choice(
            coords_s.shape[0], size=[N_rand], replace=False
        )
        select_coords = coords_s[select_inds_s]
    elif dynamic:
        select_inds_d = np.random.choice(
            coords_d.shape[0], size=[N_rand], replace=False
        )
        select_coords = coords_d[select_inds_d]
    else:
        raise ValueError("Either static or dynamic must be set to True")

    rays_o = select_batch(rays_o, select_coords)  # (N_rand, 3)
    rays_d = select_batch(rays_d, select_coords)  # (N_rand, 3)
    batch_mask = select_batch(mask[..., None], select_coords)
    batch_rays = torch.stack([rays_o, rays_d], 0)

    #####  Core optimization loop  #####
    ret = render(
        t,
        chain_5frames,
        H,
        W,
        focal,
        chunk=chunk,
        rays=batch_rays,
        **render_kwargs_train,
    )
    return ret, select_coords, batch_mask
