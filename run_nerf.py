import os
import time
from parser import config_parser

import imageio
import numpy as np
import torch
from tensorboardX import SummaryWriter

from load_llff import get_data_variables
from losses import (
    blending_loss,
    consistency_loss,
    depth_loss,
    img2mae,
    img2mse,
    loss_RGB,
    loss_RGB_full,
    mask_loss,
    motion_loss,
    mse2psnr,
    order_loss,
    slow_scene_flow,
    smooth_scene_flow,
    sparsity_loss,
)
from render_samples import render_fix, render_novel_view_and_time
from render_utils import render
from run_nerf_helpers import create_nerf, to8b
from tboard_helpers import write_dynamic_imgs, write_static_imgs
from train_helpers import (
    decay_lr,
    img_rot_homo,
    run_nerf_batch,
    save_ckpt,
    select_batch,
    select_batch_multiple,
)
from utils.flow_utils import flow_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print("Fixing random seed", args.random_seed)
        np.random.seed(args.random_seed)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # Get data variables
    (
        images,
        invdepths,
        masks,
        poses,
        bds_dict,
        render_poses,
        grids,
        hwf,
        num_img,
        N_rand,
    ) = get_data_variables(args)
    H, W, focal = hwf

    # Create nerf model
    num_objects = len(masks[0]) - 1 or 1
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args, num_objects
    )
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, "summaries", expname))

    # Move training data to GPU
    images = torch.Tensor(images)
    invdepths = torch.Tensor(invdepths)
    masks[:, 0] = 1.0 - masks[:, 0]
    masks = torch.Tensor(masks)
    poses = torch.Tensor(poses)
    grids = torch.Tensor(grids)

    # Pre-train StaticNeRF
    if args.pretrain:

        render_kwargs_train.update({"pretrain": True})
        render_kwargs_test.update({"pretrain": False})
        print("BEGIN PRETRAINING")
        i_train = np.arange(int(num_img))
        print("TRAIN views are", i_train)

        if args.ft_path_S is not None and args.ft_path_S != "None":
            # Load Pre-trained StaticNeRF
            ckpt_path = args.ft_path_S
            print("Reloading StaticNeRF from", ckpt_path)
            ckpt = torch.load(ckpt_path)
            render_kwargs_train["network_fn_s"].load_state_dict(
                ckpt["network_fn_s_state_dict"]
            )
        else:
            # Train StaticNeRF from scratch
            for i in range(args.pretrain_N_iters):
                time0 = time.time()

                # No raybatching as we need to take random rays from one image at a time
                img_i = np.ones([1]) * np.random.choice(i_train)
                t = img_i / num_img * 2.0 - 1.0  # time of the current frame
                ret, select_coords, batch_mask = run_nerf_batch(
                    img_i,
                    poses,
                    masks,
                    hwf,
                    N_rand,
                    args.chunk,
                    render_kwargs_train,
                    chain_5frames=False,
                    static=True,
                    dynamic=False,
                )
                img_i = img_i.astype(int)
                target_rgb = select_batch(images[img_i[0]], select_coords)

                optimizer.zero_grad()
                loss_dict = loss_RGB_full(
                    ret["rgb_map_full"], target_rgb, {}, "_pretrain"
                )
                loss = args.static_loss_lambda * loss_dict["img_pretrain_loss"]
                # TODO rerun experiment without this loss to determine if needed
                loss_dict["mask_pretrain_loss"] = img2mae(
                    ret["dynamicness_map_full"], batch_mask
                )
                loss += args.static_loss_lambda * loss_dict["mask_pretrain_loss"]
                loss.backward()
                optimizer.step()
                new_lrate = decay_lr(args, i, optimizer)

                dt = time.time() - time0

                if i % args.i_print == 0:
                    print(
                        f"Pretraining step: {i}, Loss: {loss}, Time: {dt}, expname: {expname}"
                    )
                    writer.add_scalar("pretrain_loss", loss.item(), i)
                    writer.add_scalar(
                        "pretrain_img_loss", loss_dict["img_pretrain_loss"].item(), i
                    )
                    writer.add_scalar(
                        "pretrain_mask_loss", loss_dict["mask_pretrain_loss"].item(), i
                    )
                    writer.add_scalar(
                        "pretrain_psnr_s", loss_dict["psnr_pretrain"].item(), i
                    )
                    writer.add_scalar("pretrain_lr", new_lrate, i)

                if i % args.i_img == 0:
                    with torch.no_grad():
                        pose = poses[img_i, :3, :4]
                        ret = render(
                            t,
                            False,
                            H,
                            W,
                            focal,
                            chunk=1024 * 16,
                            c2w=pose,
                            pretrain=True,
                            **render_kwargs_test,
                        )
                        write_static_imgs(
                            writer,
                            i,
                            ret,
                            images[img_i][0],
                            masks[img_i][0][0],
                            "pretrain_",
                        )

                if i % args.i_testset == 0 and i > 0:
                    fix_values = [
                        (None, None),
                        (args.view_idx, None),
                        (None, args.time_idx),
                    ]
                    for fix_value in fix_values:
                        render_fix(
                            basedir,
                            expname,
                            i,
                            args.chunk,
                            hwf,
                            render_kwargs_test,
                            poses,
                            view_idx=fix_value[0],
                            time_idx=fix_value[1],
                            key="pretrain_",
                        )

                if i % args.i_video == 0 and i > 0:
                    render_novel_view_and_time(
                        basedir,
                        expname,
                        i,
                        args.chunk,
                        hwf,
                        render_kwargs_test,
                        render_poses,
                        key="pretrain_",
                    )

            # Save the pretrained weight
            path = os.path.join(basedir, expname, "Pretrained_S.tar")
            save_ckpt(path, i, render_kwargs_train, optimizer, False)

        # Reset
        render_kwargs_train.update({"pretrain": False})
        render_kwargs_test.update({"pretrain": False})

        # Fix the StaticNeRF and only train the DynamicNeRF
        grad_vars_d = list()
        for network_d in render_kwargs_train["network_fn_d"]:
            grad_vars_d += list(network_d.parameters())

        if not args.fix_static_nerf_after_pretraining:
            grad_vars_d += list(render_kwargs_train["network_fn_s"].parameters())

        optimizer = torch.optim.Adam(
            params=grad_vars_d, lr=args.lrate, betas=(0.9, 0.999)
        )
        start = 0
        print("Resetting step to ", start)

    # Start actual training
    print("BEGIN")
    i_train = np.arange(int(num_img))
    decay_iteration = max(25, num_img)
    print("TRAIN views are", i_train)

    for i in range(start, args.N_iters):
        time0 = time.time()

        # Use frames at t-2, t-1, t, t+1, t+2 (adapted from NSFF)
        # chain_5frames = i >= (decay_iteration * 2000)
        chain_5frames = False

        # Lambda decay.
        Temp = 1.0 / (10 ** (i // (decay_iteration * 1000)))

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        # No raybatching as we need to take random rays from one image at a time
        img_i = np.ones([num_objects + 1]) * np.random.choice(i_train)
        ret, select_coords, batch_mask, rotations, axes = run_nerf_batch(
            img_i,
            poses,
            masks,
            hwf,
            N_rand,
            args.chunk,
            render_kwargs_train,
            chain_5frames=chain_5frames,
            static=True,
            dynamic=True,
            novel_view=(i % 3 == 0),
        )
        img_i = img_i.astype(int)

        # use rotations to warp images, grids, invdepths
        warped_images = []
        warped_grids = []
        warped_invdepths = []
        warped_masks = []

        for idx, img_idx in enumerate(img_i):
            tup = (rotations[idx], H, W, focal, axes[idx])
            warped_images.append(img_rot_homo(images[img_idx], *tup))
            warped_grids.append(img_rot_homo(grids[img_idx], *tup))
            warped_invdepths.append(img_rot_homo(invdepths[img_idx], *tup))
            warped_masks.append(img_rot_homo(masks[img_idx, idx], *tup))

        warped_images = torch.stack(warped_images)
        warped_grids = torch.stack(warped_grids)
        warped_invdepths = torch.stack(warped_invdepths)
        warped_masks = torch.stack(warped_masks)
        warped_full_mask = warped_masks[0].clone()
        for cmask in warped_masks[1:]:
            warped_full_mask[cmask >= 0.5] = 0

        target_rgb = select_batch_multiple(warped_images, select_coords)
        batch_grid = select_batch_multiple(warped_grids, select_coords)  # (N_rand, 8)
        batch_invdepth = select_batch_multiple(warped_invdepths, select_coords)

        # create warped full image using masks with dynamic objects at rotated places
        warped_full_image = (
            torch.sum(warped_images[1:] * warped_masks[1:, ..., None], dim=0)
            + warped_images[0] * warped_full_mask[..., None]
        )
        warped_full_invdepth = (
            torch.sum(warped_invdepths[1:] * warped_masks[1:], dim=0)
            + warped_invdepths[0] * warped_full_mask
        )
        target_rgb_full = select_batch(warped_full_image, select_coords)
        batch_invdepth_full = select_batch(warped_full_invdepth, select_coords)

        optimizer.zero_grad()
        loss = 0
        loss_dict = {}

        # TODO revisit mask loss for points where both static and dynamic are present
        loss_dict = mask_loss(
            loss_dict,
            "",
            ret["blending"],
            ret["dynamicness_map_obj"],
            ret["alpha_obj"],
            batch_mask,
        )
        loss += args.mask_loss_lambda * loss_dict["mask_loss"]

        loss_dict = blending_loss(loss_dict, ret)
        loss += args.mask_loss_lambda * loss_dict["blending_loss"]

        loss_dict = loss_RGB_full(
            ret["rgb_map_full"], target_rgb_full, loss_dict, "_full"
        )
        loss += args.full_loss_lambda * loss_dict["img_full_loss"]

        loss_dict = loss_RGB(
            ret["rgb_map_obj"], target_rgb, loss_dict, "_obj", batch_mask
        )
        loss += args.dynamic_loss_lambda * loss_dict["img_obj_loss"]

        loss_dict = order_loss(ret, loss_dict, batch_mask)
        loss += args.order_loss_lambda * loss_dict["order_loss"]

        # Depth in NDC space equals to negative disparity in Euclidean space.
        loss_dict["depth_loss"] = depth_loss(
            ret["depth_map_obj"], -batch_invdepth, batch_mask
        )
        # loss += args.depth_loss_lambda * loss_dict["depth_loss"]
        loss_dict["depth_loss_full"] = depth_loss(
            [ret["depth_map_full"]], [-batch_invdepth_full]
        )
        # loss += args.depth_loss_lambda * loss_dict["depth_loss_full"]

        # TODO check and fix scene flow losses
        loss_dict = slow_scene_flow(ret, loss_dict)
        # loss += args.slow_loss_lambda * loss_dict["slow_loss"]

        loss_dict = smooth_scene_flow(ret, loss_dict, hwf, batch_mask)
        # loss += args.smooth_loss_lambda * loss_dict["smooth_loss"]
        # loss += args.smooth_loss_lambda * loss_dict["sp_smooth_loss"]
        # loss += args.smooth_loss_lambda * loss_dict["sf_smooth_loss"]

        loss_dict = consistency_loss(ret, loss_dict)
        # loss += args.consistency_loss_lambda * loss_dict["consistency_loss"]

        loss_dict = sparsity_loss(ret, loss_dict)
        # loss += args.sparse_loss_lambda * loss_dict["sparse_loss"]

        loss_dict = motion_loss(ret, loss_dict, poses, img_i, batch_grid, hwf)
        # if "flow_f_loss" in loss_dict:
        #     loss += args.flow_loss_lambda * Temp * loss_dict["flow_f_loss"]
        # if "flow_b_loss" in loss_dict:
        #     loss += args.flow_loss_lambda * Temp * loss_dict["flow_b_loss"]

        loss_dict = loss_RGB(
            ret["rgb_map_d_f"], target_rgb, loss_dict, "_d_f", batch_mask[1:], 1
        )
        # loss += args.dynamic_loss_lambda * loss_dict["img_d_f_loss"]

        loss_dict = loss_RGB(
            ret["rgb_map_d_b"], target_rgb, loss_dict, "_d_b", batch_mask[1:], 1
        )
        # loss += args.dynamic_loss_lambda * loss_dict["img_d_b_loss"]

        if chain_5frames:
            loss_dict = loss_RGB(
                ret["rgb_map_d_b_b"], target_rgb, loss_dict, "_d_b_b", batch_mask[1:]
            )
            # loss += args.dynamic_loss_lambda * loss_dict["img_d_b_b_loss"]

            loss_dict = loss_RGB(
                ret["rgb_map_d_f_f"], target_rgb, loss_dict, "_d_f_f", batch_mask[1:]
            )
            # loss += args.dynamic_loss_lambda * loss_dict["img_d_f_f_loss"]

        loss.backward()
        optimizer.step()
        new_lrate = decay_lr(args, i, optimizer)

        dt = time.time() - time0

        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            save_ckpt(path, i, render_kwargs_train, optimizer, True)

        if i % args.i_testset == 0 and i > 0:
            fix_values = [(None, None), (args.view_idx, None), (None, args.time_idx)]
            for fix_value in fix_values:
                render_fix(
                    basedir,
                    expname,
                    i,
                    args.chunk,
                    hwf,
                    render_kwargs_test,
                    poses,
                    view_idx=fix_value[0],
                    time_idx=fix_value[1],
                )

        if i % args.i_video == 0 and i > 0:
            render_novel_view_and_time(
                basedir,
                expname,
                i,
                args.chunk,
                hwf,
                render_kwargs_test,
                render_poses,
            )

        if i % args.i_print == 0:
            print(
                f"Step: {i}, Loss: {loss}, Time: {dt}, chain_5frames: {chain_5frames}, expname: {expname}"
            )
            writer.add_scalar("loss", loss.item(), i)
            writer.add_scalar("lr", new_lrate, i)
            writer.add_scalar("Temp", Temp, i)
            for loss_key in loss_dict:
                writer.add_scalar(loss_key, loss_dict[loss_key].item(), i)

        if i % args.i_img == 0:
            # change poses, masks using rotations
            warped_images = []
            warped_grids = []
            warped_invdepths = []
            warped_masks = []
            pose = poses[img_i, :3, :4]

            for idx, img_idx in enumerate(img_i):
                pose[idx, :3, :3] = pose[idx, :3, :3] @ rotations[idx].T
                tup = (rotations[idx], H, W, focal, axes[idx])
                warped_images.append(img_rot_homo(images[img_idx], *tup))
                warped_grids.append(img_rot_homo(grids[img_idx], *tup))
                warped_invdepths.append(img_rot_homo(invdepths[img_idx], *tup))
                warped_masks.append(img_rot_homo(masks[img_idx, idx], *tup))

            warped_images = torch.stack(warped_images)
            warped_grids = torch.stack(warped_grids)
            warped_invdepths = torch.stack(warped_invdepths)
            warped_masks = torch.stack(warped_masks)

            # get collective mask instead of this original static background mask
            warped_full_mask = warped_masks[0].clone()
            for cmask in warped_masks[1:]:
                warped_full_mask[cmask >= 0.5] = 0
            warped_full_image = (
                torch.sum(warped_images[1:] * warped_masks[1:, ..., None], dim=0)
                + warped_images[0] * warped_full_mask[..., None]
            )

            with torch.no_grad():
                ret = render(
                    img_i / num_img * 2.0 - 1.0,
                    False,
                    H,
                    W,
                    focal,
                    chunk=1024 * 16,
                    c2w=pose,
                    **render_kwargs_test,
                )

            # Save out the validation image for Tensorboard-free monitoring
            testimgdir = os.path.join(basedir, expname, "tboard_val_imgs")
            os.makedirs(testimgdir, exist_ok=True)
            imageio.imwrite(
                os.path.join(testimgdir, f"{i:06d}.png"),
                to8b(ret["rgb_map_full"].cpu().numpy()),
            )

            pose_f = poses[np.clip(img_i + 1, 0, int(num_img) - 1), :3, :4]
            pose_b = poses[np.clip(img_i - 1, 0, int(num_img) - 1), :3, :4]
            write_static_imgs(writer, i, ret, warped_full_image, warped_full_mask)
            write_dynamic_imgs(
                writer,
                i,
                ret,
                warped_images,
                warped_grids,
                warped_invdepths,
                warped_masks,
                pose_f,
                pose_b,
                hwf,
            )


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    train()
