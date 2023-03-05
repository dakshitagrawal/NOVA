import torch

from run_nerf_helpers import NDC2world, induce_flow


def img2mse(x, y, M=None):
    if M == None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def img2mae(x, y, M=None):
    if M == None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x - y) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L1(x, M=None):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L2(x, M=None):
    if M == None:
        return torch.mean(x**2)
    else:
        return torch.sum((x**2) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-19)) / x.shape[0]


def mse2psnr(x):
    return -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))


def loss_RGB_full(pred_rgb, target_rgb, loss_dict, key, mask=None):
    img_loss = img2mse(pred_rgb, target_rgb, mask)
    psnr = mse2psnr(img_loss)
    loss_dict[f"psnr{key}"] = psnr
    loss_dict[f"img{key}_loss"] = img_loss
    return loss_dict


def loss_RGB(pred_rgb, target_rgb, loss_dict, key, mask=None, start_idx=0):
    loss_dict[f"img{key}_loss"] = []
    for obj_idx in range(len(pred_rgb)):
        img_loss = img2mse(
            pred_rgb[obj_idx], target_rgb[obj_idx], mask[obj_idx][:, None]
        )
        psnr = mse2psnr(img_loss)
        loss_dict[f"psnr{key}/{obj_idx+start_idx:02d}"] = psnr
        loss_dict[f"img{key}_loss/{obj_idx+start_idx:02d}"] = img_loss
        loss_dict[f"img{key}_loss"].append(img_loss)

    loss_dict[f"img{key}_loss"] = torch.sum(torch.stack(loss_dict[f"img{key}_loss"]))
    return loss_dict


def consistency_loss(ret, loss_dict):
    loss_dict["consistency_loss"] = L1(ret["sceneflow_f"] + ret["sceneflow_f_b"]) + L1(
        ret["sceneflow_b"] + ret["sceneflow_b_f"]
    )
    return loss_dict


def mask_loss(loss_dict, key, blending, dynamicness, alpha, mask):
    for obj_idx in range(len(mask)):
        obj_blending = blending[obj_idx]
        obj_mask = mask[obj_idx]
        obj_dynamicness = dynamicness[obj_idx]
        obj_alpha = alpha[obj_idx]

        loss_dict[f"mask{key}_loss/{obj_idx:02d}"] = img2mae(obj_dynamicness, obj_mask)

        if obj_idx != 0:
            # TODO
            # Penalize blending of static if it is at same point as dynamic object point
            # The above will fail when we move the dynamic object moves through time or
            # the dynamic camera moves
            # Sparsity loss with blending loss might help ensure this.

            # Make blending outside dynamic mask be zero
            loss_dict[f"mask{key}_loss/{obj_idx:02d}"] += L1(
                obj_blending[(1 - obj_mask).type(torch.bool)]
            )
            # Make alphas outside dynamic mask be zero
            loss_dict[f"mask{key}_loss/{obj_idx:02d}"] += L1(
                obj_alpha[(1 - obj_mask).type(torch.bool)]
            )

    loss_dict[f"mask{key}_loss"] = torch.sum(
        torch.stack([loss_dict[f"mask{key}_loss/{i:02d}"] for i in range(len(mask))])
    )
    return loss_dict


def blending_loss(loss_dict, ret):
    loss_dict["blending_loss"] = L1(1.0 - torch.sum(ret["blending"], dim=0))
    return loss_dict


def sparsity_loss(ret, loss_dict):
    # Ensures that weights and blending of individual cameras are either 0 or 1
    loss_dict["sparse_loss"] = entropy(ret["weights_obj"]) + entropy(ret["blending"])
    return loss_dict


def slow_scene_flow(ret, loss_dict):
    # Slow scene flow. The forward and backward sceneflow should be small.
    loss_dict[f"slow_loss"] = L1(ret["sceneflow_b"]) + L1(ret["sceneflow_f"])
    return loss_dict


def order_loss(ret, loss_dict, mask):
    # Ensure depth for background is same by dynamic nerf and static nerf
    loss_dict["order_loss"] = img2mae(
        ret["depth_map_obj"][1:], ret["depth_map_obj"][0:1], mask[0]
    )

    # TODO add loss to ensure depth map of obj is before static background for
    # pixels where the mask is positive for dynamic
    # loss_dict["order_loss"] += L2(
    #     torch.maximum(ret["depth_map_obj"][1:] - ret["depth_map_obj"][0:1], 0.0), mask[1:]
    # )
    return loss_dict


def motion_loss(ret, loss_dict, poses, img_i_list, batch_grid, hwf):
    H, W, focal = tuple(hwf)
    num_img = len(poses)

    # Compuate EPE between induced flow and true flow (forward flow).
    # The last frame does not have forward flow.
    for idx, img_i in enumerate(img_i_list[1:]):
        if img_i < num_img - 1:
            obj_grid = batch_grid[idx + 1]
            pts_f = ret["raw_pts_f"][idx]
            weight = ret["weights_obj"][idx + 1]
            pose_f = poses[img_i + 1, :3, :4]
            induced_flow_f = induce_flow(
                H, W, focal, pose_f, weight, pts_f, obj_grid[..., :2]
            )
            flow_f_loss = img2mae(induced_flow_f, obj_grid[:, 2:4], obj_grid[:, 4:5])
            if "flow_f_loss" not in loss_dict:
                loss_dict["flow_f_loss"] = flow_f_loss
            else:
                loss_dict["flow_f_loss"] += flow_f_loss

    # Compuate EPE between induced flow and true flow (backward flow).
    # The first frame does not have backward flow.
    for idx, img_i in enumerate(img_i_list[1:]):
        if img_i > 0:
            obj_grid = batch_grid[idx + 1]
            pts_b = ret["raw_pts_b"][idx]
            weight = ret["weights_obj"][idx + 1]
            pose_b = poses[img_i - 1, :3, :4]
            induced_flow_b = induce_flow(
                H, W, focal, pose_b, weight, pts_b, obj_grid[..., :2]
            )
            flow_b_loss = img2mae(induced_flow_b, obj_grid[:, 5:7], obj_grid[:, 7:8])
            if "flow_b_loss" not in loss_dict:
                loss_dict["flow_b_loss"] = flow_b_loss
            else:
                loss_dict["flow_b_loss"] += flow_b_loss

    return loss_dict


def smooth_scene_flow(ret, loss_dict, hwf, mask):
    # Smooth scene flow. The summation of the forward and backward sceneflow should be small.
    H, W, focal = tuple(hwf)
    num_dobj = len(ret["raw_pts"])
    loss_dict["smooth_loss"] = []
    loss_dict["sf_smooth_loss"] = []
    loss_dict["sp_smooth_loss"] = []
    for i in range(num_dobj):
        obj_mask = mask[i] > 0
        obj_pts = ret["raw_pts"][i, obj_mask]
        obj_pts_f = ret["raw_pts_f"][i, obj_mask]
        obj_pts_b = ret["raw_pts_b"][i, obj_mask]
        obj_pts_b_b = ret["raw_pts_b_b"][i, obj_mask]
        obj_pts_f_f = ret["raw_pts_f_f"][i, obj_mask]

        loss_dict["smooth_loss"].append(
            compute_sf_smooth_loss(obj_pts, obj_pts_f, obj_pts_b, H, W, focal)
        )
        loss_dict["sf_smooth_loss"].append(
            compute_sf_smooth_loss(obj_pts_b, obj_pts, obj_pts_b_b, H, W, focal)
            + compute_sf_smooth_loss(obj_pts_f, obj_pts_f_f, obj_pts, H, W, focal)
        )

        # Spatial smooth scene flow. (loss adapted from NSFF)
        loss_dict["sp_smooth_loss"].append(
            compute_sf_smooth_s_loss(obj_pts, obj_pts_f, H, W, focal)
            + compute_sf_smooth_s_loss(obj_pts, obj_pts_b, H, W, focal)
        )

    loss_dict["smooth_loss"] = torch.sum(torch.stack(loss_dict["smooth_loss"]))
    loss_dict["sf_smooth_loss"] = torch.sum(torch.stack(loss_dict["sf_smooth_loss"]))
    loss_dict["sp_smooth_loss"] = torch.sum(torch.stack(loss_dict["sp_smooth_loss"]))
    return loss_dict


# Spatial smoothness (adapted from NSFF)
def compute_sf_smooth_s_loss(pts1, pts2, H, W, f):

    N_samples = pts1.shape[1]

    # NDC coordinate to world coordinate
    pts1_world = NDC2world(pts1[..., : int(N_samples * 0.95), :], H, W, f)
    pts2_world = NDC2world(pts2[..., : int(N_samples * 0.95), :], H, W, f)

    # scene flow in world coordinate
    scene_flow_world = pts1_world - pts2_world

    return L1(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])


# Temporal smoothness
def compute_sf_smooth_loss(pts, pts_f, pts_b, H, W, f):

    N_samples = pts.shape[1]

    pts_world = NDC2world(pts[..., : int(N_samples * 0.9), :], H, W, f)
    pts_f_world = NDC2world(pts_f[..., : int(N_samples * 0.9), :], H, W, f)
    pts_b_world = NDC2world(pts_b[..., : int(N_samples * 0.9), :], H, W, f)

    # scene flow in world coordinate
    sceneflow_f = pts_f_world - pts_world
    sceneflow_b = pts_b_world - pts_world

    # For a 3D point, its forward and backward sceneflow should be opposite.
    return L2(sceneflow_f + sceneflow_b)


def depth_loss(dyn_depth, gt_depth, mask=None):
    def norm_depth_map(depth_map, M=None):
        d_map = depth_map[M.type(torch.bool)] if M is not None else depth_map
        t_d = torch.median(d_map)
        s_d = torch.mean(torch.abs(d_map))
        return (d_map - t_d) / s_d

    loss = None
    for i in range(len(dyn_depth)):
        mask_i = mask[i] if mask is not None else None
        dyn_depth_norm = norm_depth_map(dyn_depth[i], mask_i)
        gt_depth_norm = norm_depth_map(gt_depth[i], mask_i)

        if loss is None:
            loss = img2mse(dyn_depth_norm, gt_depth_norm)
        else:
            loss += img2mse(dyn_depth_norm, gt_depth_norm)

    return loss
