import torch

from render_utils import render
from run_nerf_helpers import induce_flow, normalize_depth, percentile
from utils.flow_utils import flow_to_image


def write_static_imgs(writer, i, ret, target, mask, key=""):
    writer.add_image(f"{key}rgb_gt", target, global_step=i, dataformats="HWC")
    writer.add_image(f"{key}mask", mask, global_step=i, dataformats="HW")
    writer.add_image(
        f"{key}rgb_s",
        torch.clamp(ret["rgb_map_s"], 0.0, 1.0),
        global_step=i,
        dataformats="HWC",
    )
    writer.add_image(
        f"{key}depth_s",
        normalize_depth(ret["depth_map_s"]),
        global_step=i,
        dataformats="HW",
    )
    writer.add_image(f"{key}acc_s", ret["acc_map_s"], global_step=i, dataformats="HW")


def write_dynamic_imgs(writer, i, ret, grid, invdepth, pose_f, pose_b, hwf):
    H, W, focal = tuple(hwf)
    flow_f_img = flow_to_image(grid[..., 2:4].cpu().numpy())
    flow_b_img = flow_to_image(grid[..., 5:7].cpu().numpy())

    induced_flow_f = induce_flow(
        H,
        W,
        focal,
        pose_f,
        ret["weights_d"],
        ret["raw_pts_f"],
        grid[..., :2],
    )
    induced_flow_f_img = flow_to_image(induced_flow_f.cpu().numpy())

    induced_flow_b = induce_flow(
        H,
        W,
        focal,
        pose_b,
        ret["weights_d"],
        ret["raw_pts_b"],
        grid[..., :2],
    )
    induced_flow_b_img = flow_to_image(induced_flow_b.cpu().numpy())

    writer.add_image(
        "disp",
        torch.clamp(invdepth / percentile(invdepth, 97), 0.0, 1.0),
        global_step=i,
        dataformats="HW",
    )

    writer.add_image(
        "rgb",
        torch.clamp(ret["rgb_map_full"], 0.0, 1.0),
        global_step=i,
        dataformats="HWC",
    )
    writer.add_image(
        "depth",
        normalize_depth(ret["depth_map_full"]),
        global_step=i,
        dataformats="HW",
    )
    writer.add_image("acc", ret["acc_map_full"], global_step=i, dataformats="HW")

    writer.add_image(
        "rgb_d",
        torch.clamp(ret["rgb_map_d"], 0.0, 1.0),
        global_step=i,
        dataformats="HWC",
    )
    writer.add_image(
        "depth_d",
        normalize_depth(ret["depth_map_d"]),
        global_step=i,
        dataformats="HW",
    )
    writer.add_image("acc_d", ret["acc_map_d"], global_step=i, dataformats="HW")

    writer.add_image(
        "induced_flow_f",
        induced_flow_f_img,
        global_step=i,
        dataformats="HWC",
    )
    writer.add_image(
        "induced_flow_b",
        induced_flow_b_img,
        global_step=i,
        dataformats="HWC",
    )
    writer.add_image("flow_f_gt", flow_f_img, global_step=i, dataformats="HWC")
    writer.add_image("flow_b_gt", flow_b_img, global_step=i, dataformats="HWC")

    writer.add_image(
        "dynamicness",
        ret["dynamicness_map"],
        global_step=i,
        dataformats="HW",
    )
