import torch

from render_utils import render
from run_nerf_helpers import induce_flow, normalize_depth, percentile
from utils.flow_utils import flow_to_image


def write_static_imgs(writer, i, ret, target, mask, key=""):
    writer.add_image(f"{key}rgb_gt", target, global_step=i, dataformats="HWC")
    writer.add_image(f"{key}mask", mask, global_step=i, dataformats="HW")
    writer.add_image(
        f"{key}rgb",
        torch.clamp(ret["rgb_map_full"], 0.0, 1.0),
        global_step=i,
        dataformats="HWC",
    )
    writer.add_image(
        f"{key}depth",
        normalize_depth(ret["depth_map_full"]),
        global_step=i,
        dataformats="HW",
    )
    writer.add_image(f"{key}acc", ret["acc_map_full"], global_step=i, dataformats="HW")


def write_dynamic_imgs(writer, i, ret, grid, invdepth, pose_f, pose_b, hwf):
    H, W, focal = tuple(hwf)
    # flow_f_img = flow_to_image(grid[..., 2:4].cpu().numpy())
    # flow_b_img = flow_to_image(grid[..., 5:7].cpu().numpy())

    # induced_flow_f = induce_flow(
    #     H,
    #     W,
    #     focal,
    #     pose_f,
    #     ret["weights_d"],
    #     ret["raw_pts_f"],
    #     grid[..., :2],
    # )
    # induced_flow_f_img = flow_to_image(induced_flow_f.cpu().numpy())

    # induced_flow_b = induce_flow(
    #     H,
    #     W,
    #     focal,
    #     pose_b,
    #     ret["weights_d"],
    #     ret["raw_pts_b"],
    #     grid[..., :2],
    # )
    # induced_flow_b_img = flow_to_image(induced_flow_b.cpu().numpy())

    # writer.add_image(
    #     "induced_flow_f",
    #     induced_flow_f_img,
    #     global_step=i,
    #     dataformats="HWC",
    # )
    # writer.add_image(
    #     "induced_flow_b",
    #     induced_flow_b_img,
    #     global_step=i,
    #     dataformats="HWC",
    # )
    # writer.add_image("flow_f_gt", flow_f_img, global_step=i, dataformats="HWC")
    # writer.add_image("flow_b_gt", flow_b_img, global_step=i, dataformats="HWC")

    for idx, inv in enumerate(invdepth):
        writer.add_image(
            f"disp/{idx}",
            torch.clamp(inv / percentile(inv, 97), 0.0, 1.0),
            global_step=i,
            dataformats="HW",
        )

    for idx, inv in enumerate(ret["rgb_map_obj"]):
        writer.add_image(
            f"rgb_obj/{idx}",
            torch.clamp(ret["rgb_map_obj"][idx], 0.0, 1.0),
            global_step=i,
            dataformats="HWC",
        )
        writer.add_image(
            f"depth_obj/{idx}",
            normalize_depth(ret["depth_map_obj"][idx]),
            global_step=i,
            dataformats="HW",
        )
        writer.add_image(
            f"acc_obj/{idx}", ret["acc_map_obj"][idx], global_step=i, dataformats="HW"
        )
        writer.add_image(
            f"dynamicness/{idx}",
            ret["dynamicness_map_obj"][idx],
            global_step=i,
            dataformats="HW",
        )
