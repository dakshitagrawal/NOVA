import os
from parser import config_parser

import numpy as np
import torch

from load_llff import get_data_variables
from render_utils import render_path, save_res
from run_nerf_helpers import create_nerf


def save_render(
    basedir,
    expname,
    result_type,
    idx,
    pose2render,
    time2render,
    hwf,
    chunk,
    render_kwargs_test,
):
    testsavedir = os.path.join(basedir, expname, result_type + f"_{idx:06d}")
    os.makedirs(testsavedir, exist_ok=True)
    with torch.no_grad():
        ret = render_path(
            pose2render,
            time2render,
            hwf,
            chunk,
            render_kwargs_test,
            savedir=testsavedir,
        )
    moviebase = os.path.join(testsavedir, f"{expname}_{result_type}_{idx:06d}_")
    save_res(moviebase, ret)


def render_fix(
    basedir,
    expname,
    idx,
    chunk,
    hwf,
    render_kwargs_test,
    poses,
    view_idx=None,
    time_idx=None,
    key="",
):
    """
    Fix view if view_idx is not None.
    Fix time if time_idx is not None.
    If both view_idx and time_idx are None, render test views.
    """

    num_img = int(render_kwargs_test["num_img"])
    i_train = np.arange(num_img)

    if view_idx is not None:
        result_type = f"{key}testset_view{view_idx:03d}"
        time2render = i_train / float(num_img) * 2.0 - 1.0
        pose2render = torch.Tensor(poses[view_idx : view_idx + 1, ...]).expand(
            [num_img, 3, 4]
        )
    elif time_idx is not None:
        result_type = f"{key}testset_time{time_idx:03d}"
        time2render = np.tile(time_idx, [int(num_img)]) / float(num_img) * 2.0 - 1.0
        pose2render = torch.Tensor(poses)
    else:
        result_type = f"{key}testset"
        time2render = i_train / float(num_img) * 2.0 - 1.0
        pose2render = torch.Tensor(poses)

    time2render = np.stack([time2render] * 2, 1)
    pose2render = torch.stack([pose2render] * 2, 1)

    save_render(
        basedir,
        expname,
        result_type,
        idx,
        pose2render,
        time2render,
        hwf,
        chunk,
        render_kwargs_test,
    )


def render_novel_view_and_time(
    basedir,
    expname,
    idx,
    chunk,
    hwf,
    render_kwargs_test,
    render_poses,
    key="",
):
    """
    Change time and view at the same time.
    """

    result_type = f"{key}novelviewtime"
    num_img = int(render_kwargs_test["num_img"])
    i_train = np.arange(num_img)
    time2render = np.concatenate(
        (
            np.repeat((i_train / float(num_img) * 2.0 - 1.0), 4),
            np.repeat((i_train / float(num_img) * 2.0 - 1.0)[::-1][1:-1], 4),
        )
    )
    if len(time2render) > len(render_poses):
        pose2render = np.tile(
            render_poses, (int(np.ceil(len(time2render) / len(render_poses))), 1, 1)
        )
        pose2render = pose2render[: len(time2render)]
        pose2render = torch.Tensor(pose2render)
    else:
        time2render = np.tile(
            time2render, int(np.ceil(len(render_poses) / len(time2render)))
        )
        time2render = time2render[: len(render_poses)]
        pose2render = torch.Tensor(render_poses)

    time2render = np.stack([time2render] * 2, 1)
    pose2render = torch.stack([pose2render] * 2, 1)

    save_render(
        basedir,
        expname,
        result_type,
        idx,
        pose2render,
        time2render,
        hwf,
        chunk,
        render_kwargs_test,
    )


def main():
    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print("Fixing random seed", args.random_seed)
        np.random.seed(args.random_seed)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

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

    # Create nerf model
    num_objects = len(masks[0]) - 1 or 1
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args, num_objects
    )
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    fix_values = [
        (None, None),
        (args.view_idx, None),
        (None, args.time_idx),
    ]
    for fix_value in fix_values:
        render_fix(
            basedir,
            expname,
            start + 1,
            args.chunk,
            hwf,
            render_kwargs_test,
            poses,
            view_idx=fix_value[0],
            time_idx=fix_value[1],
            key="testing_",
        )
    render_novel_view_and_time(
        basedir,
        expname,
        start + 1,
        args.chunk,
        hwf,
        render_kwargs_test,
        render_poses,
        key="testing_",
    )


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    main()
