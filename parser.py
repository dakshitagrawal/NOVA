import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/llff/fern", help="input data directory"
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="layers in fine network"
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=300000,
        help="exponential learning rate decay",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 128,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 128,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--random_seed", type=int, default=1, help="fix random seed for repeatability"
    )

    # rendering options
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--use_viewdirsDyn",
        action="store_true",
        help="use full 5D input instead of 3D for D-NeRF",
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type", type=str, default="llff", help="options: llff"
    )

    # llff flags
    parser.add_argument(
        "--factor", type=int, default=8, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=500,
        help="frequency of console printout and metric logging",
    )
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=50000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=50000,
        help="frequency of render_poses video saving",
    )
    parser.add_argument(
        "--N_iters", type=int, default=1000000, help="number of training iterations"
    )
    parser.add_argument(
        "--pretrain_N_iters",
        type=int,
        default=1000000,
        help="number of pretraining iterations",
    )
    parser.add_argument(
        "--view_idx",
        type=int,
        default=0,
        help="which view to keep constant during rendering of novel time",
    )
    parser.add_argument(
        "--time_idx",
        type=int,
        default=0,
        help="which time to keep constant during rendering of novel view",
    )

    # Dynamic NeRF lambdas
    parser.add_argument(
        "--dynamic_loss_lambda", type=float, default=1.0, help="lambda of dynamic loss"
    )
    parser.add_argument(
        "--static_loss_lambda", type=float, default=1.0, help="lambda of static loss"
    )
    parser.add_argument(
        "--full_loss_lambda", type=float, default=3.0, help="lambda of full loss"
    )
    parser.add_argument(
        "--depth_loss_lambda", type=float, default=0.04, help="lambda of depth loss"
    )
    parser.add_argument(
        "--order_loss_lambda", type=float, default=0.1, help="lambda of order loss"
    )
    parser.add_argument(
        "--flow_loss_lambda",
        type=float,
        default=0.02,
        help="lambda of optical flow loss",
    )
    parser.add_argument(
        "--slow_loss_lambda",
        type=float,
        default=0.1,
        help="lambda of sf slow regularization",
    )
    parser.add_argument(
        "--smooth_loss_lambda",
        type=float,
        default=0.1,
        help="lambda of sf smooth regularization",
    )
    parser.add_argument(
        "--consistency_loss_lambda",
        type=float,
        default=0.1,
        help="lambda of sf cycle consistency regularization",
    )
    parser.add_argument(
        "--mask_loss_lambda", type=float, default=0.1, help="lambda of the mask loss"
    )
    parser.add_argument(
        "--sparse_loss_lambda", type=float, default=0.1, help="lambda of sparse loss"
    )
    parser.add_argument(
        "--DyNeRF_blending",
        action="store_true",
        help="use Dynamic NeRF to predict blending weight",
    )
    parser.add_argument(
        "--pretrain", action="store_true", help="Pretrain the StaticneRF"
    )
    parser.add_argument(
        "--ft_path_S",
        type=str,
        default=None,
        help="specific weights npy file to reload for StaticNeRF",
    )

    # For rendering teasers
    parser.add_argument(
        "--frame2dolly", type=int, default=-1, help="choose frame to perform dolly zoom"
    )
    parser.add_argument(
        "--x_trans_multiplier", type=float, default=1.0, help="x_trans_multiplier"
    )
    parser.add_argument(
        "--y_trans_multiplier", type=float, default=0.33, help="y_trans_multiplier"
    )
    parser.add_argument(
        "--z_trans_multiplier", type=float, default=5.0, help="z_trans_multiplier"
    )
    parser.add_argument("--num_novelviews", type=int, default=60, help="num_novelviews")
    parser.add_argument(
        "--focal_decrease", type=float, default=200, help="focal_decrease"
    )
    return parser
