import torch
import os.path as osp
import logging
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D
from lib_camera.moca import moca_solve
from lib_camera.camera import MonocularCameras

from recon_utils import (
    seed_everything,
    setup_recon_ws,
    auto_get_depth_dir_tap_mode,
    SEED,
)


def static_reconstruct(ws, log_path, fit_cfg):
    # config
    seed_everything(SEED)
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)
    EPI_TH = getattr(fit_cfg, "ba_epi_th", getattr(fit_cfg, "epi_th", 1e-3))
    logging.info(f"Static BA with EPI_TH={EPI_TH}")
    device = torch.device("cuda:0")

    # load 2d data
    s2d: Saved2D = (
        Saved2D(ws)
        .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=DEP_MEDIAN)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            f"*uniform*{TAP_MODE}",
            min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4),
        )
        .load_vos()
    )

    # init camera (GT camera initialization removed for demo-only version)
    cams = None

    # BA solve camera
    logging.info("*" * 20 + "MoCa BA" + "*" * 20)
    cams, s2d, _ = moca_solve(
        ws=log_path,
        s2d=s2d,
        device=device,
        epi_th=EPI_TH,
        ba_total_steps=getattr(fit_cfg, "ba_total_steps", 2000),
        ba_switch_to_ind_step=getattr(fit_cfg, "ba_switch_to_ind_step", 500),
        ba_depth_correction_after_step=getattr(
            fit_cfg, "ba_depth_correction_after_step", 500
        ),
        ba_max_frames_per_step=32,
        static_id_mode="raft" if s2d.has_epi else "track",
        # * robust setting
        robust_depth_decay_th=getattr(fit_cfg, "robust_depth_decay_th", 2.0),
        robust_depth_decay_sigma=getattr(fit_cfg, "robust_depth_decay_sigma", 1.0),
        robust_std_decay_th=getattr(fit_cfg, "robust_std_decay_th", 0.2),
        robust_std_decay_sigma=getattr(fit_cfg, "robust_std_decay_sigma", 0.2),
        #
        iso_focal=getattr(fit_cfg, "iso_focal", False),
        ba_lr_cam_f=getattr(fit_cfg, "ba_lr_cam_f", 0.0003),
        ba_lr_dep_c=getattr(fit_cfg, "ba_lr_dep_c", 0.001),
        ba_lr_dep_s=getattr(fit_cfg, "ba_lr_dep_s", 0.001),
        ba_lr_cam_q=getattr(fit_cfg, "ba_lr_cam_q", 0.0003),
        ba_lr_cam_t=getattr(fit_cfg, "ba_lr_cam_t", 0.0003),
        #
        ba_lambda_flow=getattr(fit_cfg, "ba_lambda_flow", 1.0),
        ba_lambda_depth=getattr(fit_cfg, "ba_lambda_depth", 0.1),
        ba_lambda_small_correction=getattr(fit_cfg, "ba_lambda_small_correction", 0.03),
        ba_lambda_cam_smooth_trans=getattr(fit_cfg, "ba_lambda_cam_smooth_trans", 0.0),
        ba_lambda_cam_smooth_rot=getattr(fit_cfg, "ba_lambda_cam_smooth_rot", 0.0),
        #
        depth_filter_th=getattr(fit_cfg, "ba_depth_remove_th", -1.0),
        init_cam_with_optimal_fov_results=getattr(
            fit_cfg, "init_cam_with_optimal_fov_results", True
        ),
        # fov
        fov_search_fallback=getattr(fit_cfg, "ba_fov_search_fallback", 53.0),
        fov_search_N=getattr(fit_cfg, "ba_fov_search_N", 100),
        fov_search_start=getattr(fit_cfg, "ba_fov_search_start", 30.0),
        fov_search_end=getattr(fit_cfg, "ba_fov_search_end", 90.0),
        viz_valid_ba_points=getattr(fit_cfg, "ba_viz_valid_points", False),
    )  # ! S2D is changed becuase the depth is re-scaled

    return s2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoCa Reconstruction Camera Only")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    logdir = setup_recon_ws(args.ws, fit_cfg=cfg)

    static_reconstruct(args.ws, logdir, cfg)
