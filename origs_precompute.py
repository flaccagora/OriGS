# Standard library
import logging
import os
import os.path as osp
import random
import sys
import time

# Third-party
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

# Local - lib_prior
from lib_prior.moca_processor import MoCaPrep, mark_dynamic_region
from lib_prior.prior_loading import Saved2D

# Local - lib_camera
from lib_camera.camera import MonocularCameras
from lib_camera.epi_helpers import analyze_track_epi, identify_tracks
from lib_camera.moca import moca_solve
from lib_camera.moca_misc import make_pair_list

# Local - lib_render
from lib_render.render_helper import GS_BACKEND

# Local - visualization
from viz_utils import viz_list_of_colored_points_in_cam_frame


def seed_everything(seed):
    """Set random seeds for reproducibility across all libraries.

    Note: Sets deterministic behavior for torch.backends.cudnn which may impact performance.
    """
    logging.info(f"Setting random seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_moca_processor(pre_cfg):
    """Initialize MoCa preprocessor with configuration.

    Configuration options:
        dep_mode: Depth estimation method ("sensor", "depthcrafter", "metric3d", "uni")
        tap_mode: Tracking method ("bootstapir", "spatracker", "cotracker")
        flow_mode: Optical flow method (default: "raft")
        align_metric_flag: Whether to align metric depth (default: True)
    """
    moca_processor = MoCaPrep(
        dep_mode=getattr(pre_cfg, "dep_mode", "sensor"),
        tap_mode=getattr(pre_cfg, "tap_mode", "bootstapir"),
        flow_mode=getattr(pre_cfg, "flow_mode", "raft"),
        align_metric_flag=getattr(pre_cfg, "align_metric_flag", True),
    )
    return moca_processor


def load_imgs_from_dir(src):
    img_dir = osp.join(src, "images")
    img_fns = sorted(
        [it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")]
    )
    img_list = [imageio.v2.imread(osp.join(img_dir, it))[..., :3] for it in img_fns]
    return img_list, img_fns


def preprocess(
    img_list: list,
    img_fns: list,
    ws: str,
    moca_processor: MoCaPrep,
    pre_cfg: OmegaConf,
    resample_for_dynamic=True,
):
    """Run two-phase preprocessing pipeline for OriGS reconstruction.

    Phase 1: Extract depth maps, tracks, and optical flow from images
    Phase 2: Densify dynamic regions based on epipolar error analysis

    Output files:
        {ws}/{dep_mode}_depth/: Per-frame depth maps
        {ws}/*_uniform_{tap_mode}_tap.npz: Uniform track samples
        {ws}/*_dynamic_{tap_mode}_tap.npz: Dynamic region tracks (if resample_for_dynamic)
        {ws}/epi_resample_mask.gif: Visualization of dynamic regions
    """
    seed_everything(getattr(pre_cfg, "seed", 12345))
    start_t = time.time()
    logging.info("=" * 60)
    logging.info(f"Preprocessing Phase 1: {ws}")
    logging.info("=" * 60)

    # Depth boundary enhancement threshold for sharpening (-1 disables)
    BOUNDARY_EHNAHCE_TH = getattr(pre_cfg, "boundary_enhance_th", -1)
    DEPTH_DIR_POSTFIX = "_depth_sharp" if BOUNDARY_EHNAHCE_TH > 0 else "_depth"

    # Epipolar error threshold in pixels for static/dynamic classification
    EPI_TH = getattr(pre_cfg, "epi_th", 1e-3)
    # Laplacian filter threshold for depth discontinuities (relative to median depth)
    DEPTH_BOUNDARY_TH = getattr(pre_cfg, "depth_boundary_th", 1.0)

    TAP_CHUNK_SIZE = getattr(pre_cfg, "tap_chunk_size", 5000)

    moca_processor.process(
        t_list=None,
        img_list=img_list,
        img_name_list=img_fns,
        save_dir=ws,
        n_track=getattr(pre_cfg, "n_track_uniform", 8192),
        # depth crafter
        depthcrafter_denoising_steps=getattr(
            pre_cfg, "depthcrafter_denoising_steps", 25
        ),
        metric_alignment_frames=getattr(pre_cfg, "metric_alignment_frames", 10),
        metric_alignment_first_quantil=getattr(
            pre_cfg, "metric_alignment_first_quantil", 0.7
        ),
        metric_alignment_bias_flag=getattr(pre_cfg, "metric_alignment_bias_flag", True),
        metric_alignment_kernel=getattr(pre_cfg, "metric_alignment_kernel", "cauchy"),
        metric_alignment_fscale=getattr(pre_cfg, "metric_alignment_fscale", 0.001),
        # TAP
        compute_tap=True,
        tap_chunk_size=TAP_CHUNK_SIZE,
        # Flow
        flow_steps=getattr(pre_cfg, "flow_steps", [1, 3]),
        epi_num_threads=getattr(pre_cfg, "epi_num_threads", 64),
        # Dep enhance for spatracker
        boundary_enhance_th=BOUNDARY_EHNAHCE_TH,  # if > 0 will create a sharp dir
        # boost
        compute_flow=getattr(pre_cfg, "compute_flow", True),
    )

    if not resample_for_dynamic:
        duration = (time.time() - start_t) / 60.0
        logging.info(f"Preprocessing Phase 1 completed in {duration:.2f} min (skipped dynamic resample)")
        logging.info("=" * 60)
        return

    logging.info("=" * 60)
    logging.info(f"Preprocessing Phase 2: Dynamic region densification")
    logging.info("=" * 60)

    # Load preprocessed 2D priors
    s2d = (
        Saved2D(ws)
        .load_epi()  # Optical flow epipolar errors
        .load_dep(f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}", DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=1.0)  # Rescale to median=1 for consistent hyperparams
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(f"*uniform*{moca_processor.tap_mode}", min_valid_cnt=4)
        .load_vos()  # Video object segmentation (optional)
    )

    # Identify dynamic regions using epipolar error
    if hasattr(s2d, "epi"):
        # Use pre-computed 2D optical flow EPI errors
        sample_mask = s2d.epi > EPI_TH
    else:
        # Fallback: Compute EPI from track fundamental matrices
        continuous_pair_list = make_pair_list(s2d.T, interval=[1, 4], dense_flag=True)
        F_list, epierr_list, _ = analyze_track_epi(
            continuous_pair_list, s2d.track, s2d.track_mask, H=s2d.H, W=s2d.W
        )
        track_static_selection, _ = identify_tracks(epierr_list, EPI_TH)
        sample_mask = mark_dynamic_region(
            s2d.track[:, ~track_static_selection],
            s2d.track_mask[:, ~track_static_selection],
            s2d.H,
            s2d.W,
            0.1,
        )
    resampling_mask_dilate_ksize = getattr(pre_cfg, "resampling_mask_dilate_ksize", 7)
    sample_mask = (
        torch.nn.functional.max_pool2d(
            sample_mask[:, None].float(),
            kernel_size=resampling_mask_dilate_ksize,
            stride=1,
            padding=(resampling_mask_dilate_ksize - 1) // 2,
        )[:, 0]
        > 0.5
    )
    imageio.mimsave(
        osp.join(ws, "epi_resample_mask.gif"),
        sample_mask.cpu().numpy().astype(np.uint8) * 255,
    )

    moca_processor.compute_tap(
        ws=ws,
        save_name=f"dynamic_dep={moca_processor.dep_mode}",
        # n_track=8192 * 3,
        n_track=getattr(pre_cfg, "n_track_pdynamic", 8192 * 3),
        img_list=img_list,
        mask_list=sample_mask.detach().cpu().numpy() > 0,
        dep_list=moca_processor.load_dep_list(
            ws, f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}"
        ),
        # K=cams.default_K.detach().cpu().numpy(), # ! maintain the same K as the first infered static one
        max_viz_cnt=getattr(pre_cfg, "max_viz_cnt", 512),
        chunk_size=TAP_CHUNK_SIZE,
    )

    duration = (time.time() - start_t) / 60.0
    logging.info("=" * 60)
    logging.info(f"Preprocessing completed in {duration:.2f} min")
    logging.info("=" * 60)
    return


if __name__ == "__main__":
    import argparse

    # Configure logging to suppress warnings
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        force=True
    )
    # Suppress warnings from all loggers
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser("MoSca-V2 Preprocessing")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument(
        "--skip_dynamic_resample", action="store_true", help="skip dynamic resample"
    )
    args, unknown = parser.parse_known_args()

    img_list, img_fns = load_imgs_from_dir(args.ws)
    prep_cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip('--') for arg in unknown])
    prep_cfg = OmegaConf.merge(prep_cfg, cli_cfg)

    moca_processor = get_moca_processor(prep_cfg)

    preprocess(
        img_list=img_list,
        img_fns=img_fns,
        ws=args.ws,
        moca_processor=moca_processor,
        pre_cfg=prep_cfg,
        resample_for_dynamic=not args.skip_dynamic_resample,
    )
