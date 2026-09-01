"""Unified evaluation pipeline: volume generation + 2.5D FID under torchrun DDP.

Phases:
  - generate : run VAE/UNet on UKB MRI volumes, save base/recon/gen NIfTI + slices.
  - fid      : extract 2.5D features and compute FID across XY/YZ/ZX planes.

Layout under --exp_dir:
  {exp_dir}/outputs/volumes/                          (base/recon/gen NIfTI)
  {exp_dir}/outputs/slices/                           (XYZ PNG visualizations)
  {exp_dir}/outputs/features/features-{shape}/...     (.pt feature caches)

eval_mode auto-derives the synthetic filelist + features dir:
  real_vs_recon -> filelist_recon_{N}.txt + recon_features
  real_vs_gen   -> filelist_gen_{N}.txt   + gen_features
  real_vs_real  -> generation only (FID skipped)
"""
from __future__ import annotations

import os
import sys
import json
import warnings
import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast

import monai
from monai.config import print_config
from monai.utils import set_determinism
from monai.metrics.fid import FIDMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    EnsureTyped,
    ScaleIntensityRangePercentilesd,
    Resize,
    Resized,
    EnsureChannelFirst,
)
from monai.losses.perceptual import PerceptualLoss
from scripts.config_utils import load_json
from scripts.utils import define_instance, compute_psnr
from datasets import get_adapter
import patches  # noqa: F401  # registers DiffusionModelUNetMaisiV2 / RFlowSchedulerV2 for ConfigParser
from scripts.utils_plot import get_xyz_plot

warnings.filterwarnings("ignore")

logger = logging.getLogger("compute_metric")
if not logger.handlers:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)


# =============================================================================
# Config + path derivation
# =============================================================================
def load_config():
    parser = argparse.ArgumentParser()
    # Experiment plumbing
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--dataset_config_path", type=str, required=True,
                        help="Path to dataset config (e.g. configs/ukb_20252/dataset.json).")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to merged model+inference config (model_fm.json content).")
    parser.add_argument("--pretrained_vae_path", type=str, default=None)
    parser.add_argument("--pretrained_unet_path", type=str, default=None)

    # Mode + workload
    parser.add_argument("--eval_mode", type=str, required=True,
                        choices=["real_vs_real", "real_vs_recon", "real_vs_gen"])
    parser.add_argument("--phase", type=str, default="all",
                        choices=["generate", "fid", "all"])
    parser.add_argument("--num_images", type=int, default=2500)
    parser.add_argument("--deterministic_recon", action="store_true",
                        help="real_vs_recon: decode the posterior MEAN z=z_mu instead of "
                             "the stochastic sample z=z_mu+eps*z_sigma (matches a deterministic "
                             "recon; removes the sampling-noise asymmetry vs VQ models)")
    parser.add_argument("--ssim_foreground", action="store_true",
                        help="real_vs_recon: ALSO report a background-excluded SSIM (ssim_fg) "
                             "alongside the whole-volume SSIM. Whole-volume SSIM is dominated "
                             "by background: where real and recon are both constant 0 the local "
                             "SSIM is ~1 (inflating the mean), but a small non-zero background "
                             "residue in the recon collapses the structure term to ~0 over most "
                             "of the volume (deflating it). Both failure modes are decoupled "
                             "from LPIPS/PSNR, which is why SSIM disagrees with them. ssim_fg "
                             "averages the per-voxel SSIM map over a foreground mask taken from "
                             "the REAL volume only, so the mask is identical across models.")
    parser.add_argument("--ssim_fg_thresh", type=float, default=1e-6,
                        help="Foreground threshold on the clamped REAL volume for "
                             "--ssim_foreground (default 1e-6; inputs are skull-stripped so the "
                             "background is exactly 0).")
    parser.add_argument("--ssim_fg_erode", type=int, default=0,
                        help="Erode the --ssim_foreground mask by this many voxels before "
                             "averaging (default 0). SSIM's 7^3 window straddles the brain "
                             "boundary, so erode>=3 drops windows that still see background.")
    parser.add_argument("--postfix", type=str, default="30step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="real_vs_gen classifier-free guidance scale. 1.0 = no "
                             "guidance (single conditional pass, backwards-compatible). "
                             ">1.0 runs a second 'null' pass (modality-only, mirroring "
                             "the training whole-set drop) and combines: "
                             "out = null + g*(cond - null). Only used when conditioning "
                             "is enabled; ignored for the legacy scalar-meta path.")

    # Generation phase
    parser.add_argument("--base_label_dir", type=str, default=None)
    parser.add_argument("--other_label_dir", type=str, default=None)
    # Per-user paths. Passed by launcher from env.local.sh; if unset, dataset.json values (typically null) win.
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Source NIfTI dir. Overrides dataset.data_dir.")
    parser.add_argument("--feature_extractor_path", type=str, default=None,
                        help="RadImageNet checkpoint .pth. Overrides dataset.feature_extractor_path.")
    parser.add_argument("--save_real", action="store_true")
    parser.add_argument("--save_volume", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--weight_dtype", type=str, default="fp32")

    # FID phase
    parser.add_argument("--fid_model_name", type=str, default="radimagenet_resnet50",
                        choices=["radimagenet_resnet50", "imagenet_inception",
                                 "imagenet_swav", "dinov2", "med3d"],
                        help="FID feature extractor. radimagenet_resnet50 = 2.5D RadImageNet "
                             "(MAISI-faithful, current). imagenet_inception = 2.5D torchvision "
                             "InceptionV3 (ImageNet; MICCAI 2311.13717 reports better human-"
                             "perception alignment than RadImageNet). imagenet_swav = 2.5D "
                             "ImageNet-SwAV ResNet50 backbone (Woodland 2311.13717's best "
                             "human-aligned medical extractor; 2048-d). dinov2 = 2.5D FD-DINOv2 "
                             "ViT (Stein NeurIPS'23 general-domain standard; CLS embedding). "
                             "med3d = TRUE-3D MedicalNet ResNet-10 (what 3D-MedDiffusion uses; "
                             "whole-volume, single FID). med3d bypasses 2.5D slicing (run_fid_3d).")
    parser.add_argument("--med3d_repo", type=str, default=None,
                        help="Path to the vendored MedicalNet package root (the dir that "
                             "contains medicalnet_models/). Required for --fid_model_name med3d.")
    parser.add_argument("--med3d_weight", type=str, default=None,
                        help="Path to resnet_10_23dataset.pth for --fid_model_name med3d.")
    parser.add_argument("--swav_weight", type=str, default=None,
                        help="Path to a SwAV ResNet50 checkpoint (e.g. swav_800ep_pretrain."
                             "pth.tar). Required for --fid_model_name imagenet_swav.")
    parser.add_argument("--dino_repo", type=str, default=None,
                        help="Path to a LOCAL clone of facebookresearch/dinov2 (loaded via "
                             "torch.hub source='local'). Required for --fid_model_name dinov2.")
    parser.add_argument("--dino_weight", type=str, default=None,
                        help="Path to DINOv2 backbone weights (.pth) for --fid_model_name "
                             "dinov2. If omitted the hub arch is built without pretrained weights.")
    parser.add_argument("--dino_arch", type=str, default="dinov2_vitl14",
                        help="DINOv2 hub entrypoint (default dinov2_vitl14 -> 1024-d CLS, the "
                             "FD-DINOv2 default). Others: dinov2_vitb14 (768), dinov2_vits14 (384).")
    parser.add_argument("--fid_woodland", action="store_true",
                        help="Woodland-exact 2D FID protocol: axial-only (single plane) + "
                             "per-slice content-fraction filter (--fid_content_frac) + 256^2 "
                             "center-pad (no resize). Overrides center-slices; XY=YZ=ZX=Avg = "
                             "the single axial value. Intended for imagenet_swav (Woodland's "
                             "best medical extractor); matches fid-med-eval prepare_MSDbt.py.")
    parser.add_argument("--fid_content_frac", type=float, default=0.15,
                        help="Woodland content-fraction threshold: keep an axial slice only if "
                             "its nonzero (foreground) pixel fraction exceeds this (default 0.15).")
    parser.add_argument("--fid_resampling_spacing", type=str, default="1.0x1.0x1.0")
    parser.add_argument("--fid_center_slices_ratio", type=float, default=1.0,
                        help="Central fraction of slices used per axis. DEFAULT 1.0 = all "
                             "slices (matches all historical numbers; the ablation baseline). "
                             "MAISI-official = 0.4 (excludes empty/background slices) — pass "
                             "explicitly as the background-dilution probe / MAISI-matched arm. "
                             "Keep default 1.0 so ratio is a clean controlled variable; decide "
                             "0.4 adoption AFTER the probe, don't bake it into every eval.")
    parser.add_argument("--fid_padding", type=lambda s: str(s).lower() == "true", default=True)
    parser.add_argument("--fid_center_cropping", type=lambda s: str(s).lower() == "true", default=True)
    parser.add_argument("--fid_ignore_existing", type=lambda s: str(s).lower() == "true", default=True)
    parser.add_argument("--fid_bootstrap", type=int, default=0,
                        help="If >0, bootstrap-resample the gathered feature sets this "
                             "many times (with replacement, seeded by --seed) to report "
                             "FID mean ± std + 95%% CI per plane and for the 3-plane avg. "
                             "Quantifies finite-sample FID uncertainty so small model "
                             "gaps (e.g. SID/VAD vs MAISI ~1pt) are statistically "
                             "credible. 0 = point estimate only (current behavior). "
                             "Runs on CPU over already-extracted features (no GPU/regen).")
    parser.add_argument("--fid_decompose", type=lambda s: str(s).lower() == "true", default=True,
                        help="Also log the FID split into its mean-shift term "
                             "||mu_r - mu_g||^2 (systematic/coherent bias, e.g. uniform "
                             "blur) and covariance term tr(S_r + S_g - 2 sqrt(S_r S_g)) "
                             "(distributional spread/diversity mismatch) per plane + avg. "
                             "Additive log lines only; the FID value is unchanged. "
                             "Sum == monai FIDMetric exactly (reuses its _cov/_sqrtm).")

    args = parser.parse_args()

    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}

    dataset_dict = load_json(args.dataset_config_path)
    for k, v in dataset_dict.items():
        setattr(args, k, v)

    config_dict = load_json(args.config_path)
    for k, v in config_dict.items():
        setattr(args, k, v)

    for k, v in cli_overrides.items():
        setattr(args, k, v)
    return args


def derive_paths(args):
    exp = args.exp_dir
    paths = {
        "volumes_dir":       os.path.join(exp, "outputs", "volumes"),
        "slices_dir":        os.path.join(exp, "outputs", "slices"),
        "features_root":     os.path.join(exp, "outputs", "features", f"features-{args.fid_target_shape}"),
        "real_filelist":     os.path.join(exp, f"filelist_real_{args.num_images}.txt"),
        "real_features_dir": "real_features",
    }
    if args.eval_mode == "real_vs_recon":
        paths["synth_filelist"]     = os.path.join(exp, f"filelist_recon_{args.num_images}.txt")
        paths["synth_features_dir"] = "recon_features"
    elif args.eval_mode == "real_vs_gen":
        paths["synth_filelist"]     = os.path.join(exp, f"filelist_gen_{args.num_images}.txt")
        paths["synth_features_dir"] = "gen_features"
    else:
        # real_vs_real: the "synthetic" side is the second real set (other_*), so
        # the FID it reports is the NULL FLOOR -- what a perfect generator would
        # score at this n and this extractor. base/other must be DISJOINT real
        # samples of the same slice (scripts/build_gfid_slices.py writes the
        # _nullA / _nullB pair for exactly this).
        paths["synth_filelist"]     = os.path.join(exp, f"filelist_other_{args.num_images}.txt")
        paths["synth_features_dir"] = "other_features"
    return paths


# =============================================================================
# DDP helpers
# =============================================================================
def init_distributed():
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=timedelta(seconds=7200))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        local_rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return local_rank, world_size, device


# =============================================================================
# Generation phase
# =============================================================================
def save_wandb_style_xyz_plot(tensor_1chw, filename, output_dir, center=None):
    if center is None:
        center = [tensor_1chw.shape[d + 1] // 2 for d in range(3)]
    vis_image = get_xyz_plot(tensor_1chw, center, mask_bool=False)
    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
    vis_image = (vis_image * 255).astype(np.uint8)
    save_path = os.path.join(output_dir, f"{filename}_xyz.png")
    plt.imsave(save_path, vis_image, cmap="gray")


def _percentile_norm(intensity_norm: dict):
    """Construct a ScaleIntensityRangePercentilesd from a norm config dict."""
    return ScaleIntensityRangePercentilesd(
        keys="image",
        lower=intensity_norm["lower"],
        upper=intensity_norm["upper"],
        b_min=intensity_norm["b_min"],
        b_max=intensity_norm["b_max"],
        clip=intensity_norm.get("clip", True),
    )


def build_transforms(weight_dtype, args):
    fid_resolution = tuple(args.fid_resolution)
    base_resolution = tuple(args.resolution)
    intensity_norm = args.intensity_norm_metric
    transform = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes=args.orientation_axcodes),
        _percentile_norm(intensity_norm),
        Resized(keys=["image"], spatial_size=fid_resolution, mode="trilinear"),
        EnsureTyped(keys="image", dtype=weight_dtype),
    ])
    gen_transform = Compose([
        _percentile_norm(intensity_norm),
        EnsureTyped(keys="image", dtype=weight_dtype),
    ])
    slice_transform = Compose([
        EnsureChannelFirst(channel_dim=0),
        Resize(spatial_size=base_resolution, mode="trilinear"),
    ])
    return transform, gen_transform, slice_transform


def load_models(args, device):
    autoencoder, unet, noise_scheduler, loss_perceptual = None, None, None, None
    scale_factor, global_mean = 1.0, 0.0

    if args.eval_mode in ["real_vs_recon", "real_vs_gen"]:
        autoencoder = define_instance(args, "autoencoder_def").to(device)
        ckpt = torch.load(os.path.join(args.pretrained_vae_path, "model.pt"),
                          map_location=device)
        autoencoder.load_state_dict(ckpt["autoencoder"])
        autoencoder.eval()

    if args.eval_mode == "real_vs_gen":
        noise_scheduler = define_instance(args, "noise_scheduler")
        noise_scheduler.set_timesteps(
            num_inference_steps=args.num_inference_steps,
            input_img_size_numel=torch.prod(torch.tensor(args.latent_shape)),
        )
        unet = define_instance(args, "diffusion_unet_def").to(device)
        ckpt = torch.load(os.path.join(args.pretrained_unet_path, "model.pt"),
                          map_location=device, weights_only=False)
        # "unet_state_dict" is a legacy key from older train_UNET checkpoints; new
        # checkpoints use "unet".
        unet.load_state_dict(ckpt.get("unet", ckpt.get("unet_state_dict")), strict=True)
        unet.eval()
        scale_factor = args.scale_factor
        global_mean = args.global_mean

    if args.eval_mode == "real_vs_recon":
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                         is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

    return autoencoder, unet, noise_scheduler, loss_perceptual, scale_factor, global_mean


def load_data_lists(args, adapter):
    base_files, other_files, meta_values = [], [], []

    if args.eval_mode in ["real_vs_real", "real_vs_recon"] or \
       (args.eval_mode == "real_vs_gen" and args.save_real):
        base_files = adapter.load_manifest(args.base_label_dir, args.data_dir,
                                           n=args.num_images)

    if args.eval_mode == "real_vs_real":
        other_files = adapter.load_manifest(args.other_label_dir, args.data_dir,
                                            n=args.num_images)

    if args.eval_mode == "real_vs_gen":
        cond_cfg = getattr(args, "conditioning", None)
        if cond_cfg and cond_cfg.get("enabled", False):
            # Typed token-set: condition generation on real token-sets sampled
            # (with replacement) from base_label_dir. When base CSV is a single
            # cell (cohort×modality[×dx]) the generated set matches that cell's
            # real metadata distribution → per-cell gFID.
            df = adapter.normalize_label_df(pd.read_csv(args.base_label_dir))
            rng = np.random.default_rng(args.seed)
            idx = rng.integers(0, len(df), args.num_images)
            meta_values = [adapter.derive_conditions(df.iloc[int(j)]) for j in idx]
        else:
            meta_values = adapter.meta_value_distribution(args.num_images, args.seed)
            if meta_values is None:
                np.random.seed(args.seed)
                meta_values = np.random.uniform(0.0, 1.0, args.num_images)

    return base_files, other_files, meta_values


def run_generation(args, paths, device, local_rank, world_size):
    adapter = get_adapter(args.dataset_adapter)
    # D2: eval defaults to fp32 (weight_dtype). amp_dtype only matters when --amp
    # is left on; with --no_amp (fp32 eval) autocast is disabled regardless.
    if args.weight_dtype == "fp16":
        weight_dtype, amp_dtype = torch.float16, torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype, amp_dtype = torch.bfloat16, torch.bfloat16
    else:
        weight_dtype, amp_dtype = torch.float32, torch.bfloat16
    cond_cfg = getattr(args, "conditioning", None)
    use_token_set = bool(cond_cfg) and bool(cond_cfg.get("enabled", False))
    guidance_scale = float(getattr(args, "guidance_scale", 1.0))
    # CFG null baseline = modality-only, mirroring the training whole-set drop
    # (cfg_drop_presence keep_idx). keep_idx is configurable but defaults to (0,)
    # = modality (first categorical attribute), per decision A.
    cfg_keep_idx = tuple(cond_cfg.get("cfg_keep_idx", (0,))) if use_token_set else (0,)
    use_cfg = use_token_set and guidance_scale != 1.0
    if local_rank == 0 and use_token_set:
        logger.info(f"[gen] conditioning ON | guidance_scale={guidance_scale} "
                    f"| CFG={'on' if use_cfg else 'off'} | null=modality-only(keep_idx={cfg_keep_idx})")
    transform, gen_transform, slice_transform = build_transforms(weight_dtype, args)
    autoencoder, unet, noise_scheduler, loss_perceptual, scale_factor, global_mean = \
        load_models(args, device)

    base_files, other_files, meta_values = load_data_lists(args, adapter)

    all_idx = list(range(args.num_images))
    my_indices = monai.data.partition_dataset(
        data=all_idx, num_partitions=world_size, even_divisible=False, shuffle=False
    )[local_rank]

    slice_dir = paths["slices_dir"]
    volumes_dir = paths["volumes_dir"]

    local_metrics = {"lpips": 0.0, "psnr": 0.0, "ssim": 0.0, "ssim_fg": 0.0, "count": 0}

    for i in tqdm(my_indices, desc=f"GPU {local_rank} [{args.eval_mode}]",
                  position=local_rank, leave=True):
        torch.manual_seed(args.seed + i)

        # ----- real_vs_real -----
        if args.eval_mode == "real_vs_real":
            if base_files and i < len(base_files):
                slice_path = os.path.join(slice_dir, f"{i:04d}_base_xyz.png")
                vol_path = os.path.join(volumes_dir, f"base_{i:04d}.nii.gz")
                skip = os.path.exists(slice_path) and (not args.save_volume or os.path.exists(vol_path))
                if not skip:
                    base_data = transform(base_files[i])
                    base_images = base_data["image"].unsqueeze(0).to(device, dtype=weight_dtype)
                    save_wandb_style_xyz_plot(slice_transform(base_images.cpu()[0]), f"{i:04d}_base", slice_dir)
                    if args.save_volume:
                        nib.save(nib.Nifti1Image(base_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), vol_path)

            if other_files and i < len(other_files):
                slice_path = os.path.join(slice_dir, f"{i:04d}_other_xyz.png")
                vol_path = os.path.join(volumes_dir, f"other_{i:04d}.nii.gz")
                skip = os.path.exists(slice_path) and (not args.save_volume or os.path.exists(vol_path))
                if not skip:
                    other_data = transform(other_files[i])
                    other_images = other_data["image"].unsqueeze(0).to(device, dtype=weight_dtype)
                    save_wandb_style_xyz_plot(slice_transform(other_images.cpu()[0]), f"{i:04d}_other", slice_dir)
                    if args.save_volume:
                        nib.save(nib.Nifti1Image(other_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), vol_path)

        # ----- real_vs_recon -----
        elif args.eval_mode == "real_vs_recon":
            if base_files and i < len(base_files):
                base_data = transform(base_files[i])
                base_images = base_data["image"].unsqueeze(0).to(device, dtype=weight_dtype)

                slice_path = os.path.join(slice_dir, f"{i:04d}_base_xyz.png")
                vol_path = os.path.join(volumes_dir, f"base_{i:04d}.nii.gz")
                if not os.path.exists(slice_path):
                    save_wandb_style_xyz_plot(slice_transform(base_images.cpu()[0]), f"{i:04d}_base", slice_dir)
                if args.save_volume and not os.path.exists(vol_path):
                    nib.save(nib.Nifti1Image(base_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), vol_path)

                with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                    z_mu, z_sigma = autoencoder.encode(base_images)
                    if args.deterministic_recon:
                        z = z_mu
                    else:
                        z = z_mu + torch.randn_like(z_sigma) * z_sigma
                    reconstruction = autoencoder.decode(z)

                    save_wandb_style_xyz_plot(slice_transform(reconstruction.cpu()[0]), f"{i:04d}_recon", slice_dir)
                    if args.save_volume:
                        recon_path = os.path.join(volumes_dir, f"recon_{i:04d}.nii.gz")
                        nib.save(nib.Nifti1Image(reconstruction.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), recon_path)

                    images_clip = torch.clamp(base_images, 0.0, 1.0)
                    recon_clip = torch.clamp(reconstruction, 0.0, 1.0)

                    local_metrics["lpips"] += loss_perceptual(recon_clip, images_clip).item()
                    local_metrics["psnr"] += compute_psnr(recon_clip.cpu().numpy(), images_clip.cpu().numpy())
                    recon_np = recon_clip.squeeze().cpu().numpy()
                    real_np = images_clip.squeeze().cpu().numpy()
                    if args.ssim_foreground:
                        # full=True also returns the per-voxel SSIM map; the scalar it
                        # returns alongside is exactly the unmasked mean, so this stays
                        # bit-identical to the legacy ssim column.
                        ssim_val, ssim_map = ssim(recon_np, real_np, data_range=1.0, full=True)
                        fg = real_np > args.ssim_fg_thresh
                        if args.ssim_fg_erode > 0:
                            fg = binary_erosion(fg, iterations=args.ssim_fg_erode)
                        # An all-background volume would make the mask empty; fall back to
                        # the unmasked value rather than emitting a NaN into the average.
                        local_metrics["ssim_fg"] += float(ssim_map[fg].mean()) if fg.any() else ssim_val
                    else:
                        ssim_val = ssim(recon_np, real_np, data_range=1.0)
                    local_metrics["ssim"] += ssim_val
                    local_metrics["count"] += 1

        # ----- real_vs_gen -----
        elif args.eval_mode == "real_vs_gen":
            if args.save_real and base_files and i < len(base_files):
                slice_path = os.path.join(slice_dir, f"{i:04d}_base_xyz.png")
                vol_path = os.path.join(volumes_dir, f"base_{i:04d}.nii.gz")
                skip_real = os.path.exists(slice_path) and (not args.save_volume or os.path.exists(vol_path))
                if not skip_real:
                    base_data = transform(base_files[i])
                    base_images = base_data["image"].unsqueeze(0).to(device, dtype=weight_dtype)
                    save_wandb_style_xyz_plot(slice_transform(base_images.cpu()[0]), f"{i:04d}_base", slice_dir)
                    if args.save_volume:
                        nib.save(nib.Nifti1Image(base_images.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), vol_path)

            expected_nii_path = os.path.join(volumes_dir, f"gen_{i:04d}_{args.postfix}.nii.gz")
            expected_png_path = os.path.join(slice_dir, f"{i:04d}_gen_{args.postfix}_xyz.png")
            skip_gen = os.path.exists(expected_nii_path) if args.save_volume else os.path.exists(expected_png_path)

            if not skip_gen:
                noise = torch.randn(
                    (1, args.latent_channels, *args.latent_shape),
                    device=device,
                )
                latent = noise

                spacing_tensor = np.array(args.inference_spacing).astype(float) * 1e2
                spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).to(device, dtype=weight_dtype)

                null_meta_tensor = None
                if use_token_set:
                    from patches.token_set_encoder import encode_token_set
                    ci, cv, pr = encode_token_set(meta_values[i], cond_cfg["attributes"])
                    cond_cat = torch.tensor([ci], dtype=torch.long, device=device)
                    cond_cont = torch.tensor([cv], dtype=torch.float32, device=device)
                    cond_presence = torch.tensor([pr], dtype=torch.bool, device=device)
                    meta_tensor = {
                        "cond_cat": cond_cat,
                        "cond_cont": cond_cont,
                        "cond_presence": cond_presence,
                    }
                    if use_cfg:
                        # null = modality-only: drop every attribute except keep_idx,
                        # preserving the original (present) value at the kept slots.
                        # Mirrors cfg_drop_presence's whole-set drop in train_UNET.
                        null_presence = torch.zeros_like(cond_presence)
                        for k in cfg_keep_idx:
                            null_presence[:, k] = cond_presence[:, k]
                        null_meta_tensor = {
                            "cond_cat": cond_cat,
                            "cond_cont": cond_cont,
                            "cond_presence": null_presence,
                        }
                else:
                    meta_tensor = torch.tensor([[meta_values[i]]], device=device, dtype=weight_dtype)

                all_timesteps = noise_scheduler.timesteps
                all_next_timesteps = torch.cat((all_timesteps[1:],
                                                torch.tensor([0], dtype=all_timesteps.dtype)))

                with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                    for t, next_t in zip(all_timesteps, all_next_timesteps):
                        unet_inputs = {
                            "x": latent,
                            "timesteps": torch.Tensor((t,)).to(device),
                            "spacing_tensor": spacing_tensor,
                            "meta_tensor": meta_tensor,
                        }
                        model_output = unet(**unet_inputs)
                        if use_cfg:
                            # second pass with the modality-only null conditioning,
                            # then classifier-free guidance on the model output
                            # (velocity for RFlow): out = null + g*(cond - null).
                            null_output = unet(**{**unet_inputs, "meta_tensor": null_meta_tensor})
                            model_output = null_output + guidance_scale * (model_output - null_output)
                        latent, _ = noise_scheduler.step(
                            model_output, t, latent, next_t,
                            args.stochastic_scale,
                        )

                    synthetic_images = autoencoder.decode_stage_2_outputs((latent / scale_factor) + global_mean)
                    transformed_synthetic = gen_transform({"image": synthetic_images.cpu()})["image"]

                    save_wandb_style_xyz_plot(slice_transform(transformed_synthetic[0]),
                                              f"{i:04d}_gen_{args.postfix}", slice_dir)
                    if args.save_volume:
                        nib.save(nib.Nifti1Image(transformed_synthetic.cpu().numpy().squeeze().astype(np.float32),
                                                 np.eye(4)), expected_nii_path)
                        # Adherence sidecar: record the INTENDED condition this
                        # volume was generated from, so scripts/adherence_eval.py
                        # can compare predictor(gen) vs the condition. Only the
                        # token-set path carries a structured metadata dict.
                        if use_token_set:
                            cond_raw = meta_values[i]
                            cond_json = {k: (None if v is None else
                                             (float(v) if isinstance(v, (int, float, np.floating))
                                              else str(v)))
                                         for k, v in dict(cond_raw).items()}
                            cond_json["guidance_scale"] = guidance_scale
                            with open(expected_nii_path.replace(".nii.gz", ".cond.json"), "w") as cf:
                                json.dump(cond_json, cf)

    # Aggregate recon metrics across ranks
    if args.eval_mode == "real_vs_recon":
        t = torch.tensor([local_metrics["lpips"], local_metrics["psnr"],
                          local_metrics["ssim"], local_metrics["ssim_fg"],
                          float(local_metrics["count"])],
                         device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            count = t[4].item()
            if count > 0:
                lpips_avg = t[0].item() / count
                psnr_avg = t[1].item() / count
                ssim_avg = t[2].item() / count
                print("\n" + "=" * 50)
                print(f"Final Average Reconstruction Metrics (over {int(count)} imgs):")
                print(f"LPIPS : {lpips_avg:.4f}")
                print(f"PSNR  : {psnr_avg:.4f}")
                print(f"SSIM  : {ssim_avg:.4f}")
                if args.ssim_foreground:
                    # NOT comparable to the SSIM above or to any historical SSIM number:
                    # removing the background drops the inflation term for every model.
                    print(f"SSIM_FG: {t[3].item() / count:.4f} "
                          f"(thresh={args.ssim_fg_thresh:g}, erode={args.ssim_fg_erode})")
                print("=" * 50)
            else:
                print("No metrics computed (count = 0).")


# =============================================================================
# FID phase  (ported from compute_fid.py)
# =============================================================================
def drop_empty_slice(slices, empty_threshold):
    outputs, n_drop = [], 0
    for s in slices:
        if s.max() < empty_threshold:
            outputs.append(False); n_drop += 1
        else:
            outputs.append(True)
    logger.info(f"Empty slice drop rate {round((n_drop/len(slices))*100,1)}%")
    return outputs


def content_frac_mask(slices, content_frac):
    # Woodland fid-med-eval slice filter (prepare_MSDbt.py): keep a slice only if the
    # fraction of foreground (nonzero) pixels EXCEEDS content_frac (default 0.15).
    # Slices are grayscale-replicated 3ch (post skull-strip bg is exactly 0), so
    # channel 0 gives the content fraction. B=1 here (matches drop_empty_slice: one
    # bool per slice position, applied after torch.cat(dim=0)).
    outputs, n_drop = [], 0
    for s in slices:
        frac = (s[:, 0] > 0).float().mean().item()
        if frac > content_frac:
            outputs.append(True)
        else:
            outputs.append(False); n_drop += 1
    logger.info(f"Woodland content<={content_frac} drop rate "
                f"{round((n_drop/len(slices))*100,1)}% (kept {len(slices)-n_drop}/{len(slices)})")
    return outputs


def subtract_mean(x):
    mean = [0.406, 0.456, 0.485]
    x[:, 0, ...] -= mean[0]
    x[:, 1, ...] -= mean[1]
    x[:, 2, ...] -= mean[2]
    return x


def spatial_average(x, keepdim=True):
    dim = len(x.shape)
    if dim == 2: return x
    if dim == 3: return x.mean([2], keepdim=keepdim)
    if dim == 4: return x.mean([2, 3], keepdim=keepdim)
    if dim == 5: return x.mean([2, 3, 4], keepdim=keepdim)
    return x


def radimagenet_intensity_normalisation(volume, norm2d=False):
    # Verbatim port of NV-Generate-CTMR/scripts/compute_fid_2-5d_ct.py: each
    # volume (or 2D slice, if norm2d) is min-max normalised to [0,1] BEFORE the
    # ImageNet-mean subtraction. On our inputs (percentile clip=True already maps
    # each volume to [0,1]) the min-max is a numeric no-op, but we keep it verbatim
    # so this function is byte-identical to the reference extractor normalisation.
    dim = len(volume.shape)
    if dim == 4 and norm2d:
        max2d, _ = torch.max(volume, dim=2, keepdim=True)
        max2d, _ = torch.max(max2d, dim=3, keepdim=True)
        min2d, _ = torch.min(volume, dim=2, keepdim=True)
        min2d, _ = torch.min(min2d, dim=3, keepdim=True)
        volume = (volume - min2d) / (max2d - min2d + 1e-10)
        return subtract_mean(volume)
    elif dim == 4:
        max3d = torch.max(volume)
        min3d = torch.min(volume)
        volume = (volume - min3d) / (max3d - min3d + 1e-10)
        return subtract_mean(volume)
    if dim == 5:
        maxval = torch.max(volume)
        minval = torch.min(volume)
        volume = (volume - minval) / (maxval - minval + 1e-10)
        return subtract_mean(volume)
    return volume


def inception_intensity_normalisation(images_2d):
    # 2.5D ImageNet-InceptionV3 input prep: per-slice min-max -> [0,1], resize to
    # 299x299 (inception's native), then map to [-1,1] (standard FID inception input).
    # Grayscale-replicated 3ch, so no BGR/mean handling. Returns ready-to-forward tensor.
    x = images_2d
    mn = torch.amin(x, dim=(2, 3), keepdim=True)
    mx = torch.amax(x, dim=(2, 3), keepdim=True)
    x = (x - mn) / (mx - mn + 1e-10)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_rgb_normalisation(images_2d, size=224):
    # 2.5D ImageNet-RGB input prep shared by SwAV (ResNet50) and DINOv2 (ViT/14):
    # per-slice min-max -> [0,1], resize to size x size (224 = 16*14, valid for the
    # /14 patch grid), then standard ImageNet mean/std standardisation. Slices arrive
    # grayscale-replicated to 3 identical channels, so the per-channel mean/std is the
    # usual grayscale->RGB FID handling.
    x = images_2d
    mn = torch.amin(x, dim=(2, 3), keepdim=True)
    mx = torch.amax(x, dim=(2, 3), keepdim=True)
    x = (x - mn) / (mx - mn + 1e-10)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def woodland_rgb_normalisation(images_2d, size=256):
    # Woodland fid-med-eval input prep: per-slice min-max -> [0,1], center pad (or
    # crop) to size x size (256, NO resize -> preserves native scale), then ImageNet
    # mean/std. Feeds a ResNet backbone (adaptive-pooled, any HxW ok). Pairs with the
    # content_frac_mask axial-only path for a Woodland-exact SwAV FID.
    x = images_2d
    mn = torch.amin(x, dim=(2, 3), keepdim=True)
    mx = torch.amax(x, dim=(2, 3), keepdim=True)
    x = (x - mn) / (mx - mn + 1e-10)
    H, W = x.shape[-2], x.shape[-1]

    def _fit(v, target):
        if v == target:
            return 0, 0, 0, 0          # pad_lo, pad_hi, crop_lo, crop_hi
        if v < target:
            a = (target - v) // 2
            return a, target - v - a, 0, 0
        a = (v - target) // 2
        return 0, 0, a, v - target - a

    ph_a, ph_b, ch_a, ch_b = _fit(H, size)
    pw_a, pw_b, cw_a, cw_b = _fit(W, size)
    if ch_a or ch_b:
        x = x[..., ch_a:H - ch_b, :]
    if cw_a or cw_b:
        x = x[..., cw_a:W - cw_b]
    x = F.pad(x, (pw_a, pw_b, ph_a, ph_b), mode="constant", value=0.0)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def get_features_2p5d(image, feature_network, center_slices=False,
                     center_slices_ratio=1.0, sample_every_k=1,
                     xy_only=True, drop_empty=False, empty_threshold=-700,
                     feat_norm=None, content_filter=False, content_frac=0.15):
    # feat_norm: per-slice normalisation applied just before feature_network.forward.
    # Defaults to the RadImageNet normalisation (byte-identical to the validated path).
    # content_filter: Woodland content-fraction slice keep (> content_frac nonzero);
    # takes precedence over drop_empty. Typically paired with xy_only=True (axial-only).
    def _mask(slices):
        if content_filter:
            return content_frac_mask(slices, content_frac)
        if drop_empty:
            return drop_empty_slice(slices, empty_threshold)
        return [True] * len(slices)
    if feat_norm is None:
        feat_norm = radimagenet_intensity_normalisation
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1, 1)
    image = image[:, [2, 1, 0], ...]

    B, C, H, W, D = image.size()
    with torch.no_grad():
        # XY (along D)
        if center_slices:
            start_d = int((1.0 - center_slices_ratio) / 2.0 * D)
            end_d = int((1.0 + center_slices_ratio) / 2.0 * D)
            slices = torch.unbind(image[:, :, :, :, start_d:end_d:sample_every_k], dim=-1)
        else:
            slices = torch.unbind(image, dim=-1)
        mapping_index = _mask(slices)
        images_2d = torch.cat(slices, dim=0)
        images_2d = feat_norm(images_2d)
        images_2d = images_2d[mapping_index]
        feat_xy = spatial_average(feature_network.forward(images_2d), keepdim=False)
        if xy_only:
            return feat_xy, None, None

        # YZ (along H)
        if center_slices:
            start_h = int((1.0 - center_slices_ratio) / 2.0 * H)
            end_h = int((1.0 + center_slices_ratio) / 2.0 * H)
            slices = torch.unbind(image[:, :, start_h:end_h:sample_every_k, :, :], dim=2)
        else:
            slices = torch.unbind(image, dim=2)
        mapping_index = _mask(slices)
        images_2d = torch.cat(slices, dim=0)
        images_2d = feat_norm(images_2d)
        images_2d = images_2d[mapping_index]
        feat_yz = spatial_average(feature_network.forward(images_2d), keepdim=False)

        # ZX (along W)
        if center_slices:
            start_w = int((1.0 - center_slices_ratio) / 2.0 * W)
            end_w = int((1.0 + center_slices_ratio) / 2.0 * W)
            slices = torch.unbind(image[:, :, :, start_w:end_w:sample_every_k, :], dim=3)
        else:
            slices = torch.unbind(image, dim=3)
        mapping_index = _mask(slices)
        images_2d = torch.cat(slices, dim=0)
        images_2d = feat_norm(images_2d)
        images_2d = images_2d[mapping_index]
        feat_zx = spatial_average(feature_network.forward(images_2d), keepdim=False)

    return feat_xy, feat_yz, feat_zx


def pad_to_max_size(tensor, max_size, padding_value=0.0):
    pad_size = [0, 0] * (len(tensor.shape) - 1) + [0, max_size - tensor.shape[0]]
    return F.pad(tensor, pad_size, "constant", padding_value)


def load_feature_network(model_name, device, local_rank, weight_path=None, med3d_repo=None,
                         dino_repo=None, dino_arch="dinov2_vitl14"):
    if model_name == "radimagenet_resnet50":
        import torchvision
        feature_network = torchvision.models.resnet50(weights=None)
        feature_network.fc = nn.Identity()
        if weight_path is None:
            raise ValueError("radimagenet_resnet50 requires a weight_path "
                             "(set feature_extractor_path in dataset config).")
        local_weight_path = weight_path
        if local_rank == 0:
            logger.info(f"Loading local weights from: {local_weight_path}")
        if not os.path.exists(local_weight_path):
            raise FileNotFoundError(f"Weight file not found at: {local_weight_path}")
        state_dict = torch.load(local_weight_path, map_location="cpu")
        new_state_dict = {(k[7:] if k.startswith("module.") else k): v
                          for k, v in state_dict.items()}
        msg = feature_network.load_state_dict(new_state_dict, strict=False)
        if local_rank == 0:
            logger.info(f"Weights loaded. (Missing keys expected due to notop): {msg}")
    elif model_name == "imagenet_inception":
        # 2.5D ImageNet InceptionV3 (fc->Identity => 2048-d pool3 features). Standard
        # ImageNet-FID backbone; MICCAI 2311.13717 finds ImageNet extractors align
        # with human perception better than RadImageNet. transform_input=False since
        # inception_intensity_normalisation already maps slices to [-1,1] @ 299x299.
        import torchvision
        try:
            weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            feature_network = torchvision.models.inception_v3(weights=weights, aux_logits=True)
        except Exception:
            feature_network = torchvision.models.inception_v3(pretrained=True, aux_logits=True)
        feature_network.transform_input = False
        feature_network.fc = nn.Identity()
        feature_network.AuxLogits = None       # unused in eval; drop to avoid aux path
        if local_rank == 0:
            logger.info("Loaded torchvision InceptionV3 (ImageNet, fc=Identity, 2048-d).")
    elif model_name == "imagenet_swav":
        # ImageNet self-supervised SwAV, ResNet50 backbone -> 2048-d (fc=Identity).
        # Woodland arXiv:2311.13717 finds ImageNet-SSL extractors (SwAV best) align with
        # human perception for medical FID. Weights = facebookresearch/swav resnet50
        # checkpoint; projection-head / prototype keys are ignored (strict=False loads
        # only the conv backbone, which shares torchvision resnet50 key names).
        import torchvision
        feature_network = torchvision.models.resnet50(weights=None)
        feature_network.fc = nn.Identity()
        if not weight_path:
            raise ValueError("imagenet_swav requires --swav_weight (SwAV resnet50 .pth.tar).")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"SwAV weight not found at: {weight_path}")
        sd = torch.load(weight_path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items()
              if not (k.startswith("projection_head") or k.startswith("prototypes"))}
        msg = feature_network.load_state_dict(sd, strict=False)
        if local_rank == 0:
            logger.info(f"Loaded SwAV ResNet50 backbone (imagenet_swav, fc=Identity, "
                        f"2048-d). load msg: {msg}")
    elif model_name == "dinov2":
        # FD-DINOv2 (Stein et al. NeurIPS'23): DINOv2 features give the most human-
        # aligned FID and are now the general-domain standard. ViT-L/14 -> 1024-d CLS
        # (model(x) returns x_norm_clstoken). Loaded from a LOCAL clone of
        # facebookresearch/dinov2 (torch.hub source='local') so GSDS needs no internet;
        # weights via --dino_weight.
        if not dino_repo:
            raise ValueError("dinov2 requires --dino_repo (local facebookresearch/dinov2 clone).")
        feature_network = torch.hub.load(dino_repo, dino_arch, source="local", pretrained=False)
        if weight_path:
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"DINOv2 weight not found at: {weight_path}")
            sd = torch.load(weight_path, map_location="cpu")
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            msg = feature_network.load_state_dict(sd, strict=False)
            if local_rank == 0:
                logger.info(f"Loaded DINOv2 {dino_arch} weights. load msg: {msg}")
        elif local_rank == 0:
            logger.info(f"DINOv2 {dino_arch} built WITHOUT pretrained weights "
                        f"(--dino_weight not given).")
    elif model_name == "med3d":
        # TRUE-3D MedicalNet ResNet-10 (23-dataset pretrain) = the extractor 3D-Med-
        # Diffusion evaluates with. Whole-volume 1-channel input; forward returns the
        # layer4 feature map (512ch), GAP'd to 512-d in get_features_3d. Loaded from the
        # vendored warvito/MedicalNet-models package (see external/3d_meddiff).
        if not weight_path:
            raise ValueError("med3d requires --med3d_weight (resnet_10_23dataset.pth).")
        if med3d_repo and med3d_repo not in sys.path:
            sys.path.insert(0, med3d_repo)
        # The vendored medicalnet resnet.py does `import gdown` at module load (only for
        # its download helper, which we bypass via local weights). Stub it if absent.
        if "gdown" not in sys.modules:
            try:
                import gdown  # noqa: F401
            except Exception:
                import types
                sys.modules["gdown"] = types.ModuleType("gdown")
        from medicalnet_models.models.resnet import ResNet, BasicBlock
        feature_network = ResNet(BasicBlock, [1, 1, 1, 1])   # resnet10
        sd = torch.load(weight_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        msg = feature_network.load_state_dict(sd, strict=False)
        if local_rank == 0:
            logger.info(f"Loaded MedicalNet ResNet-10 (med3d). load msg: {msg}")
    else:
        import torchvision
        feature_network = torchvision.models.squeezenet1_1(pretrained=True)
    feature_network.to(device)
    feature_network.eval()
    return feature_network


def build_fid_transforms(target_shape_tuple, rs_spacing_tuple,
                         enable_resampling, enable_padding, enable_center_cropping,
                         orientation_axcodes, intensity_norm):
    transform_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes=orientation_axcodes),
    ]
    if enable_resampling:
        transform_list.append(monai.transforms.Spacingd(
            keys=["image"], pixdim=rs_spacing_tuple, mode=["bilinear"]))
    if enable_padding:
        transform_list.append(monai.transforms.SpatialPadd(
            keys=["image"], spatial_size=target_shape_tuple, mode="constant", value=0))
    if enable_center_cropping:
        transform_list.append(monai.transforms.CenterSpatialCropd(
            keys=["image"], roi_size=target_shape_tuple))
    transform_list.append(monai.transforms.ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=intensity_norm["lower"], upper=intensity_norm["upper"],
        b_min=intensity_norm["b_min"], b_max=intensity_norm["b_max"],
        clip=intensity_norm.get("clip", True),
    ))
    return Compose(transform_list)


def _extract_features(loader, dataset_root, output_root, feature_network, device,
                      enable_center_slices, center_slices_ratio_final,
                      ignore_existing, label, local_rank, num_files, feat_norm=None,
                      xy_only=False, content_filter=False, content_frac=0.15):
    feats_xy, feats_yz, feats_zx = [], [], []
    for idx, batch_data in enumerate(loader, start=1):
        img = batch_data["image"].to(device)
        fn = img.meta["filename_or_obj"][0]
        logger.info(f"[Rank {local_rank}] {label} {idx}/{num_files}: {fn}")

        out_fp = Path(fn.replace(dataset_root, output_root).replace(".nii.gz", ".pt"))
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if (not ignore_existing) and os.path.isfile(out_fp):
            feats = torch.load(out_fp, weights_only=True, map_location="cpu")
        else:
            img_t = img.as_tensor()
            logger.info(f"image shape: {tuple(img_t.shape)}")
            feats = get_features_2p5d(
                img_t, feature_network,
                center_slices=enable_center_slices,
                center_slices_ratio=center_slices_ratio_final,
                xy_only=xy_only,
                feat_norm=feat_norm,
                content_filter=content_filter, content_frac=content_frac,
            )
            # Single-plane (Woodland axial-only) returns (xy, None, None); mirror xy
            # into yz/zx so the downstream vstack / per-plane FID all read xy -> Avg=xy.
            if feats[1] is None:
                feats = (feats[0], feats[0], feats[0])
            logger.info(f"feats shapes: {feats[0].shape}, {feats[1].shape}, {feats[2].shape}")
            feats = tuple(f.cpu() for f in feats)
            torch.save(feats, out_fp)

        feats_xy.append(feats[0])
        feats_yz.append(feats[1])
        feats_zx.append(feats[2])

    return torch.vstack(feats_xy), torch.vstack(feats_yz), torch.vstack(feats_zx)


def frechet_terms(y_pred, y):
    """FID split into its two additive components, byte-matching monai FIDMetric.

    FID = ||mu_pred - mu_real||^2  +  tr(S_pred + S_real - 2 sqrt(S_pred S_real))
          \_____ mean-shift term ____/  \_________ covariance term _____________/

    mean-term: coherent/systematic bias — every sample shifted the same way in
      feature space (e.g. uniform blur attenuating high-freq texture). Survives
      averaging over the set; this is what per-sample PSNR/SSIM/LPIPS under-weight.
    cov-term : distributional spread / diversity mismatch (mode coverage).

    Reuses monai's own _cov/_sqrtm and singular/complex handling so the returned
    total is identical to FIDMetric()(y_pred, y). Returns (total, mean_term, cov_term).
    """
    from monai.metrics.fid import _cov, _sqrtm
    y = y.double(); y_pred = y_pred.double()
    mu_x = torch.mean(y_pred, dim=0); sigma_x = _cov(y_pred, rowvar=False)
    mu_y = torch.mean(y, dim=0);      sigma_y = _cov(y, rowvar=False)
    diff = mu_x - mu_y
    covmean = _sqrtm(sigma_x.mm(sigma_y))
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma_x.size(0), device=mu_x.device, dtype=mu_x.dtype) * 1e-6
        covmean = _sqrtm((sigma_x + offset).mm(sigma_y + offset))
    if torch.is_complex(covmean):
        covmean = covmean.real
    mean_term = float(diff.dot(diff))
    cov_term = float(torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * torch.trace(covmean))
    return mean_term + cov_term, mean_term, cov_term


def get_features_3d(volume, feature_network):
    # med3d whole-volume features. volume: (B,1,H,W,D). Per-volume z-score (matches
    # 3D-MedDiffusion's mednet_norm), forward -> (B,512,h,w,d), GAP -> (B,512).
    x = volume
    if x.shape[1] != 1:
        x = x[:, :1, ...]
    x = (x - x.mean()) / (x.std() + 1e-8)
    with torch.no_grad():
        feat = feature_network(x)               # (B, 512, h, w, d)
    return feat.mean(dim=[2, 3, 4])             # GAP -> (B, 512)


def run_fid_3d(args, paths, device, local_rank, world_size):
    # TRUE-3D FID with MedicalNet ResNet-10 (the 3D-MedDiffusion extractor). One feature
    # vector per whole volume -> a single FID (no XY/YZ/ZX planes). Reuses the same
    # filelists / transforms / DDP gather as run_fid.
    if local_rank == 0:
        logger.info("FID model              : med3d (MedicalNet ResNet-10, TRUE-3D)")
        logger.info(f"FID target shape       : {args.fid_target_shape}")
        logger.info(f"med3d weight           : {args.med3d_weight}")
        logger.info(f"med3d repo             : {args.med3d_repo}")
    feature_network = load_feature_network("med3d", device, local_rank,
                                           weight_path=args.med3d_weight,
                                           med3d_repo=args.med3d_repo)
    target_shape_tuple = tuple(int(x) for x in args.fid_target_shape.split("x"))
    enable_resampling = args.fid_resampling_spacing is not None
    rs_spacing_tuple = (tuple(float(x) for x in args.fid_resampling_spacing.split("x"))
                        if enable_resampling else (1.0, 1.0, 1.0))
    dataset_root = paths["volumes_dir"]

    if local_rank == 0:
        if args.eval_mode == "real_vs_recon":
            synth_glob = "recon_*.nii.gz"
        elif args.eval_mode == "real_vs_real":
            synth_glob = "other_*.nii.gz"
        else:
            synth_glob = f"gen_*_{args.postfix}.nii.gz"
        for fl, pattern in ((paths["real_filelist"], "base_*.nii.gz"),
                            (paths["synth_filelist"], synth_glob)):
            if fl and not os.path.isfile(fl):
                names = sorted(p.name for p in Path(dataset_root).glob(pattern))
                with open(fl, "w") as f:
                    f.write("\n".join(names) + ("\n" if names else ""))
    if dist.is_initialized():
        dist.barrier()

    with open(paths["real_filelist"]) as rf:
        real_lines = sorted(l.strip() for l in rf.readlines())[:args.num_images]
    with open(paths["synth_filelist"]) as sf:
        synth_lines = sorted(l.strip() for l in sf.readlines())[:args.num_images]
    real_fn = [{"image": os.path.join(dataset_root, f)} for f in real_lines]
    synth_fn = [{"image": os.path.join(dataset_root, f)} for f in synth_lines]
    part = lambda d: monai.data.partition_dataset(
        data=d, shuffle=False, num_partitions=world_size, even_divisible=False)[local_rank]
    real_fn, synth_fn = part(real_fn), part(synth_fn)

    transforms = build_fid_transforms(target_shape_tuple, rs_spacing_tuple,
                                      enable_resampling, args.fid_padding,
                                      args.fid_center_cropping,
                                      args.orientation_axcodes,
                                      args.intensity_norm_metric)

    def _feats(fnlist, label):
        loader = monai.data.DataLoader(
            monai.data.Dataset(data=fnlist, transform=transforms),
            num_workers=6, batch_size=1, shuffle=False)
        out = []
        for idx, batch in enumerate(loader, start=1):
            img = batch["image"].to(device)
            out.append(get_features_3d(img.as_tensor(), feature_network).cpu())
            logger.info(f"[Rank {local_rank}] med3d {label} {idx}/{len(fnlist)}")
        return torch.vstack(out) if out else torch.zeros(0, 512)

    real_f = _feats(real_fn, "Real")
    synth_f = _feats(synth_fn, "Synth")

    def _gather(ft):
        if not dist.is_initialized():
            return ft
        ls = torch.tensor([ft.shape[0]], dtype=torch.int64, device=device)
        rs = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(rs, ls)
        maxn = max(int(x.item()) for x in rs)
        pad = pad_to_max_size(ft.to(device), maxn)
        gl = [torch.empty_like(pad) for _ in range(world_size)]
        dist.all_gather(gl, pad)
        return torch.vstack([gl[i][:int(rs[i].item())].cpu() for i in range(world_size)])

    real_all, synth_all = _gather(real_f), _gather(synth_f)
    if local_rank == 0:
        logger.info(f"[med3d] real {tuple(real_all.shape)} synth {tuple(synth_all.shape)}")
        val = float(FIDMetric()(synth_all, real_all))
        logger.info(f"FID (med3d 3D): {val}")
        logger.info(f"FID Avg: {val}")   # harvester-compatible ('FID Avg:' line)


def run_fid(args, paths, device, local_rank, world_size):
    # med3d is a TRUE-3D extractor (whole-volume features, single FID) — entirely
    # different aggregation from the 2.5D XY/YZ/ZX path, so it dispatches out here.
    if args.fid_model_name == "med3d":
        return run_fid_3d(args, paths, device, local_rank, world_size)

    woodland = getattr(args, "fid_woodland", False)
    # Woodland-exact = axial-only + content-fraction filter; center-slices ratio does
    # not apply (all axial slices considered, kept by content, not by central band).
    enable_center_slices = (not woodland) and (args.fid_center_slices_ratio is not None)
    enable_resampling = args.fid_resampling_spacing is not None

    if local_rank == 0:
        logger.info(f"FID model              : {args.fid_model_name}")
        logger.info(f"FID target shape       : {args.fid_target_shape}")
        logger.info(f"woodland_exact         : {woodland} (content_frac={args.fid_content_frac})")
        logger.info(f"enable_center_slices   : {enable_center_slices} (ratio={args.fid_center_slices_ratio})")
        logger.info(f"enable_padding         : {args.fid_padding}")
        logger.info(f"enable_center_cropping : {args.fid_center_cropping}")
        logger.info(f"enable_resampling      : {enable_resampling} (spacing={args.fid_resampling_spacing})")
        logger.info(f"ignore_existing        : {args.fid_ignore_existing}")

    # Each 2.5D extractor pulls its weights from its own arg.
    weight_by_model = {
        "radimagenet_resnet50": args.feature_extractor_path,
        "imagenet_inception": None,          # torchvision downloads / caches ImageNet weights
        "imagenet_swav": args.swav_weight,
        "dinov2": args.dino_weight,
    }
    feature_network = load_feature_network(
        args.fid_model_name, device, local_rank,
        weight_path=weight_by_model.get(args.fid_model_name, args.feature_extractor_path),
        dino_repo=args.dino_repo, dino_arch=args.dino_arch)
    # Extractor-specific per-slice normalisation. RadImageNet keeps the validated
    # default; Inception needs [-1,1] @ 299x299; SwAV/DINOv2 need ImageNet mean/std @ 224.
    if woodland:
        feat_norm = woodland_rgb_normalisation           # 256^2 pad, no resize
    elif args.fid_model_name == "imagenet_inception":
        feat_norm = inception_intensity_normalisation
    elif args.fid_model_name in ("imagenet_swav", "dinov2"):
        feat_norm = imagenet_rgb_normalisation
    else:
        feat_norm = radimagenet_intensity_normalisation

    target_shape_tuple = tuple(int(x) for x in args.fid_target_shape.split("x"))
    if enable_resampling:
        rs_spacing_tuple = tuple(float(x) for x in args.fid_resampling_spacing.split("x"))
    else:
        rs_spacing_tuple = (1.0, 1.0, 1.0)
    center_slices_ratio_final = args.fid_center_slices_ratio if enable_center_slices else 1.0

    dataset_root = paths["volumes_dir"]
    output_root_real = os.path.join(paths["features_root"], paths["real_features_dir"])
    output_root_synth = os.path.join(paths["features_root"], paths["synth_features_dir"])

    # The fid phase reads filelist_{real,synth}_N.txt (basenames under
    # volumes_dir), but the generate step only saves the volumes and never wrote
    # these lists -> a bare `--phase fid` (or the fid leg of `--phase all`) died
    # with FileNotFoundError. Build them here from whatever volumes exist (rank 0
    # writes, all ranks wait at the barrier). A pre-existing filelist is kept
    # as-is (e.g. a manually trimmed N=126 list), so this only fills the gap.
    #   real  = base_*                 synth = recon_*            (real_vs_recon)
    #                                         = gen_*_<postfix>*  (real_vs_gen)
    #                                         = other_*           (real_vs_real, null floor)
    if local_rank == 0:
        if args.eval_mode == "real_vs_recon":
            synth_glob = "recon_*.nii.gz"
        elif args.eval_mode == "real_vs_real":
            synth_glob = "other_*.nii.gz"
        else:
            synth_glob = f"gen_*_{args.postfix}.nii.gz"
        for fl, pattern in ((paths["real_filelist"], "base_*.nii.gz"),
                            (paths["synth_filelist"], synth_glob)):
            if fl and not os.path.isfile(fl):
                names = sorted(p.name for p in Path(dataset_root).glob(pattern))
                with open(fl, "w") as f:
                    f.write("\n".join(names) + ("\n" if names else ""))
                logger.info(f"[fid] wrote {os.path.basename(fl)} "
                            f"({len(names)} files matching {pattern})")
    if dist.is_initialized():
        dist.barrier()

    # ----- real -----
    with open(paths["real_filelist"]) as rf:
        real_lines = sorted(l.strip() for l in rf.readlines())[:args.num_images]
    real_filenames = [{"image": os.path.join(dataset_root, f)} for f in real_lines]
    real_filenames = monai.data.partition_dataset(
        data=real_filenames, shuffle=False,
        num_partitions=world_size, even_divisible=False,
    )[local_rank]

    # ----- synth -----
    with open(paths["synth_filelist"]) as sf:
        synth_lines = sorted(l.strip() for l in sf.readlines())[:args.num_images]
    synth_filenames = [{"image": os.path.join(dataset_root, f)} for f in synth_lines]
    synth_filenames = monai.data.partition_dataset(
        data=synth_filenames, shuffle=False,
        num_partitions=world_size, even_divisible=False,
    )[local_rank]

    transforms = build_fid_transforms(target_shape_tuple, rs_spacing_tuple,
                                      enable_resampling, args.fid_padding,
                                      args.fid_center_cropping,
                                      args.orientation_axcodes,
                                      args.intensity_norm_metric)

    real_loader = monai.data.DataLoader(
        monai.data.Dataset(data=real_filenames, transform=transforms),
        num_workers=6, batch_size=1, shuffle=False)
    synth_loader = monai.data.DataLoader(
        monai.data.Dataset(data=synth_filenames, transform=transforms),
        num_workers=6, batch_size=1, shuffle=False)

    real_xy, real_yz, real_zx = _extract_features(
        real_loader, dataset_root, output_root_real, feature_network, device,
        enable_center_slices, center_slices_ratio_final,
        args.fid_ignore_existing, "Real data", local_rank, len(real_filenames),
        feat_norm=feat_norm, xy_only=woodland,
        content_filter=woodland, content_frac=args.fid_content_frac)
    logger.info(f"Real feature shapes: {real_xy.shape}, {real_yz.shape}, {real_zx.shape}")

    synth_xy, synth_yz, synth_zx = _extract_features(
        synth_loader, dataset_root, output_root_synth, feature_network, device,
        enable_center_slices, center_slices_ratio_final,
        args.fid_ignore_existing, "Synth data", local_rank, len(synth_filenames),
        feat_norm=feat_norm, xy_only=woodland,
        content_filter=woodland, content_frac=args.fid_content_frac)
    logger.info(f"Synth feature shapes: {synth_xy.shape}, {synth_yz.shape}, {synth_zx.shape}")

    del feature_network
    torch.cuda.empty_cache()
    logger.info("Feature network deleted to free GPU memory.")

    features = [real_xy, real_yz, real_zx, synth_xy, synth_yz, synth_zx]

    # all_gather sizes
    local_sizes = [torch.tensor([f.shape[0]], dtype=torch.int64, device=device)
                   for f in features]
    all_sizes = []
    for ls in local_sizes:
        rs = [torch.tensor([0], dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(rs, ls)
        all_sizes.append(rs)

    all_tensors_list = []
    for ft_idx, ft in enumerate(features):
        max_size = max(all_sizes[ft_idx]).item()
        ft_padded = pad_to_max_size(ft, max_size).to(device)
        gather_list_gpu = [torch.empty_like(ft_padded) for _ in range(world_size)]
        dist.all_gather(gather_list_gpu, ft_padded)
        gather_list_cpu = [t.cpu() for t in gather_list_gpu]
        del ft_padded, gather_list_gpu
        torch.cuda.empty_cache()
        for rk in range(world_size):
            gather_list_cpu[rk] = gather_list_cpu[rk][: all_sizes[ft_idx][rk], :]
        all_tensors_list.append(gather_list_cpu)

    if local_rank == 0:
        logger.info("Gathering complete. Computing FID on CPU...")
        del features, real_xy, real_yz, real_zx, synth_xy, synth_yz, synth_zx
        torch.cuda.empty_cache()

        rxy = torch.vstack(all_tensors_list[0])
        ryz = torch.vstack(all_tensors_list[1])
        rzx = torch.vstack(all_tensors_list[2])
        sxy = torch.vstack(all_tensors_list[3])
        syz = torch.vstack(all_tensors_list[4])
        szx = torch.vstack(all_tensors_list[5])

        logger.info(f"Final Real shapes : {rxy.shape}, {ryz.shape}, {rzx.shape}")
        logger.info(f"Final Synth shapes: {sxy.shape}, {syz.shape}, {szx.shape}")

        fid = FIDMetric()
        logger.info(f"Computing FID for: {output_root_real} | {output_root_synth}")
        fid_xy = fid(sxy, rxy)
        fid_yz = fid(syz, ryz)
        fid_zx = fid(szx, rzx)
        logger.info(f"FID XY : {fid_xy}")
        logger.info(f"FID YZ : {fid_yz}")
        logger.info(f"FID ZX : {fid_zx}")
        logger.info(f"FID Avg: {(fid_xy + fid_yz + fid_zx) / 3.0}")

        # ---- FID term decomposition (mean-shift vs covariance) ----
        # Diagnoses WHY rFID is high: a large mean-term = coherent systematic bias
        # (e.g. uniform blur / high-freq loss — the "paired metrics fine, rFID high"
        # fingerprint), a large cov-term = diversity/spread mismatch. Additive log
        # only; sum matches the FID values above (reuses monai _cov/_sqrtm).
        if getattr(args, "fid_decompose", True):
            m_sum = c_sum = t_sum = 0.0
            for name, (s, r) in (("XY", (sxy, rxy)), ("YZ", (syz, ryz)), ("ZX", (szx, rzx))):
                tot, m_t, c_t = frechet_terms(s, r)
                m_sum += m_t; c_sum += c_t; t_sum += tot
                logger.info(f"FID {name} decomp: total {tot:.4f} = mean-term {m_t:.4f} "
                            f"({100*m_t/tot:.1f}%) + cov-term {c_t:.4f} ({100*c_t/tot:.1f}%)")
            logger.info(f"FID Avg decomp: total {t_sum/3:.4f} = mean-term {m_sum/3:.4f} "
                        f"({100*m_sum/t_sum:.1f}%) + cov-term {c_sum/3:.4f} ({100*c_sum/t_sum:.1f}%)")

        # ---- bootstrap standard error (finite-sample FID uncertainty) ----
        # Resample synth + real feature rows (with replacement) B times → bootstrap
        # std = standard error of the FID estimate. Cheap (CPU, reuses gathered
        # features), reproducible via --seed.
        #   ⚠️ FID is biased UPWARD as effective sample size shrinks, so the
        #   resample MEAN sits above the full-sample point estimate — do NOT report
        #   the resample mean as the FID. We report the POINT estimate as the value
        #   and the bootstrap std as its SE (CI = point ± 1.96·SE). For MODEL-vs-MODEL
        #   gaps (MAISI↔SID↔VAD) this bias is common-mode and cancels in the
        #   difference, so the SE is what tells you whether a ~1pt gap is real.
        if getattr(args, "fid_bootstrap", 0) and args.fid_bootstrap > 0:
            B = int(args.fid_bootstrap)
            rng = np.random.default_rng(args.seed)
            point = {"XY": float(fid_xy), "YZ": float(fid_yz), "ZX": float(fid_zx)}
            point["AVG"] = sum(point.values()) / 3.0
            planes = {"XY": (sxy, rxy), "YZ": (syz, ryz), "ZX": (szx, rzx)}
            n_s, n_r = sxy.shape[0], rxy.shape[0]
            boot = {k: [] for k in planes}
            boot["AVG"] = []
            logger.info(f"FID bootstrap SE: B={B}, n_synth={n_s}, n_real={n_r}, seed={args.seed}")
            for _ in range(B):
                si = torch.from_numpy(rng.integers(0, n_s, n_s))
                ri = torch.from_numpy(rng.integers(0, n_r, n_r))
                vals = []
                for k, (s, r) in planes.items():
                    v = float(fid(s[si], r[ri]))
                    boot[k].append(v)
                    vals.append(v)
                boot["AVG"].append(sum(vals) / 3.0)
            for k in ("XY", "YZ", "ZX", "AVG"):
                se = float(np.asarray(boot[k]).std())
                rmean = float(np.asarray(boot[k]).mean())
                p = point[k]
                logger.info(f"FID {k:3s}: point {p:.4f}  SE {se:.4f}  "
                            f"~95% CI [{p - 1.96 * se:.4f}, {p + 1.96 * se:.4f}]  "
                            f"(resample mean {rmean:.4f} — biased↑, not the estimate)")


# =============================================================================
# Main orchestration
# =============================================================================
def main():
    args = load_config()
    paths = derive_paths(args)
    local_rank, world_size, device = init_distributed()

    if local_rank == 0:
        print_config()
        logger.info("=" * 72)
        logger.info("RUN CONFIG (resolved args + hyperparams)")
        logger.info("=" * 72)
        logger.info(f"  running on        : {device} (rank {local_rank}/{world_size})")
        logger.info(f"  exp_dir           : {args.exp_dir}")
        logger.info(f"  eval_mode         : {args.eval_mode}")
        logger.info(f"  phase             : {args.phase}")
        logger.info(f"  num_images        : {args.num_images}")
        logger.info(f"  seed              : {args.seed}")
        logger.info("  -- checkpoints --")
        logger.info(f"  pretrained_vae_path : {args.pretrained_vae_path}")
        logger.info(f"  pretrained_unet_path: {args.pretrained_unet_path}")
        logger.info("  -- generation / inference hyperparams --")
        for k in ("num_inference_steps", "scale_factor", "global_mean",
                  "stochastic_scale", "latent_channels", "latent_shape",
                  "inference_spacing", "weight_dtype", "amp", "postfix",
                  "guidance_scale"):
            if hasattr(args, k):
                logger.info(f"    {k:22s}: {getattr(args, k)}")
        logger.info("  -- preprocessing / resolution --")
        for k in ("resolution", "orientation_axcodes", "fid_resolution",
                  "fid_target_shape", "intensity_norm_metric"):
            if hasattr(args, k):
                logger.info(f"    {k:22s}: {getattr(args, k)}")
        logger.info("  -- FID params --")
        for k in ("fid_model_name", "fid_resampling_spacing",
                  "fid_center_slices_ratio", "fid_padding",
                  "fid_center_cropping", "fid_ignore_existing",
                  "swav_weight", "dino_repo", "dino_weight", "dino_arch",
                  "fid_woodland", "fid_content_frac"):
            if hasattr(args, k):
                logger.info(f"    {k:22s}: {getattr(args, k)}")
        logger.info("  -- paths --")
        for k in ("volumes_dir", "slices_dir", "features_root",
                  "real_filelist", "synth_filelist", "synth_features_dir"):
            logger.info(f"    {k:22s}: {paths[k]}")
        logger.info("=" * 72)

    set_determinism(seed=args.seed)

    if local_rank == 0:
        os.makedirs(paths["volumes_dir"], exist_ok=True)
        os.makedirs(paths["slices_dir"], exist_ok=True)
        os.makedirs(paths["features_root"], exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    if args.phase in ("generate", "all"):
        run_generation(args, paths, device, local_rank, world_size)
        if dist.is_initialized():
            dist.barrier()

    if args.phase in ("fid", "all"):
        run_fid(args, paths, device, local_rank, world_size)

    if local_rank == 0:
        print("\nAll tasks finished.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
