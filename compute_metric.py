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
    parser.add_argument("--postfix", type=str, default="30step")
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--fid_model_name", type=str, default="radimagenet_resnet50")
    parser.add_argument("--fid_resampling_spacing", type=str, default="1.0x1.0x1.0")
    parser.add_argument("--fid_center_slices_ratio", type=float, default=1.0)
    parser.add_argument("--fid_padding", type=lambda s: str(s).lower() == "true", default=True)
    parser.add_argument("--fid_center_cropping", type=lambda s: str(s).lower() == "true", default=True)
    parser.add_argument("--fid_ignore_existing", type=lambda s: str(s).lower() == "true", default=True)

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
        paths["synth_filelist"]     = None
        paths["synth_features_dir"] = None
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

    local_metrics = {"lpips": 0.0, "psnr": 0.0, "ssim": 0.0, "count": 0}

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
                    local_metrics["ssim"] += ssim(recon_clip.squeeze().cpu().numpy(),
                                                  images_clip.squeeze().cpu().numpy(),
                                                  data_range=1.0)
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

                if use_token_set:
                    from patches.token_set_encoder import encode_token_set
                    ci, cv, pr = encode_token_set(meta_values[i], cond_cfg["attributes"])
                    meta_tensor = {
                        "cond_cat": torch.tensor([ci], dtype=torch.long, device=device),
                        "cond_cont": torch.tensor([cv], dtype=torch.float32, device=device),
                        "cond_presence": torch.tensor([pr], dtype=torch.bool, device=device),
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

    # Aggregate recon metrics across ranks
    if args.eval_mode == "real_vs_recon":
        t = torch.tensor([local_metrics["lpips"], local_metrics["psnr"],
                          local_metrics["ssim"], float(local_metrics["count"])],
                         device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            count = t[3].item()
            if count > 0:
                lpips_avg = t[0].item() / count
                psnr_avg = t[1].item() / count
                ssim_avg = t[2].item() / count
                print("\n" + "=" * 50)
                print(f"Final Average Reconstruction Metrics (over {int(count)} imgs):")
                print(f"LPIPS : {lpips_avg:.4f}")
                print(f"PSNR  : {psnr_avg:.4f}")
                print(f"SSIM  : {ssim_avg:.4f}")
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
    dim = len(volume.shape)
    if dim == 4 and norm2d:
        return subtract_mean(volume)
    elif dim == 4:
        return subtract_mean(volume)
    if dim == 5:
        return subtract_mean(volume)
    return volume


def get_features_2p5d(image, feature_network, center_slices=False,
                     center_slices_ratio=1.0, sample_every_k=1,
                     xy_only=True, drop_empty=False, empty_threshold=-700):
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
        mapping_index = drop_empty_slice(slices, empty_threshold) if drop_empty \
                        else [True] * len(slices)
        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
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
        mapping_index = drop_empty_slice(slices, empty_threshold) if drop_empty \
                        else [True] * len(slices)
        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]
        feat_yz = spatial_average(feature_network.forward(images_2d), keepdim=False)

        # ZX (along W)
        if center_slices:
            start_w = int((1.0 - center_slices_ratio) / 2.0 * W)
            end_w = int((1.0 + center_slices_ratio) / 2.0 * W)
            slices = torch.unbind(image[:, :, :, start_w:end_w:sample_every_k, :], dim=3)
        else:
            slices = torch.unbind(image, dim=3)
        mapping_index = drop_empty_slice(slices, empty_threshold) if drop_empty \
                        else [True] * len(slices)
        images_2d = torch.cat(slices, dim=0)
        images_2d = radimagenet_intensity_normalisation(images_2d)
        images_2d = images_2d[mapping_index]
        feat_zx = spatial_average(feature_network.forward(images_2d), keepdim=False)

    return feat_xy, feat_yz, feat_zx


def pad_to_max_size(tensor, max_size, padding_value=0.0):
    pad_size = [0, 0] * (len(tensor.shape) - 1) + [0, max_size - tensor.shape[0]]
    return F.pad(tensor, pad_size, "constant", padding_value)


def load_feature_network(model_name, device, local_rank, weight_path=None):
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
                      ignore_existing, label, local_rank, num_files):
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
                xy_only=False,
            )
            logger.info(f"feats shapes: {feats[0].shape}, {feats[1].shape}, {feats[2].shape}")
            feats = tuple(f.cpu() for f in feats)
            torch.save(feats, out_fp)

        feats_xy.append(feats[0])
        feats_yz.append(feats[1])
        feats_zx.append(feats[2])

    return torch.vstack(feats_xy), torch.vstack(feats_yz), torch.vstack(feats_zx)


def run_fid(args, paths, device, local_rank, world_size):
    enable_center_slices = args.fid_center_slices_ratio is not None
    enable_resampling = args.fid_resampling_spacing is not None

    if local_rank == 0:
        logger.info(f"FID model              : {args.fid_model_name}")
        logger.info(f"FID target shape       : {args.fid_target_shape}")
        logger.info(f"enable_center_slices   : {enable_center_slices} (ratio={args.fid_center_slices_ratio})")
        logger.info(f"enable_padding         : {args.fid_padding}")
        logger.info(f"enable_center_cropping : {args.fid_center_cropping}")
        logger.info(f"enable_resampling      : {enable_resampling} (spacing={args.fid_resampling_spacing})")
        logger.info(f"ignore_existing        : {args.fid_ignore_existing}")

    feature_network = load_feature_network(args.fid_model_name, device, local_rank,
                                           weight_path=args.feature_extractor_path)

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
    if local_rank == 0:
        synth_glob = ("recon_*.nii.gz" if args.eval_mode == "real_vs_recon"
                      else f"gen_*_{args.postfix}.nii.gz")
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
        args.fid_ignore_existing, "Real data", local_rank, len(real_filenames))
    logger.info(f"Real feature shapes: {real_xy.shape}, {real_yz.shape}, {real_zx.shape}")

    synth_xy, synth_yz, synth_zx = _extract_features(
        synth_loader, dataset_root, output_root_synth, feature_network, device,
        enable_center_slices, center_slices_ratio_final,
        args.fid_ignore_existing, "Synth data", local_rank, len(synth_filenames))
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
                  "inference_spacing", "weight_dtype", "amp", "postfix"):
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
                  "fid_center_cropping", "fid_ignore_existing"):
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

    if args.phase in ("fid", "all") and args.eval_mode != "real_vs_real":
        run_fid(args, paths, device, local_rank, world_size)

    if local_rank == 0:
        print("\nAll tasks finished.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
