"""Synthesize BraTS-like volumes conditioned on tumor masks (ControlNet).

Pipeline:

    real masks ─► ControlNet(noise, t, mask) ──► down_block/mid residuals
                                                 │
                                                 ▼
                                    UNet(noise, t, additional_residuals=...)
                                                 │
                                                 ▼
                                          VAE.decode → NIfTI

Output: ``<output_dir>/{volumes/*-t1n.nii.gz, *-seg.nii.gz, synth_seg_index.csv}``
matching the schema expected by ``downstream.seg_dataset.make_dataset``.

NOTE: skeleton. Real use depends on a trained ControlNet (see
``train_CONTROLNET.py``, TODO — adapt from
https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/blob/main/scripts/train_controlnet.py).
Without that checkpoint, run with ``--dry_run`` to validate path wiring +
CSV emission.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

from scripts.utils import define_instance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_adapter", default="brats")
    p.add_argument("--dataset_config_path", required=True)
    p.add_argument("--model_config_path", required=True,
                   help="model_fm.json — the base UNet (unconditional).")
    p.add_argument("--inference_config_path", required=True)
    p.add_argument("--pretrained_vae_path", required=True)
    p.add_argument("--pretrained_unet_path", required=True,
                   help="Base (unconditional) UNet checkpoint.")
    p.add_argument("--pretrained_controlnet_path", required=True,
                   help="ControlNetMaisi checkpoint (model.pt).")
    p.add_argument("--latent_stats_csv", required=True)
    p.add_argument("--mask_csv", required=True,
                   help="CSV with rel_path_seg column — masks to condition on.")
    p.add_argument("--mask_data_dir", required=True,
                   help="Root dir for rel_path_seg paths.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=1125,
                   help="Defaults to BraTS train set size. >len(mask_csv) loops with reshuffle.")
    p.add_argument("--num_inference_steps", type=int, default=1000)
    p.add_argument("--stochastic_scale", type=float, default=0.0)
    p.add_argument("--conditioning_scale", type=float, default=1.0)
    p.add_argument("--copy_mask", action="store_true", default=True,
                   help="Copy the conditioning mask into the output dir as the seg pair.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    vol_dir = out_dir / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "synth_seg_index.csv"

    with open(args.dataset_config_path) as f:
        ds_cfg = json.load(f)
    with open(args.model_config_path) as f:
        model_cfg = json.load(f)
    with open(args.inference_config_path) as f:
        inf_cfg = json.load(f)

    mask_df = pd.read_csv(args.mask_csv)
    rng = np.random.default_rng(args.seed)
    mask_indices = rng.integers(0, len(mask_df), size=args.num_samples)

    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eid", "rel_path", "rel_path_seg"])
        for i, mi in enumerate(mask_indices):
            eid = f"synth_{i:06d}"
            w.writerow([eid, f"volumes/{eid}-t1n.nii.gz", f"volumes/{eid}-seg.nii.gz"])
    print(f"[index] {index_path} ({len(mask_indices)} rows)")

    if args.dry_run:
        print("[dry_run] skipping model load + inference.")
        return

    cfg_args = argparse.Namespace(**{**vars(args), **ds_cfg, **model_cfg, **inf_cfg})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = define_instance(cfg_args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(os.path.join(args.pretrained_vae_path, "model.pt"),
                         map_location=device, weights_only=False)
    autoencoder.load_state_dict(ae_ckpt["autoencoder"])
    autoencoder.eval()

    unet = define_instance(cfg_args, "diffusion_unet_def").to(device)
    unet_ckpt = torch.load(os.path.join(args.pretrained_unet_path, "model.pt"),
                           map_location=device, weights_only=False)
    unet.load_state_dict(unet_ckpt.get("unet", unet_ckpt.get("unet_state_dict")), strict=True)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    # ControlNetMaisi must be instantiable via define_instance with a
    # `controlnet_def` block in model_fm.json. TODO: extend model_fm.json
    # with controlnet_def once train_CONTROLNET.py is written.
    controlnet = define_instance(cfg_args, "controlnet_def").to(device)
    cn_ckpt = torch.load(os.path.join(args.pretrained_controlnet_path, "model.pt"),
                         map_location=device, weights_only=False)
    controlnet.load_state_dict(cn_ckpt.get("controlnet", cn_ckpt.get("controlnet_state_dict")), strict=True)
    controlnet.eval()

    noise_scheduler = define_instance(cfg_args, "noise_scheduler")
    latent_shape = ds_cfg["latent_shape"]
    noise_scheduler.set_timesteps(num_inference_steps=args.num_inference_steps,
                                  input_img_size_numel=int(np.prod(latent_shape)))

    stats = pd.read_csv(args.latent_stats_csv)
    global_mean = float(stats.iloc[0]["global_mean"])
    scale_factor = float(stats.iloc[0]["scale_factor"])

    latent_channels = cfg_args.latent_channels
    inference_spacing = ds_cfg.get("inference_spacing", [1.0, 1.0, 1.0])
    spacing_tensor = torch.tensor([[s * 1e2 for s in inference_spacing]],
                                  dtype=torch.float16, device=device)

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:],
                                    torch.tensor([0], dtype=all_timesteps.dtype)))

    for i, mi in enumerate(mask_indices):
        eid = f"synth_{i:06d}"
        vol_out = vol_dir / f"{eid}-t1n.nii.gz"
        seg_out = vol_dir / f"{eid}-seg.nii.gz"
        if vol_out.exists() and seg_out.exists():
            continue
        torch.manual_seed(args.seed + i)

        mask_src = os.path.join(args.mask_data_dir, mask_df.iloc[mi]["rel_path_seg"])
        mask_img = nib.load(mask_src)
        mask_np = mask_img.get_fdata().astype(np.float32)
        # ControlNet conditioning expects the mask resampled/cropped to the same
        # spatial footprint as the latent input (post-VAE-downsample). Quick
        # nearest-neighbor resize via numpy for skeleton; replace with MONAI
        # Resized + ConvertToMultiChannel when this hits real use.
        cond = torch.from_numpy(mask_np)[None, None].to(device).float()
        cond = torch.nn.functional.interpolate(cond, size=tuple(latent_shape),
                                                mode="nearest")

        latent = torch.randn((1, latent_channels, *latent_shape), device=device)
        with torch.no_grad(), autocast(dtype=torch.float16, enabled=args.amp):
            for t, next_t in zip(all_timesteps, all_next_timesteps):
                t_tensor = torch.tensor((t,), device=device)
                down_res, mid_res = controlnet(x=latent, timesteps=t_tensor,
                                                controlnet_cond=cond,
                                                conditioning_scale=args.conditioning_scale)
                model_out = unet(x=latent, timesteps=t_tensor,
                                 spacing_tensor=spacing_tensor,
                                 down_block_additional_residuals=down_res,
                                 mid_block_additional_residual=mid_res)
                latent, _ = noise_scheduler.step(model_out, t, latent, next_t,
                                                 args.stochastic_scale)
            vol = autoencoder.decode_stage_2_outputs((latent / scale_factor) + global_mean)
        arr = vol.squeeze().float().cpu().numpy()
        nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), vol_out)
        if args.copy_mask:
            shutil.copyfile(mask_src, seg_out)
        if (i + 1) % 20 == 0:
            print(f"[gen] {i + 1}/{len(mask_indices)}", flush=True)


if __name__ == "__main__":
    main()
