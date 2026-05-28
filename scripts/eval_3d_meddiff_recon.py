"""Stage 1 of the 3D-MedDiffusion reconstruction-metric pipeline (3d_meddiff env).

Produces `base_{i}.nii.gz` (real) and `recon_{i}.nii.gz` (3D MedDiff VQ
reconstruction) in compute_metric's volume layout, so the SAME metric code
(LPIPS/PSNR/SSIM in deco_v15, rFID via compute_metric --phase fid) can score
them identically to MAISI / L_SID / L_VAD.

Fairness: the real `base` volume uses the compute_metric.build_transforms
pipeline (Orientation -> percentile [0,1] -> trilinear resize to fid_resolution),
at fid_resolution (128x256x128 for UKB) — the same eval resolution MAISI
reconstructs at. 128/256/128 are multiples of patch_size=64.

Intensity normalization deviates from the dataset's `intensity_norm_metric`
(lower=0.0) on purpose: the PatchVolume AE was trained — and its latents were
extracted (scripts/extract_3d_meddiff_latents.py) — with
tio.RescaleIntensity(percentiles=(0.5, 99.5)). Feeding it the lower=0.0
normalization (true-min -> 0) leaves a ~0.044 background floor the VQ codebook
maps to ~0 in the recon, which crushes whole-volume SSIM even though the
foreground recon is faithful (fg corr ~0.99). We therefore normalize the base
with the AE's training percentiles (0.5, 99.5) so base and recon share the same
background level — self-consistent for measuring 3D MedDiff's recon fidelity.
Reconstruction uses the true VQ path (encode -> codebook quantize -> decode).

The PatchVolume AE was trained with heavy flip/rot90 augmentation, so it is
orientation-robust; we feed the monai-layout volume directly (only [0,1]->[-1,1]).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged, Orientationd,
                              ScaleIntensityRangePercentilesd, Resized)

_EXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "3d_meddiff"))
sys.path.insert(0, _EXT)
from AutoEncoder.model.PatchVolume import patchvolumeAE  # noqa: E402


def build_transform(cfg, norm_lower=0.5, norm_upper=99.5):
    """compute_metric.build_transforms with a configurable intensity percentile.

    Two regimes (see module docstring):
      - norm_lower=0.5 (default): the AE's training intensity norm
        (percentiles 0.5-99.5) -> background floor matches the recon's, giving a
        faithful whole-volume SSIM/PSNR/LPIPS. Use for PAIRED reconstruction
        metrics (3D MedDiff in its native operating space).
      - norm_lower=0.0: the dataset's `intensity_norm_metric`, identical to how
        MAISI/L_SID/L_VAD are scored in compute_metric.sh. Use for rFID/gFID so
        the FID feature distribution (hence the FID scale) matches the baselines.
        The lower=0.5 clip is destructive and deflates FID ~45x vs lower=0.0,
        so distribution metrics MUST use lower=0.0 to be comparable.
    """
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes=cfg["orientation_axcodes"]),
        ScaleIntensityRangePercentilesd(keys="image", lower=norm_lower, upper=norm_upper,
                                        b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=tuple(cfg["fid_resolution"]), mode="trilinear"),
        EnsureTyped(keys="image", dtype=torch.float32),
    ])


@torch.no_grad()
def reconstruct(ae, x, patch=64):
    """True VQ reconstruction: unfold -> encode -> codebook -> decode (PatchVolume.forward val path)."""
    B, C, D, H, W = x.shape
    Dc, Hc, Wc = (D // patch) * patch, (H // patch) * patch, (W // patch) * patch
    x = x[:, :, :Dc, :Hc, :Wc]
    xi = x.unfold(2, patch, patch).unfold(3, patch, patch).unfold(4, patch, patch)
    xi = rearrange(xi, "b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w")
    z = ae.pre_vq_conv(ae.encoder(xi))
    emb = ae.codebook(z)["embeddings"]
    emb = rearrange(emb, "(b p) c d h w -> b p c d h w", b=B)
    emb = rearrange(emb, "b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)",
                    p1=Dc // patch, p2=Hc // patch, p3=Wc // patch)
    return ae.decoder(ae.post_vq_conv(emb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--base-csv", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ae-ckpt", required=True)
    ap.add_argument("--out-dir", required=True, help="volumes dir (base_/recon_ saved here)")
    ap.add_argument("--num-images", type=int, default=2500)
    ap.add_argument("--norm-lower", type=float, default=0.5,
                    help="lower intensity percentile: 0.5 (paired, AE-native) | "
                         "0.0 (rFID/gFID, baseline-matched scale)")
    ap.add_argument("--norm-upper", type=float, default=99.5)
    ap.add_argument("--no-clamp", action="store_true",
                    help="save recon WITHOUT clamping to [0,1] (match compute_metric/MAISI "
                         "which saves the raw decoder output unclamped)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    with open(args.dataset_config) as f:
        cfg = json.load(f)
    transform = build_transform(cfg, norm_lower=args.norm_lower, norm_upper=args.norm_upper)
    print(f"[recon] intensity norm percentiles=({args.norm_lower}, {args.norm_upper})", flush=True)

    df = pd.read_csv(args.base_csv)
    files = [{"image": os.path.join(args.data_dir, rel)} for rel in df["rel_path"]][: args.num_images]
    idxs = list(range(len(files)))[args.shard_index :: args.num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[recon] shard {args.shard_index}/{args.num_shards}: {len(idxs)} of {len(files)} -> {args.out_dir}",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = patchvolumeAE.load_from_checkpoint(args.ae_ckpt).to(device).eval()
    print("[recon] AE loaded", flush=True)

    for n, i in enumerate(idxs):
        base_path = os.path.join(args.out_dir, f"base_{i:04d}.nii.gz")
        recon_path = os.path.join(args.out_dir, f"recon_{i:04d}.nii.gz")
        if os.path.exists(base_path) and os.path.exists(recon_path):
            continue
        try:
            base01 = transform(files[i])["image"].unsqueeze(0).to(device)  # [1,1,Dx,Hy,Wz] in [0,1]
            with torch.no_grad():
                x_recon = reconstruct(ae, base01 * 2.0 - 1.0)
                recon01 = (x_recon + 1.0) / 2.0
                if not args.no_clamp:
                    recon01 = torch.clamp(recon01, 0.0, 1.0)
            nib.save(nib.Nifti1Image(base01.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), base_path)
            nib.save(nib.Nifti1Image(recon01.cpu().numpy().squeeze().astype(np.float32), np.eye(4)), recon_path)
        except Exception as e:  # noqa: BLE001
            print(f"[recon] ERROR i={i} ({files[i]['image']}): {e}", flush=True)
            continue
        if (n + 1) % 50 == 0 or n == 0:
            print(f"[recon] {n + 1}/{len(idxs)} (i={i}) base{tuple(base01.shape[2:])} "
                  f"recon{tuple(recon01.shape[2:])}", flush=True)

    print(f"[recon] DONE shard {args.shard_index}", flush=True)


if __name__ == "__main__":
    main()
