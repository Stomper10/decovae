"""Stage 1 of the 3D-MedDiffusion reconstruction-metric pipeline (3d_meddiff env).

Produces `base_{i}.nii.gz` (real) and `recon_{i}.nii.gz` (3D MedDiff VQ
reconstruction) in compute_metric's volume layout, so the SAME metric code
(LPIPS/PSNR/SSIM in deco_v15, rFID via compute_metric --phase fid) can score
them identically to MAISI / L_SID / L_VAD.

Fairness: the real `base` volume uses the EXACT compute_metric.build_transforms
pipeline (Orientation -> percentile [0,1] -> trilinear resize to fid_resolution),
so the real reference is processed identically to the MAISI eval. Only the model
under test changes. Reconstruction uses the true VQ path (encode -> codebook
quantize -> decode), at fid_resolution (128x256x128 for UKB) — the same eval
resolution MAISI reconstructs at. 128/256/128 are multiples of patch_size=64.

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


def build_transform(cfg):
    """Replicates compute_metric.build_transforms (the `transform` branch)."""
    inten = cfg["intensity_norm_metric"]
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes=cfg["orientation_axcodes"]),
        ScaleIntensityRangePercentilesd(keys="image", lower=inten["lower"],
                                        upper=inten["upper"], b_min=inten["b_min"],
                                        b_max=inten["b_max"], clip=inten.get("clip", True)),
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
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    with open(args.dataset_config) as f:
        cfg = json.load(f)
    transform = build_transform(cfg)

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
                recon01 = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)
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
