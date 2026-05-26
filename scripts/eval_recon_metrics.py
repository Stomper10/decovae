"""Stage 2: LPIPS / PSNR / SSIM from saved base_/recon_ volume pairs (deco_v15 env).

Computes the reconstruction metrics with the EXACT definitions used by
compute_metric.py:run_generation (real_vs_recon), so the numbers line up with
the MAISI / L_SID / L_VAD rows in results_ukb_quality.csv:
  - LPIPS : monai PerceptualLoss(spatial_dims=3, network_type="squeeze",
            is_fake_3d=True, fake_3d_ratio=0.2) on [0,1]-clamped volumes
  - PSNR  : scripts.utils.compute_psnr on [0,1]-clamped volumes
  - SSIM  : skimage structural_similarity, data_range=1.0

rFID (2.5D radimagenet) is computed separately by re-running compute_metric.py
--phase fid on the same volumes dir (identical FID code).
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from monai.losses.perceptual import PerceptualLoss

from scripts.utils import compute_psnr


def load_vol(path, device):
    arr = np.asarray(nib.load(path).dataobj).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--run-name", default="3d_meddiff")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

    recon_files = sorted(glob.glob(os.path.join(args.volumes_dir, "recon_*.nii.gz")))
    pairs = []
    for rp in recon_files:
        idx = re.search(r"recon_(\d+)\.nii\.gz", os.path.basename(rp)).group(1)
        bp = os.path.join(args.volumes_dir, f"base_{idx}.nii.gz")
        if os.path.exists(bp):
            pairs.append((bp, rp))
    print(f"[metric] {len(pairs)} base/recon pairs in {args.volumes_dir}", flush=True)
    if not pairs:
        raise FileNotFoundError("no base_/recon_ pairs found")

    tot = {"lpips": 0.0, "psnr": 0.0, "ssim": 0.0, "n": 0}
    for bp, rp in pairs:
        base = torch.clamp(load_vol(bp, device), 0.0, 1.0)
        recon = torch.clamp(load_vol(rp, device), 0.0, 1.0)
        with torch.no_grad():
            tot["lpips"] += perceptual(recon, base).item()
        tot["psnr"] += compute_psnr(recon.cpu().numpy(), base.cpu().numpy())
        tot["ssim"] += ssim(recon.squeeze().cpu().numpy(), base.squeeze().cpu().numpy(), data_range=1.0)
        tot["n"] += 1
        if tot["n"] % 200 == 0:
            print(f"[metric] {tot['n']}/{len(pairs)}", flush=True)

    n = tot["n"]
    row = {
        "baseline": args.run_name,
        "eval_mode": "real_vs_recon",
        "num_images": n,
        "ssim": round(tot["ssim"] / n, 4),
        "psnr": round(tot["psnr"] / n, 4),
        "lpips": round(tot["lpips"] / n, 4),
        "rfid_2_5d": "",  # filled by compute_metric --phase fid (needs RadImageNet weight)
        "note": "recon at fid_resolution 128x256x128; metrics identical to compute_metric real_vs_recon",
    }
    print("\n" + "=" * 50)
    print(f"3D MedDiff reconstruction metrics (N={n}):")
    print(f"  SSIM  : {row['ssim']}")
    print(f"  PSNR  : {row['psnr']}")
    print(f"  LPIPS : {row['lpips']}")
    print("=" * 50)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame([row]).to_csv(args.out_csv, index=False)
    print(f"[metric] wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
