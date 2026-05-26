"""Latent geometry / stat analysis for 3D-MedDiffusion PatchVolume latents.

REFERENCE-ONLY. The geometry/stat metrics in `extract_emb.py` are designed for
continuous Gaussian (KL) latents. The 3D MedDiff latent here is the continuous
*pre-quantization* embedding rescaled to [-1, 1] by the VQ codebook range
(see scripts/extract_3d_meddiff_latents.py), so the channel-geometry metrics are
computable and meaningful — but two confounds make a head-to-head against the
4-channel MAISI/SID/VAD latents unfair:

  1. channel count:  3D MedDiff = 8,  MAISI/SID/VAD = 4.
  2. continuous KL vs VQ-codebook: there is no posterior sigma, and the values
     are bounded to [-1, 1] rather than ~Gaussian.

We therefore reproduce the exact formulas from extract_emb.py:run_geometry so the
raw numbers line up column-for-column with results_ukb_geometry.csv, and ADD two
dimension-normalized columns that ARE roughly comparable across channel counts:
  - effective_rank_ratio = effective_rank / C   (rank utilization, in [0, 1])
  - mean_l2_norm_ratio   = mean_l2_norm / sqrt(C)
`avg_pairwise_cos_sim` (concentration-of-measure confounded) and
`avg_single_sigma` (no VQ posterior) are reported as N/A for cross-comparison.

Usage:
  python scripts/analyze_3d_meddiff_latent_geometry.py \
    --latent-dir /data/wonyoungjang/decodata/3d_meddiff/ukb_c4/latents/train \
    --run-name 3d_meddiff --max-samples 5000 \
    --out-geometry .../latent_geometry_3dmd.csv \
    --out-stat     .../latent_stat_3dmd.csv
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_voxel_matrix(latent_dir: str, max_samples: int | None) -> tuple[torch.Tensor, int]:
    """Stack per-voxel channel vectors from .npy latents -> (N_voxels, C)."""
    paths = sorted(glob.glob(os.path.join(latent_dir, "*.npy")))
    if max_samples is not None:
        paths = paths[:max_samples]
    if not paths:
        raise FileNotFoundError(f"no .npy latents under {latent_dir}")
    chunks = []
    for p in paths:
        arr = np.load(p)                      # [C, D, H, W]
        C = arr.shape[0]
        chunks.append(arr.reshape(C, -1).T)   # (D*H*W, C)
    mat = np.concatenate(chunks, axis=0).astype(np.float32)
    print(f"[geometry] loaded {len(paths)} latents -> voxel matrix {mat.shape}", flush=True)
    return torch.from_numpy(mat), len(paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--run-name", default="3d_meddiff")
    ap.add_argument("--max-samples", type=int, default=5000,
                    help="number of latent files (match MAISI geometry n=5000)")
    ap.add_argument("--out-geometry", required=True)
    ap.add_argument("--out-stat", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_flat, n_files = load_voxel_matrix(args.latent_dir, args.max_samples)
    raw_flat = raw_flat.to(device)

    # --- stat / shift (extract_emb.py:run_geometry parity) -------------------
    global_mean = raw_flat.mean()
    global_std = raw_flat.std()
    scale_factor = 1.0 / global_std.clamp(min=1e-8)
    diff_flat = (raw_flat - global_mean) * scale_factor
    C = raw_flat.shape[1]
    print(f"[geometry] C={C} global_mean={global_mean.item():.4f} "
          f"global_std={global_std.item():.4f} scale_factor={scale_factor.item():.4f}",
          flush=True)

    off_diag = ~torch.eye(C, dtype=torch.bool, device=device)

    # [1] inter-channel correlation before shifting
    raw_normed = F.normalize(raw_flat, p=2, dim=0)
    raw_corr = torch.mm(raw_normed.t(), raw_normed)
    raw_cos_sim = raw_corr[off_diag].abs().mean().item()
    del raw_flat, raw_normed, raw_corr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # [2] inter-channel correlation after shifting
    diff_normed = F.normalize(diff_flat, p=2, dim=0)
    diff_corr = torch.mm(diff_normed.t(), diff_normed)
    diff_cos_sim = diff_corr[off_diag].abs().mean().item()
    del diff_normed, diff_corr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # [3] per-voxel norm + effective rank (cov of channels), sampled to 1M voxels
    N = diff_flat.shape[0]
    sample_size = min(N, 1_000_000)
    idx = torch.randint(0, N, (sample_size,), device=device)
    diff_sample = diff_flat[idx]
    l2_norms = torch.norm(diff_sample, p=2, dim=1)
    cov = torch.cov(diff_sample.t())
    eigvals = torch.linalg.eigvalsh(cov).real
    evr = torch.sort(eigvals, descending=True).values / eigvals.sum()
    effective_rank = torch.exp(-torch.sum(evr * torch.log(evr + 1e-8))).item()
    mean_l2 = l2_norms.mean().item()

    # [4] pairwise cosine between random voxel vectors (dim-confounded -> N/A xcmp)
    pair = 50_000
    a = torch.randint(0, N, (pair,), device=device)
    b = torch.randint(0, N, (pair,), device=device)
    va = F.normalize(diff_flat[a], p=2, dim=1)
    vb = F.normalize(diff_flat[b], p=2, dim=1)
    avg_pair_cos_sim = (va * vb).sum(dim=1).abs().mean().item()

    print(f"[geometry] raw_cos_sim={raw_cos_sim:.4f} diff_cos_sim={diff_cos_sim:.4f} "
          f"mean_l2={mean_l2:.4f} (ideal ~{C**0.5:.2f}) "
          f"effective_rank={effective_rank:.2f}/{C} "
          f"avg_pairwise_cos_sim={avg_pair_cos_sim:.4f}", flush=True)

    geom_row = {
        "run_name": args.run_name,
        "num_samples": n_files,
        "channels": C,
        "global_mean": global_mean.item(),
        "global_std": global_std.item(),
        "scale_factor": scale_factor.item(),
        "raw_cos_sim": raw_cos_sim,
        "diff_cos_sim": diff_cos_sim,
        "mean_l2_norm": mean_l2,
        "effective_rank": effective_rank,
        "avg_pairwise_cos_sim": avg_pair_cos_sim,
        # --- dimension-normalized, cross-comparable with the 4ch runs ---------
        "effective_rank_ratio": effective_rank / C,
        "mean_l2_norm_ratio": mean_l2 / math.sqrt(C),
        "note": "REFERENCE-ONLY 8ch VQ continuous embedding; "
                "avg_pairwise_cos_sim dim-confounded; use *_ratio for xcmp",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_geometry)), exist_ok=True)
    pd.DataFrame([geom_row]).to_csv(args.out_geometry, index=False)
    print(f"[geometry] wrote {args.out_geometry}", flush=True)

    # --- stat CSV (results_ukb_stats.csv schema) -----------------------------
    stat_row = {
        "run_name": args.run_name,
        "scaling_factor": scale_factor.item(),
        "global_mean": global_mean.item(),
        "global_std": global_std.item(),
        "avg_single_sigma": float("nan"),  # VQ has no posterior sigma
        "num_train_samples": n_files,
        "note": "REFERENCE-ONLY; avg_single_sigma N/A (no VQ posterior)",
    }
    pd.DataFrame([stat_row]).to_csv(args.out_stat, index=False)
    print(f"[stat] wrote {args.out_stat}", flush=True)


if __name__ == "__main__":
    main()
