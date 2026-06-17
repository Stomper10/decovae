"""Per-cell reconstruction-vs-original montage for meeting figures.

Backend-agnostic: consumes the `base_*.nii.gz` (original) + `recon_*.nii.gz`
(reconstruction) volume pairs that BOTH eval paths already write in
compute_metric's layout —
  MAISI-family : compute_metric.py --eval_mode real_vs_recon --save_volume
  3D-MedDiff   : scripts/eval_3d_meddiff_recon.py
— and lays out, per (cohort×modality) cell, the wandb-style XYZ triptych
(axial | coronal | sagittal centre slices) of the original over its
reconstruction, gridded over all cells. One PNG per model.

    python -m scripts.recon_montage \
        --root  <exp_dir>           # holds cells/<cell>/outputs/volumes/{base,recon}_<idx>.nii.gz
        --model_label MAISI-kl5e3 \
        --out_png figs/recon_maisi_kl5e3.png
    # (run as -m from repo root; --flat if base_/recon_ sit directly under --root)
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from scripts.utils_plot import get_xyz_plot


def _triptych(vol_np: np.ndarray) -> np.ndarray:
    """(H,W,D) volume → wandb-style [axial|coronal|sagittal] RGB triptych, min-max [0,1]."""
    t = torch.from_numpy(vol_np[None].astype(np.float32))          # (1,H,W,D)
    center = [vol_np.shape[a] // 2 for a in range(3)]
    vis = get_xyz_plot(t, center, mask_bool=False)                 # [max, 3*max, 3]
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    return vis


def _load_vol(path: str) -> np.ndarray:
    return np.squeeze(np.asanyarray(nib.load(path).get_fdata())).astype(np.float32)


def _cell_dirs(root: str, flat: bool):
    """Yield (cell_name, volumes_dir). flat=base_/recon_ directly under root."""
    if flat:
        yield (os.path.basename(root.rstrip("/")), root)
        return
    for d in sorted(glob.glob(os.path.join(root, "cells", "*"))):
        if os.path.isdir(d):
            yield (os.path.basename(d), os.path.join(d, "outputs", "volumes"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="exp dir with cells/<cell>/outputs/volumes/ (or a volumes dir with --flat).")
    p.add_argument("--flat", action="store_true",
                   help="base_/recon_ sit directly under --root (single cell).")
    p.add_argument("--index", type=int, default=0, help="which base_/recon_ index per cell (default 0).")
    p.add_argument("--model_label", default="VAE")
    p.add_argument("--ncols", type=int, default=2, help="how many cells per figure row.")
    p.add_argument("--out_png", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for cell, vdir in _cell_dirs(args.root, args.flat):
        base = os.path.join(vdir, f"base_{args.index:04d}.nii.gz")
        recon = os.path.join(vdir, f"recon_{args.index:04d}.nii.gz")
        if not (os.path.exists(base) and os.path.exists(recon)):
            print(f"[skip] {cell}: missing base/recon (idx {args.index}) under {vdir}")
            continue
        orig_t = _triptych(_load_vol(base))
        recon_t = _triptych(_load_vol(recon))
        sep = np.ones((max(2, orig_t.shape[0] // 32), orig_t.shape[1], 3), dtype=np.float32)
        rows.append((cell, np.vstack([orig_t, sep, recon_t])))   # orig (top) / recon (bottom)
    if not rows:
        raise SystemExit(f"[recon_montage] no base/recon pairs found under {args.root}")

    n = len(rows)
    ncols = min(args.ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (cell, img) in enumerate(rows):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(img)
        ax.set_title(cell, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{args.model_label} — top: ORIGINAL, bottom: RECON  "
                 f"(axial | coronal | sagittal centre slices)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"[recon_montage] {args.model_label}: {n} cells -> {args.out_png}")


if __name__ == "__main__":
    main()
