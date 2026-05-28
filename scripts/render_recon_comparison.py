"""Render a side-by-side recon comparison from saved [0,1] .nii.gz volumes.

All volumes must be the SAME subject index, processed at the SAME normalization
(lower=0.0) so the comparison is fair: identical slice positions, colormap, and
intensity window [0,1] across every panel. Rows = {real, MAISI, SID, 3D MedDiff},
cols = {axial, coronal, sagittal} mid-slices.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load(path):
    return nib.load(path).get_fdata().astype(np.float32)


def mid_slices(vol):
    d, h, w = vol.shape
    # axial (along axis0), coronal (axis1), sagittal (axis2) mid-planes
    return [np.rot90(vol[d // 2, :, :]),
            np.rot90(vol[:, h // 2, :]),
            np.rot90(vol[:, :, w // 2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=10)
    ap.add_argument("--out", default="journal_plan/recon_compare.png")
    ap.add_argument("--base", required=True)
    ap.add_argument("--maisi", default=None)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--meddiff", required=True)
    args = ap.parse_args()
    i = args.index

    rows = [("Real (base)", args.base)]
    if args.maisi:
        rows.append(("MAISI recon", args.maisi))
    rows.append(("L_SID (DFT) recon", args.sid))
    rows.append(("3D MedDiff recon", args.meddiff))

    plane = ["Axial", "Coronal", "Sagittal"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (label, path) in enumerate(rows):
        vol = load(path)
        sl = mid_slices(vol)
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(sl[c], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(plane[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle(f"UKB recon comparison — index {i:04d} (all lower=0.0, window [0,1])",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[render] wrote {args.out}")


if __name__ == "__main__":
    main()
