"""Per-cell recon montage: one PNG per cell, rows = [original, model_1, model_2, ...].

Designed to compare multiple VAE backbones on the SAME source volume at the
SAME slice indices. Inputs are the existing `base_<idx>.nii.gz` (original) and
`recon_<idx>.nii.gz` (recon) volume pairs that compute_metric and the 3D-MedDiff
recon eval already write. The same base_<idx> across models must be byte-identical
for the comparison to be meaningful (verify with --check_bases).

    python -m scripts.recon_montage_per_cell \
        --cell adni_T1 \
        --index 0 \
        --model "MAISI-kl5e3:/.../pooled-maisi-kl5e3-s1-test/cells/adni_T1/outputs/volumes" \
        --model "3DMD:/.../recon_eval_3dmd/3dmd_test_adni_T1_latest_checkpoint_/outputs/volumes" \
        --out_png figs/recon_adni_T1.png

The base_<idx>.nii.gz is taken from the FIRST --model entry (all model volumes
dirs share the same base after --check_bases verification).
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from scripts.utils_plot import get_xyz_plot


def _triptych(vol_np: np.ndarray) -> np.ndarray:
    """(H,W,D) → wandb-style [axial|coronal|sagittal] RGB triptych, min-max [0,1]."""
    t = torch.from_numpy(vol_np[None].astype(np.float32))
    center = [vol_np.shape[a] // 2 for a in range(3)]
    vis = get_xyz_plot(t, center, mask_bool=False)
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    return vis


def _load(path: str) -> np.ndarray:
    return np.squeeze(np.asanyarray(nib.load(path).get_fdata())).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, help="cell name (used in figure title only).")
    p.add_argument("--index", type=int, default=0, help="which base_/recon_ index.")
    p.add_argument("--model", action="append", required=True,
                   help="`label:volumes_dir` — repeat per model. First entry's base_ is the original.")
    p.add_argument("--check_bases", action="store_true",
                   help="verify base_<idx>.nii.gz is byte-identical across all model dirs.")
    p.add_argument("--src_csv", default=None,
                   help="base CSV (same one fed to compute_metric); row at --index gives subject_id "
                        "for the Original label. compute_metric assigns base_<i> to row <i> in CSV "
                        "order (DDP shuffle=False).")
    p.add_argument("--out_png", required=True)
    p.add_argument("--label_width_px", type=int, default=120,
                   help="left-margin width for row labels (pixels).")
    return p.parse_args()


def _subject_for_idx(src_csv: str, idx: int) -> str | None:
    import csv as _csv
    try:
        with open(src_csv, newline="") as f:
            for i, row in enumerate(_csv.DictReader(f)):
                if i == idx:
                    return row.get("subject_id") or row.get("cache_key") or None
    except Exception:
        return None
    return None


def main() -> None:
    args = parse_args()
    models = []
    for spec in args.model:
        if ":" not in spec:
            raise SystemExit(f"--model expects 'label:path', got {spec!r}")
        lab, path = spec.split(":", 1)
        models.append((lab, path))

    bases = []
    for lab, vdir in models:
        bp = os.path.join(vdir, f"base_{args.index:04d}.nii.gz")
        if not os.path.exists(bp):
            raise SystemExit(f"missing base for {lab}: {bp}")
        bases.append(_load(bp))
    if args.check_bases:
        ref = bases[0]
        for (lab, _), b in zip(models[1:], bases[1:]):
            if b.shape != ref.shape or not np.allclose(b, ref, atol=1e-4):
                raise SystemExit(f"[check_bases] {lab} base differs from first model — aborting.")
        print(f"[check_bases] OK: all {len(bases)} models share the same base_{args.index:04d}.nii.gz")

    orig_label = f"Original\nidx={args.index:04d}"
    if args.src_csv:
        subj = _subject_for_idx(args.src_csv, args.index)
        if subj:
            orig_label = f"Original\nidx={args.index:04d}\n{subj}"
    rows = [(orig_label, _triptych(bases[0]))]
    for lab, vdir in models:
        rp = os.path.join(vdir, f"recon_{args.index:04d}.nii.gz")
        if not os.path.exists(rp):
            print(f"[skip] {lab}: missing recon ({rp})")
            continue
        rows.append((lab, _triptych(_load(rp))))

    img_stack = np.vstack([img for _, img in rows])
    H_row, W_img, _ = rows[0][1].shape
    H_total = img_stack.shape[0]
    LW = max(0, int(args.label_width_px))

    if LW > 0:
        label_col = np.ones((H_total, LW, 3), dtype=np.float32)
        canvas = np.concatenate([label_col, img_stack], axis=1)
    else:
        canvas = img_stack

    H_canvas, W_canvas, _ = canvas.shape
    dpi = 130.0
    fig = plt.figure(figsize=(W_canvas / dpi, H_canvas / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(canvas, interpolation="nearest")
    if LW > 0:
        for i, (lab, _) in enumerate(rows):
            y_center = i * H_row + H_row / 2
            ax.text(LW * 0.5, y_center, lab, ha="center", va="center",
                    fontsize=10, color="black")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    fig.savefig(args.out_png, dpi=dpi, pad_inches=0)
    plt.close(fig)
    print(f"[recon_montage_per_cell] {args.cell}: {len(rows)} rows -> {args.out_png}")


if __name__ == "__main__":
    main()
