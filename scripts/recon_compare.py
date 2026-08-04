#!/usr/bin/env python3
"""Cross-MODEL recon comparison grid (eyeball blur A/B) — no decode/GPU.

Complements scripts/recon_montage.py (which is per-model, gridded over cells).
Here each ROW is one subject and each COLUMN is a model, so you can directly
compare recon sharpness across configs at the SAME subject:

    [ real | pooled recon | specialist recon | +adniT1 recon | ... ]

Consumes the wandb-style 3-plane strips that compute_metric.py (real_vs_recon,
--save_volume default) already wrote on GSDS:
    <exp>/cells/<cell>[_test]/outputs/slices/{i:04d}_{base,recon}_xyz.png
Subject i is the same volume across models (deterministic per-cell filelist),
so columns are apples-to-apples. Reads PNGs only (matplotlib + numpy).

Usage (GSDS):
  ROOT=${POOLED_OUTPUT_ROOT}/stage1
  python scripts/recon_compare.py \
    --out recon_cmp_ukb_T1_ck240k.png --subjects 0 1 2 3 \
    --model "pooled:${ROOT}/eval-pooled-ck240000/cells/ukb_T1_test/outputs/slices" \
    --model "specialist:${ROOT}/eval-ukbT1-ck240000/cells/ukb_T1_test/outputs/slices" \
    --model "+adniT1:${ROOT}/eval-ukbT1adni-ck240000/cells/ukb_T1_test/outputs/slices" \
    --model "+FLAIR:${ROOT}/eval-ukbT1flair-ck240000/cells/ukb_T1_test/outputs/slices"
  # real column taken from the first model's *_base_xyz.png (identical across models).
  # add MIUA once a MIUA-base real_vs_recon eval has produced its slices dir.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def load_png(path):
    return mpimg.imread(path) if os.path.isfile(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--model", action="append", required=True,
                    help="LABEL:SLICE_DIR (repeatable).")
    ap.add_argument("--no_real", action="store_true")
    ap.add_argument("--scale", type=float, default=3.0)
    args = ap.parse_args()

    models = [(m.split(":", 1)[0], m.split(":", 1)[1]) for m in args.model]
    cols = ([] if args.no_real else [("real", None)]) + models
    real_dir = models[0][1]

    nrows, ncols = len(args.subjects), len(cols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(args.scale * ncols, args.scale * nrows),
                             squeeze=False)
    n_missing = 0
    for r, i in enumerate(args.subjects):
        for c, (label, sdir) in enumerate(cols):
            ax = axes[r][c]
            ax.axis("off")
            if sdir is None:
                img = load_png(os.path.join(real_dir, f"{i:04d}_base_xyz.png"))
            else:
                img = load_png(os.path.join(sdir, f"{i:04d}_recon_xyz.png"))
            if img is not None:
                ax.imshow(img, cmap="gray")
            else:
                n_missing += 1
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center", color="red", fontsize=8)
            if r == 0:
                ax.set_title(label, fontsize=12)
            if c == 0:
                ax.text(-0.05, 0.5, f"subj {i}", rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=10)

    fig.suptitle("recon comparison — rows: subjects, cols: models "
                 "(each cell = axial | coronal | sagittal centre slices)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"[recon_compare] wrote {args.out}  ({nrows} subj x {ncols} cols, {n_missing} missing)")


if __name__ == "__main__":
    main()
