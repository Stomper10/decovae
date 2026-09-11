#!/usr/bin/env python3
"""§3a — spatial control for the BraTS tumour segmentor.

The segmentor was trained on raw BraTS-space FLAIR (240×240×144, no pooled
preprocessing). §3 wants to score it on pooled `.npy` volumes (192³, RAS +
1mm + center-crop, percentile-normalised). This script measures how much
dice degrades between the two spaces on the SAME cases.

Two evals per case:
  (A) NATIVE: raw BraTS FLAIR + raw seg → segmentor → dice
      (matches the training/validation pipeline, i.e. seg_dataset.build_transforms
      with resolution=(240,240,144) and Resized).
  (B) POOLED: raw BraTS FLAIR + raw seg → reorient RAS → resample 1mm →
      center-crop 192³ → run through segmentor (resize input to 240×240×144
      for the model, since roi_size=(128,128,128) needs enough spatial extent
      and the model was trained at that footprint).

Both compute dice against the SAME preprocessed mask so the two numbers are
directly comparable. In (A) the preprocess is just resize-to-(240,240,144);
in (B) the preprocess is RAS + 1mm + 192³ center-crop first.

Usage:
    python3 scripts/eval_seg_domain_shift.py \
        --brats_root /leelabsg/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train \
        --seg_ckpt /leelabsg/data/wonyoungjang/decodata/brats/downstream/tumor_seg/real_only_FLAIR/weights/best.pt \
        --n_cases 25 --out journal_plan/results_seg_domain_shift.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd,
    Orientationd, Resized, Spacingd, ResizeWithPadOrCropd, EnsureTyped,
)

from downstream.seg_dataset import ConvertBratsGLI2023Labelsd

REGION_NAMES = ("TC", "WT", "ET")
NATIVE_RES = (240, 240, 144)
POOLED_RES = (192, 192, 192)
ROI = (128, 128, 128)


def build_segresnet(in_channels: int = 1, num_classes: int = 3) -> SegResNet:
    # MUST match downstream/train_tumor_seg.py::build_segresnet exactly, or
    # load_state_dict raises on missing norm/deconv keys.
    return SegResNet(
        spatial_dims=3, init_filters=32,
        in_channels=in_channels, out_channels=num_classes,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
        norm="instance", dropout_prob=0.2,
    )


def native_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=NATIVE_RES, size_mode="all", mode="trilinear"),
        Resized(keys=["mask"], spatial_size=NATIVE_RES, size_mode="all", mode="nearest"),
        ConvertBratsGLI2023Labelsd(keys=["mask"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "mask"], dtype="float32"),
    ])


def pooled_transforms() -> Compose:
    # BraTS is already skull-stripped + SRI24-registered so we skip strip/rigid,
    # matching dryrun_preproc.py's BraTS branch: RAS → 1mm → 192³ center-crop.
    # After crop we resize to NATIVE_RES so the model sees the footprint it was
    # trained at — this measures ONLY the crop/normalisation domain shift, not
    # a shape mismatch. Intensity normalisation stays MONAI's percentile-free
    # NormalizeIntensityd (channel-wise, nonzero-only), same as native. If we
    # wanted to add the raw pooled percentile step it would go here.
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Spacingd(keys=["mask"], pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=POOLED_RES),
        # Now upsample to the model's expected footprint for a fair comparison
        # (only the crop is the domain shift; shape is normalised out).
        Resized(keys=["image"], spatial_size=NATIVE_RES, size_mode="all", mode="trilinear"),
        Resized(keys=["mask"], spatial_size=NATIVE_RES, size_mode="all", mode="nearest"),
        ConvertBratsGLI2023Labelsd(keys=["mask"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "mask"], dtype="float32"),
    ])


@torch.no_grad()
def score_one(sample: dict, tf: Compose, model: torch.nn.Module,
              device: torch.device) -> tuple[float, float, float]:
    """Return per-region dice (TC, WT, ET) for one case."""
    d = tf(sample)
    img = d["image"].unsqueeze(0).to(device)          # (1,1,240,240,144)
    mask = d["mask"].unsqueeze(0).to(device)          # (1,3,240,240,144)
    logits = sliding_window_inference(img, ROI, sw_batch_size=2, predictor=model, overlap=0.5)
    pred = (torch.sigmoid(logits) > 0.5).float()
    metric = DiceMetric(include_background=True, reduction="none")
    metric(y_pred=pred, y=mask)
    dice = metric.aggregate().cpu().numpy().squeeze()  # shape (3,)
    return float(dice[0]), float(dice[1]), float(dice[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats_root", required=True)
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--n_cases", type=int, default=25)
    ap.add_argument("--out", default="journal_plan/results_seg_domain_shift.csv")
    args = ap.parse_args()

    # 1. Enumerate cases (BraTS-GLI-XXXXX-XXX with -t2f.nii.gz and -seg.nii.gz)
    root = Path(args.brats_root)
    cases = sorted([p for p in root.iterdir() if p.is_dir()])[:args.n_cases]
    print(f"[eval] scoring {len(cases)} cases from {root}")

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.seg_ckpt, map_location=device, weights_only=False)
    model = build_segresnet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded seg ckpt (epoch {ckpt.get('epoch')} val_dice_mean {ckpt.get('val_dice_mean'):.4f})")

    tf_nat = native_transforms()
    tf_pool = pooled_transforms()

    # 3. Score each case in both spaces
    rows = []
    for i, case in enumerate(cases):
        sid = case.name
        img = case / f"{sid}-t2f.nii.gz"
        mask = case / f"{sid}-seg.nii.gz"
        if not img.exists() or not mask.exists():
            print(f"  [skip] {sid}: missing files")
            continue
        sample = {"image": str(img), "mask": str(mask)}
        try:
            tc_n, wt_n, et_n = score_one(sample, tf_nat, model, device)
            tc_p, wt_p, et_p = score_one(sample, tf_pool, model, device)
        except Exception as e:
            print(f"  [err] {sid}: {e}")
            continue
        rows.append({
            "case": sid,
            "native_TC": tc_n, "native_WT": wt_n, "native_ET": et_n,
            "pooled_TC": tc_p, "pooled_WT": wt_p, "pooled_ET": et_p,
        })
        print(f"  [{i+1:3d}/{len(cases)}] {sid}  "
              f"native WT/TC/ET {wt_n:.3f}/{tc_n:.3f}/{et_n:.3f}  "
              f"pooled {wt_p:.3f}/{tc_p:.3f}/{et_p:.3f}")

    # 4. Write CSV + summary
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] wrote {out_path} ({len(rows)} rows)")

    # summary
    import statistics as st
    for reg in REGION_NAMES:
        n = [r[f"native_{reg}"] for r in rows]
        p = [r[f"pooled_{reg}"] for r in rows]
        print(f"  {reg}: native mean {st.mean(n):.3f} ± {st.stdev(n):.3f}   "
              f"pooled mean {st.mean(p):.3f} ± {st.stdev(p):.3f}   "
              f"drop {st.mean(n) - st.mean(p):+.3f}")


if __name__ == "__main__":
    main()
