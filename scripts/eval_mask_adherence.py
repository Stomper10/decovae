#!/usr/bin/env python3
"""Tumour mask adherence of ControlNet generations, with a real-data ceiling.

Runs the BraTS FLAIR segmentor on volumes in the POOLED preprocessing frame and scores
its prediction against the pooled-space mask of the same (subject, modality) from
scripts/build_controlnet_masks.py. Two sources:

  --src real   cached pooled FLAIR .npy of subjects the segmentor never trained on
               (valid + test splits). This is the CONTROL: how well the segmentor
               recovers a real tumour in the frame the generations live in. Its Dice
               is the ceiling for adherence -- a generator cannot be expected to beat
               the instrument.
  --src gen    gen_*.nii.gz from a MODE=controlnet cell. The mask is the one named in
               each volume's .cond.json (mask_cache_key), i.e. the mask it was
               generated FROM, so the score is spatial adherence to the condition.

WHY A NEW CONTROL. scripts/eval_seg_domain_shift.py measured "pooled" as RAS + 1mm +
192^3 center crop. The cache is not that: it is N4 + rigid registration to MNI152 +
crop/pad + percentile normalisation. Its native 0.928 -> "pooled" 0.914 WT therefore
does not cover the frame the generations are in. The masks built by
build_controlnet_masks.py are exact in that frame, so this measures it directly.

DEFINITIONS ARE IMPORTED, NOT REWRITTEN. Segmentor architecture, model footprint,
sliding-window ROI, sigmoid > 0.5, connected components and dispersion all come from
scripts/eval_tumor_mode_avg.py, so morphology columns here mean exactly what they mean
in journal_plan/tumor_mode/*.csv. Input preprocessing matches its gen branch: resize
192^3 -> 240x240x144, NormalizeIntensityd(nonzero). Regions follow
ConvertBratsGLI2023Labelsd: TC = {1,3}, WT = {1,2,3}, ET = {3}.

    python -m scripts.eval_mask_adherence --src real --out real_pooled.csv
    python -m scripts.eval_mask_adherence --src gen --gen_dir <cell>/outputs/volumes --out cn_vad.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd, Orientationd, Resized)

from scripts.eval_tumor_mode_avg import MODEL_RES, REGION_NAMES, ROI, build_segresnet, region_metrics

CSV_DIR = "/data/wonyoungjang/decovae/csv_files"


def seg_transforms(npy: bool) -> Compose:
    if npy:
        load = [LoadImaged(keys=["image"], reader="NumpyReader"),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")]
    else:  # generated volumes are saved with an identity affine, same array layout as the cache
        load = [LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS")]
    return Compose(load + [
        Resized(keys=["image"], spatial_size=MODEL_RES, size_mode="all", mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype="float32"),
    ])


def mask_regions(mask192: np.ndarray) -> np.ndarray:
    """(192^3 labels) -> (3, *MODEL_RES) bool, TC/WT/ET, nearest-resized to the model footprint."""
    m = torch.from_numpy(mask192.astype(np.int64))
    reg = torch.stack([(m == 1) | (m == 3), m > 0, m == 3]).float()[None]
    return (F.interpolate(reg, size=MODEL_RES, mode="nearest")[0] > 0.5).numpy()


def dice(p: np.ndarray, g: np.ndarray) -> float:
    s = p.sum() + g.sum()
    return float("nan") if s == 0 else float(2 * (p & g).sum() / s)


def cases(args):
    if args.src == "real":
        rows = []
        for sp in args.splits:
            d = pd.read_csv(os.path.join(CSV_DIR, f"pooled_manifest_{sp}.csv"))
            rows += d[(d.cohort == "brats") & (d.modality == args.modality)]["cache_key"].tolist()
        out = []
        for key in sorted(rows):
            m = os.path.join(args.mask_dir, f"{key}_seg.npy")
            if os.path.isfile(m):
                out.append((os.path.basename(key), os.path.join(args.cache_root, f"{key}.npy"), m, True))
        return out
    out = []
    for vol in sorted(glob.glob(os.path.join(args.gen_dir, "gen_*.nii.gz"))):   # gen only, never base_*
        side = vol[: -len(".nii.gz")] + ".cond.json"
        key = json.load(open(side)).get("mask_cache_key") if os.path.isfile(side) else None
        if key is None:
            raise SystemExit(f"[FATAL] {side} has no mask_cache_key: not a MODE=controlnet volume")
        out.append((os.path.basename(vol)[: -len(".nii.gz")], vol,
                    os.path.join(args.mask_dir, f"{key}_seg.npy"), False))
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=("real", "gen"), required=True)
    ap.add_argument("--gen_dir")
    ap.add_argument("--modality", default="FLAIR", help="segmentor is FLAIR-only")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"],
                    help="real: subjects the segmentor did not train on")
    ap.add_argument("--cache_root", default="/data/wonyoungjang/decovae_cache")
    ap.add_argument("--mask_dir", default="/data/wonyoungjang/decodata/pooled/controlnet_masks")
    ap.add_argument("--seg_ckpt", default="/data/wonyoungjang/decodata/brats/downstream/tumor_seg/real_only_FLAIR/weights/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--min_voxels", type=int, default=50)
    ap.add_argument("--min_cc_voxels", type=int, default=20)
    args = ap.parse_args()
    if args.src == "gen" and not args.gen_dir:
        ap.error("--gen_dir required for --src gen")

    todo = cases(args)
    if args.max_cases:
        todo = todo[: args.max_cases]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.seg_ckpt, map_location=device, weights_only=False)
    model = build_segresnet().to(device); model.load_state_dict(ck["model"]); model.eval()
    tf = {True: seg_transforms(True), False: seg_transforms(False)}
    print(f"[adherence] {args.src}: {len(todo)} volumes on {device} (seg ckpt epoch {ck.get('epoch')})", flush=True)

    fields = (["case", "src"] + [f"dice_{r}" for r in REGION_NAMES] + [f"mask_vol_{r}" for r in REGION_NAMES]
              + [f"{k}_{r}" for r in REGION_NAMES for k in ("detected", "vol", "cc_count", "disp", "largest_cc_frac")])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for i, (case, img, mfp, npy) in enumerate(todo):
        x = tf[npy]({"image": img})["image"].unsqueeze(0).to(device)
        pred = (torch.sigmoid(sliding_window_inference(x, ROI, sw_batch_size=2, predictor=model, overlap=0.5)) > 0.5)
        pred = pred.cpu().numpy()[0].astype(bool)
        gt = mask_regions(np.load(mfp))
        row = {"case": case, "src": args.src}
        for j, r in enumerate(REGION_NAMES):
            row[f"dice_{r}"] = dice(pred[j], gt[j])
            row[f"mask_vol_{r}"] = int(gt[j].sum())
            for k, v in region_metrics(pred[j], args.min_voxels, args.min_cc_voxels).items():
                row[f"{k}_{r}"] = v
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            print(f"  {i+1}/{len(todo)}  {case}  dice WT {row['dice_WT']:.3f} TC {row['dice_TC']:.3f} ET {row['dice_ET']:.3f}", flush=True)

    d = pd.DataFrame(rows)
    print("\nmedian dice  " + "  ".join(f"{r} {d[f'dice_{r}'].median():.3f}" for r in REGION_NAMES)
          + f"   | cc_count_WT median {d['cc_count_WT'].median():.0f}   disp_WT median {d['disp_WT'].median():.1f}")
    print(f"wrote {args.out} ({len(d)} rows)")


if __name__ == "__main__":
    main()
