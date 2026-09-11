#!/usr/bin/env python3
"""§3b — tumor "mode-averaging" test.

Runs the BraTS-GLI-2023 SegResNet segmentor on either real BraTS FLAIR volumes
(preprocessed the same way pooled gen volumes are) or on gen brats_FLAIR volumes,
and emits per-volume shape/count metrics. The hypothesis in the round-2 handoff
is that the gen tumor mass ends up smeared into many small components spread
across the brain, so we compare:

  detected            has any tumor voxels above `min_voxels` in a region
  vol_<region>        total positive voxels in region (raw count)
  cc_count_<region>   number of connected components (with min-size filter)
  disp_<region>       RMS voxel distance of positive voxels from their centroid
  largest_cc_frac     largest CC's fraction of the total positive volume

Regions: TC, WT, ET (index 0/1/2 out of the segmentor's 3 sigmoid channels —
matches downstream/train_tumor_seg.py::build_segresnet outputs).

Volume source (--vol_src):
  real          : BraTS-GLI raw .nii.gz — expects <brats_root>/<sid>/<sid>-t2f.nii.gz.
                  Preprocessed via RAS + 1mm + 192³ center-crop + resize-to-240,
                  same as scripts/eval_seg_domain_shift.py::pooled_transforms.
  gen           : list every .nii.gz under --gen_dir; already pooled 192³, just
                  resize-to-240 and normalise.

Output: one CSV row per volume, columns fixed. Aggregation happens in the
harvester or a plotting notebook — this script just makes the per-volume table.

Measurement conventions (must match when comparing across runs):
- Segmentor ckpt: `real_only_FLAIR/weights/best.pt` (MONAI SegResNet, ctor
  spatial_dims=3, init_filters=32, blocks_down=(1,2,2,4), blocks_up=(1,1,1),
  norm="instance", dropout_prob=0.2 — matches downstream/train_tumor_seg.py).
- Model footprint 240×240×144, ROI 128³, sliding-window overlap 0.5, batch 2.
- Binarisation: `torch.sigmoid(logits) > 0.5` per region channel (TC=0, WT=1,
  ET=2 — 3 sigmoid channels).
- Real preprocess (`real_transforms`): LoadImaged → EnsureChannelFirstd →
  Orientationd(RAS) → Spacingd(1mm iso) → ResizeWithPadOrCropd(192³) →
  Resized(240×240×144, trilinear) → NormalizeIntensityd(nonzero, channel_wise).
- Gen preprocess (`gen_transforms`): already pooled 192³ [0,1] percentile-
  normalised; skip Spacingd + PadCrop, keep the Resized + NormalizeIntensityd.
- Connected components: `scipy.ndimage.label` default structure = 6-connectivity
  (orthogonal neighbours only; not 18 or 26).
- `disp` = RMS voxel distance of positive voxels from their centroid
  (`sqrt(mean(||coord - centroid||^2))`, coord in voxel index units).
- `largest_cc_frac` = size of the largest CC / total positive voxels.
- Filters: `min_voxels` (region detected? default 50), `min_cc_voxels` (drop
  from cc_count if smaller than this, default 20).
- Real volume space: raw BraTS NIfTI on disk → pooled preprocess above. That is
  how the 500 real-brats rows in `journal_plan/tumor_mode/real.csv` were made
  (i.e. NOT native BraTS 240×240×144). It matches the gen preprocess so the
  two distributions are comparable inside the same reference frame.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd, ResizeWithPadOrCropd, Resized, Spacingd,
)
from scipy.ndimage import label as cc_label

REGION_NAMES = ("TC", "WT", "ET")
MODEL_RES = (240, 240, 144)   # segmentor's native training footprint
POOLED_RES = (192, 192, 192)
ROI = (128, 128, 128)


def build_segresnet(in_channels: int = 1, num_classes: int = 3) -> SegResNet:
    return SegResNet(
        spatial_dims=3, init_filters=32,
        in_channels=in_channels, out_channels=num_classes,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
        norm="instance", dropout_prob=0.2,
    )


def real_transforms() -> Compose:
    """Real BraTS → pooled preprocess → resize to model footprint."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=POOLED_RES),
        Resized(keys=["image"], spatial_size=MODEL_RES, size_mode="all", mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype="float32"),
    ])


def gen_transforms() -> Compose:
    """Gen volumes are already pooled 192³ percentile-normalised [0,1].
    Just resize to the model's footprint and apply the same normalise the seg
    training used, so intensity statistics match the real-branch input above.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=MODEL_RES, size_mode="all", mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype="float32"),
    ])


def region_metrics(mask: np.ndarray, min_voxels: int, min_cc_voxels: int) -> dict:
    """mask: bool (D,H,W) for one region. Returns metrics dict."""
    vol = int(mask.sum())
    detected = vol >= min_voxels
    out = {"vol": vol, "detected": int(detected), "cc_count": 0,
           "disp": float("nan"), "largest_cc_frac": float("nan")}
    if not detected:
        return out
    lab, n = cc_label(mask)
    if n == 0:
        return out
    sizes = np.bincount(lab.ravel())[1:]     # drop background
    keep = sizes >= min_cc_voxels
    out["cc_count"] = int(keep.sum())
    if keep.any():
        out["largest_cc_frac"] = float(sizes[keep].max() / max(vol, 1))
    # dispersion: RMS voxel distance of ALL positive voxels from their centroid
    coords = np.argwhere(mask)            # (N, 3)
    if coords.shape[0] > 1:
        c = coords.mean(axis=0)
        out["disp"] = float(np.sqrt(((coords - c) ** 2).sum(axis=1).mean()))
    return out


@torch.no_grad()
def score_one(sample: dict, tf: Compose, model: torch.nn.Module,
              device: torch.device, min_voxels: int, min_cc_voxels: int) -> dict:
    d = tf(sample)
    img = d["image"].unsqueeze(0).to(device)              # (1,1,240,240,144)
    logits = sliding_window_inference(img, ROI, sw_batch_size=2, predictor=model, overlap=0.5)
    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0]  # (3, D, H, W) bool
    row = {}
    for i, name in enumerate(REGION_NAMES):
        m = region_metrics(pred[i].astype(bool), min_voxels, min_cc_voxels)
        for k, v in m.items():
            row[f"{k}_{name}"] = v
    row["any_detected"] = int(any(row[f"detected_{n}"] for n in REGION_NAMES))
    return row


def iter_cases(args) -> list[tuple[str, dict]]:
    """Return list of (case_id, sample_dict) to score."""
    if args.vol_src == "real":
        root = Path(args.brats_root)
        cases = sorted(p for p in root.iterdir() if p.is_dir())
        if args.max_cases > 0:
            cases = cases[: args.max_cases]
        out = []
        for c in cases:
            sid = c.name
            img = c / f"{sid}-t2f.nii.gz"
            if img.exists():
                out.append((sid, {"image": str(img)}))
        return out
    else:
        gen = Path(args.gen_dir)
        files = sorted(gen.glob("*.nii.gz"))
        if args.max_cases > 0:
            files = files[: args.max_cases]
        return [(f.name.replace(".nii.gz", ""), {"image": str(f)}) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_src", choices=("real", "gen"), required=True)
    ap.add_argument("--brats_root", default="/leelabsg/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train")
    ap.add_argument("--gen_dir", help="For --vol_src gen: <cell>/outputs/volumes")
    ap.add_argument("--seg_ckpt", required=True)
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--max_cases", type=int, default=0, help="0 = all")
    ap.add_argument("--min_voxels", type=int, default=50,
                    help="region 'detected' threshold; keep small to catch scatter")
    ap.add_argument("--min_cc_voxels", type=int, default=20,
                    help="drop CCs smaller than this from cc_count")
    args = ap.parse_args()

    if args.vol_src == "gen" and not args.gen_dir:
        ap.error("--gen_dir required for --vol_src gen")

    cases = iter_cases(args)
    print(f"[3b] {args.vol_src}: scoring {len(cases)} volumes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.seg_ckpt, map_location=device, weights_only=False)
    model = build_segresnet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[3b] loaded seg ckpt (epoch {ckpt.get('epoch')})")

    tf = real_transforms() if args.vol_src == "real" else gen_transforms()

    rows: list[dict] = []
    fields = ["case", "src"] + [f"{k}_{r}" for r in REGION_NAMES
                                for k in ("detected", "vol", "cc_count", "disp", "largest_cc_frac")] + ["any_detected"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (case, sample) in enumerate(cases):
        try:
            m = score_one(sample, tf, model, device, args.min_voxels, args.min_cc_voxels)
        except Exception as e:
            print(f"  [err] {case}: {e}")
            continue
        row = {"case": case, "src": args.vol_src, **m}
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(cases):
            print(f"  [{i+1:4d}/{len(cases)}] {case}  any={row['any_detected']}  "
                  f"vol WT/TC/ET {row['vol_WT']}/{row['vol_TC']}/{row['vol_ET']}  "
                  f"cc {row['cc_count_WT']}/{row['cc_count_TC']}/{row['cc_count_ET']}")
            # incremental write so early results survive a walltime hit
            with out_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

    print(f"[3b] wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
