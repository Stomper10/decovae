"""Conditioning-adherence evaluation.

Measures whether the diffusion model's CONDITIONED generations actually carry the
attribute they were conditioned on, by running an independent predictor (trained
on REAL data) over the generated volumes and comparing its prediction to the
intended condition recorded in each volume's ``.cond.json`` sidecar
(written by ``compute_metric.py`` --eval_mode real_vs_gen).

One attribute per invocation (run 3× for age / sex / dx):

    # age (regression): SFCN regressor from train_brain_age
    python scripts/adherence_eval.py --target age --task reg \
        --predictor_ckpt .../brain_age/weights/best.pt \
        --gen_dir .../outputs/volumes --postfix 30step_g20 ...

    # sex / dx (classification): SFCN classifier from train_attr_predictor
    python scripts/adherence_eval.py --target dx --task cls \
        --predictor_ckpt .../dx_clf/weights/best.pt ...

Metrics: reg → MAE / RMSE / R² / Pearson r (pred age vs intended age);
cls → accuracy / balanced accuracy / confusion (pred class vs intended class).
"""
from __future__ import annotations

import argparse
import json
import glob
import os
from pathlib import Path

import numpy as np
import torch
from monai.data import Dataset, DataLoader

from downstream.brain_age_dataset import build_transforms
from downstream.sfcn import build_sfcn, build_sfcn_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True, help="outputs/volumes dir holding gen_*.nii.gz + .cond.json.")
    p.add_argument("--postfix", required=True, help="generation postfix, e.g. 30step_all_T1_g20.")
    p.add_argument("--target", required=True, help="cond.json key to score (age|sex|dx|cdrsb).")
    p.add_argument("--task", required=True, choices=["reg", "cls"])
    p.add_argument("--predictor_ckpt", required=True, help="best.pt from train_brain_age / train_attr_predictor.")
    p.add_argument("--dataset_config_path", required=True)
    p.add_argument("--label_map", default=None,
                   help="cls only; JSON string. Defaults to the map stored in the ckpt.")
    p.add_argument("--resolution", type=int, nargs=3, default=None,
                   help="override; defaults to ckpt['resolution'] then dataset.json.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_csv", required=True, help="per-volume predictions CSV.")
    p.add_argument("--output_json", required=True, help="aggregate adherence metrics JSON.")
    return p.parse_args()


def _gather_samples(gen_dir: str, postfix: str, target: str):
    """List (image_path, intended_value) for gen volumes whose sidecar carries target."""
    vols = sorted(glob.glob(os.path.join(gen_dir, f"gen_*_{postfix}.nii.gz")))
    items, n_missing = [], 0
    for v in vols:
        side = v.replace(".nii.gz", ".cond.json")
        if not os.path.exists(side):
            n_missing += 1
            continue
        with open(side) as f:
            cond = json.load(f)
        val = cond.get(target)
        if val is None:            # volume not conditioned on this attribute → skip
            continue
        items.append({"image": v, "intended": val})
    if n_missing:
        print(f"[adherence] {n_missing} volumes had no .cond.json sidecar (skipped).")
    return items, len(vols)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.predictor_ckpt, map_location="cpu", weights_only=False)
    with open(args.dataset_config_path) as f:
        ds_cfg = json.load(f)
    resolution = tuple(args.resolution or ckpt.get("resolution") or
                       ds_cfg.get("resolution", [160, 192, 160]))
    axcodes = ds_cfg.get("orientation_axcodes", "RAS")

    if args.task == "cls":
        label_map = json.loads(args.label_map) if args.label_map else ckpt.get("label_map")
        if label_map is None:
            raise ValueError("cls task needs a label_map (arg or in ckpt).")
        num_classes = ckpt.get("num_classes", len(set(label_map.values())))
        inv_map = {int(v): k for k, v in label_map.items()}
        model = build_sfcn_classifier(num_classes=num_classes, in_channels=1, dropout=0.0)
    else:
        label_map, num_classes = None, None
        model = build_sfcn(in_channels=1, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    items, n_total = _gather_samples(args.gen_dir, args.postfix, args.target)
    if not items:
        raise SystemExit(f"[adherence] no gen volumes carry target {args.target!r} "
                         f"(scanned {n_total} under {args.gen_dir}).")
    print(f"[adherence] target={args.target} task={args.task} | "
          f"{len(items)}/{n_total} volumes carry the condition | resolution={resolution}")

    # generations are .nii.gz; auto-detect in case a .npy set is ever evaluated
    npy = str(items[0]["image"]).endswith(".npy")
    loader = DataLoader(
        Dataset(data=items, transform=build_transforms(resolution, axcodes, npy=npy)),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    intended_all, pred_all = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            out = model(x)
            if args.task == "cls":
                pred_all.append(out.argmax(dim=1).cpu().numpy())
            else:
                pred_all.append(out.cpu().numpy().reshape(-1))
            intended_all.append(np.asarray(batch["intended"]))
    preds = np.concatenate(pred_all)
    # intended: cls → map category string to class id; reg → float
    if args.task == "cls":
        intended = np.array([label_map.get(str(v), -1) for v in np.concatenate(intended_all)])
    else:
        intended = np.concatenate(intended_all).astype(float)

    keep = intended >= 0 if args.task == "cls" else np.ones_like(intended, dtype=bool)
    preds, intended = preds[keep], intended[keep]

    # ---- per-volume CSV ----
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(args.output_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["image", "intended", "predicted"])
        for it, ipred, iint in zip([i["image"] for i in items], preds, intended):
            w.writerow([os.path.basename(it), iint, ipred])

    # ---- aggregate metrics ----
    if args.task == "reg":
        err = preds - intended
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((intended - intended.mean()) ** 2) + 1e-12)
        r = float(np.corrcoef(preds, intended)[0, 1]) if len(preds) > 1 else float("nan")
        metrics = {"target": args.target, "task": "reg", "n": int(len(preds)),
                   "mae": float(np.mean(np.abs(err))),
                   "rmse": float(np.sqrt(np.mean(err ** 2))),
                   "r2": 1.0 - ss_res / ss_tot, "pearson_r": r,
                   "intended_mean": float(intended.mean()),
                   "predicted_mean": float(preds.mean())}
    else:
        acc = float((preds == intended).mean())
        recalls, cm = [], np.zeros((num_classes, num_classes), dtype=int)
        for c in range(num_classes):
            m = intended == c
            if m.sum() > 0:
                recalls.append(float((preds[m] == c).mean()))
        for t, p in zip(intended, preds):
            cm[int(t), int(p)] += 1
        metrics = {"target": args.target, "task": "cls", "n": int(len(preds)),
                   "num_classes": num_classes, "label_map": label_map,
                   "accuracy": acc,
                   "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
                   "confusion": cm.tolist(),
                   "class_order": [inv_map[c] for c in range(num_classes)]}

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("[adherence] metrics:\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
