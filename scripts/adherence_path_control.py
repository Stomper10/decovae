#!/usr/bin/env python3
"""Does a judge trained on the .npy cache still work when fed .nii.gz?

THE GATE FOR EVERY ADHERENCE NUMBER. The judges in
``downstream/adherence/*/weights/best.pt`` were trained on the preprocessed .npy
cache, but the volumes they will score are VAE-decoded ``gen_*.nii.gz``. Those two
go through different branches of ``brain_age_dataset.build_transforms``:

    npy=True   NumpyReader -> EnsureChannelFirst(no_channel)
    npy=False  LoadImaged  -> EnsureChannelFirst -> Orientationd(axcodes=RAS)

Both end in the same Resize + percentile norm, so the ONLY difference is
``Orientationd``. Generations are written with an identity affine
(compute_metric.py:572, ``np.eye(4)``), under which nibabel reports RAS+ and the
transform should be a no-op — but if the .npy cache is stored in some other axis
order, the judge trained on that order would be evaluated on a RAS-forced volume,
and its accuracy would collapse for a reason that has NOTHING to do with whether
the generator obeyed its condition. We would then report a true finding
("the model ignores the modality condition") that is entirely an artefact.

This script removes the guess. It takes the SAME volumes, scores them twice — once
through each branch — and reports both. The .nii.gz copies are written exactly the
way compute_metric.py writes generations (float32, identity affine), so path B is
the real generation path and not an approximation of it.

Read it as: path A is the judge's known ceiling. If path B matches, the .nii.gz
route is sound and adherence numbers mean what they say. If path B is materially
lower, FIX THE TRANSFORM BEFORE MEASURING ANYTHING.

Usage:
  python3 scripts/adherence_path_control.py --judge modality_clf --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset

from downstream.brain_age_dataset import build_transforms
from downstream.sfcn import build_sfcn, build_sfcn_classifier

ADH = "/data/wonyoungjang/decodata/pooled/downstream/adherence"
TARGET_OF = {"modality_clf": "modality", "sex_clf": "sex",
             "dx_clf": "dx", "age_reg": "age"}


def balanced_accuracy(y, p, k):
    accs = [float((p[y == c] == c).mean()) for c in range(k) if (y == c).any()]
    return float(np.mean(accs)) if accs else float("nan")


def score(model, items, resolution, axcodes, npy, task, bs, nw, device):
    ds = Dataset(data=items, transform=build_transforms(resolution, axcodes, npy=npy))
    out = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw):
            y = model(b["image"].to(device))
            out.append(y.argmax(1).cpu().numpy() if task == "cls"
                       else y.cpu().numpy().reshape(-1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=list(TARGET_OF))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--data_dir", default="/data/wonyoungjang/decovae_cache")
    ap.add_argument("--dataset_config_path", default="configs/pooled/dataset.json")
    args = ap.parse_args()

    target = TARGET_OF[args.judge]
    task = "reg" if args.judge == "age_reg" else "cls"
    ckpt = torch.load(f"{ADH}/{args.judge}/weights/best.pt",
                      map_location="cpu", weights_only=False)
    ds_cfg = json.load(open(args.dataset_config_path))
    # train_brain_age.py stores only {epoch, val_mae} -- no resolution, no
    # label_map -- while train_attr_predictor.py stores both. Mirror the fallback
    # adherence_eval.py:88 uses so the two agree on the regressor's input size.
    resolution = tuple(ckpt.get("resolution") or ds_cfg.get("resolution", [192, 192, 192]))
    axcodes = ds_cfg.get("orientation_axcodes", "RAS")

    if task == "cls":
        label_map = ckpt["label_map"]
        model = build_sfcn_classifier(num_classes=ckpt["num_classes"], in_channels=1, dropout=0.0)
    else:
        label_map = None
        model = build_sfcn(in_channels=1, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    df = pd.read_csv(f"{ADH}/{args.judge}/adh_{target}_valid.csv")
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    truth = (np.array([label_map[str(v)] for v in df[target]]) if task == "cls"
             else df[target].to_numpy(float))

    npy_items = [{"image": os.path.join(args.data_dir, p)} for p in df["rel_path"]]

    # Path B copies: identical arrays, written the way compute_metric.py writes a
    # generation. Anything that differs between A and B is therefore the READ path.
    tmp = tempfile.mkdtemp(prefix="adhctl_")
    nii_items = []
    for i, p in enumerate(df["rel_path"]):
        arr = np.load(os.path.join(args.data_dir, p)).astype(np.float32)
        q = os.path.join(tmp, f"v{i:05d}.nii.gz")
        nib.save(nib.Nifti1Image(arr, np.eye(4)), q)
        nii_items.append({"image": q})

    print(f"judge={args.judge} target={target} task={task} n={len(df)} "
          f"resolution={list(resolution)} device={device.type}", flush=True)

    pa = score(model, npy_items, resolution, axcodes, True, task,
               args.batch_size, args.num_workers, device)
    pb = score(model, nii_items, resolution, axcodes, False, task,
               args.batch_size, args.num_workers, device)

    if task == "cls":
        k = ckpt["num_classes"]
        a = balanced_accuracy(truth, pa, k)
        b = balanced_accuracy(truth, pb, k)
        print(f"  A .npy   balanced_acc = {a:.4f}   (acc {float((pa==truth).mean()):.4f})")
        print(f"  B .nii.gz balanced_acc = {b:.4f}   (acc {float((pb==truth).mean()):.4f})")
        print(f"  A vs B agreement       = {float((pa==pb).mean()):.4f}")
        verdict = abs(a - b) <= 0.02
    else:
        a = float(np.abs(pa - truth).mean()); b = float(np.abs(pb - truth).mean())
        print(f"  A .npy    MAE = {a:.3f}")
        print(f"  B .nii.gz MAE = {b:.3f}")
        print(f"  mean |predA - predB| = {float(np.abs(pa-pb).mean()):.3f}")
        verdict = abs(a - b) <= 0.5

    print(f"\n  VERDICT: {'PASS — .nii.gz path is sound' if verdict else 'FAIL — fix the transform before measuring adherence'}")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
