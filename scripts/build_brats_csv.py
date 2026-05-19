"""Build train.csv / valid.csv for BraTS-GLI 2023.

The Synapse release has no demographics; each case is a directory with five
NIfTI files (t1n / t1c / t2w / t2f / seg). We emit one row per case with
``rel_path`` pointing at the chosen training modality (default t1n) and
``rel_path_seg`` pointing at the segmentation mask (used by mask-conditional
downstream tasks).

Usage:
    python scripts/build_brats_csv.py \
        --data-dir /data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
        --modality t1n \
        --out-dir csv_files \
        --split 0.9 --seed 42
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="BraTS root containing BraTS-GLI-XXXXX-YYY/ case dirs.")
    parser.add_argument("--modality", default="t1n",
                        choices=["t1n", "t1c", "t2w", "t2f"],
                        help="Training modality whose path goes into rel_path.")
    parser.add_argument("--out-dir", default="csv_files")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    case_dirs = sorted(glob.glob(os.path.join(args.data_dir, "BraTS-GLI-*-*")))
    if not case_dirs:
        raise SystemExit(f"No BraTS-GLI-*-* directories under {args.data_dir}")

    rows = []
    for cdir in case_dirs:
        eid = os.path.basename(cdir)
        rel_img = os.path.join(eid, f"{eid}-{args.modality}.nii.gz")
        rel_seg = os.path.join(eid, f"{eid}-seg.nii.gz")
        # Cheap consistency check: skip cases missing either file.
        if not (os.path.exists(os.path.join(args.data_dir, rel_img))
                and os.path.exists(os.path.join(args.data_dir, rel_seg))):
            continue
        rows.append({"eid": eid, "rel_path": rel_img, "rel_path_seg": rel_seg})
    df = pd.DataFrame(rows)
    print(f"Discovered {len(df)} cases (out of {len(case_dirs)} directories).")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    cut = int(len(df) * args.split)
    train_df = df.iloc[order[:cut]].sort_values("eid").reset_index(drop=True)
    valid_df = df.iloc[order[cut:]].sort_values("eid").reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "brats_train.csv")
    valid_path = os.path.join(args.out_dir, "brats_valid.csv")
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    print(f"Wrote {len(train_df)} train rows -> {train_path}")
    print(f"Wrote {len(valid_df)} valid rows -> {valid_path}")


if __name__ == "__main__":
    main()
