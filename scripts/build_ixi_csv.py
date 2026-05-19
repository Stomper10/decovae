"""Build train.csv / valid.csv for IXI by joining filenames with IXI.xls.

Filenames look like ``IXI002-Guys-0828-T1.nii.gz``. The xls keys subjects on
``IXI_ID`` (plain int, no zero-padding) which matches the integer suffix of
the filename prefix.

UKB sex convention is 0=Female, 1=Male; IXI.xls codes 1=Male, 2=Female. We
re-encode to UKB's 0/1 so cross-dataset code paths can treat sex uniformly.

Usage:
    python scripts/build_ixi_csv.py \
        --data-dir /data/wonyoungjang/IXI \
        --xls /data/wonyoungjang/IXI/IXI.xls \
        --modality T1 \
        --out-dir csv_files \
        --split 0.9 --seed 42
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


_FNAME_RE = re.compile(r"^IXI(\d+)-([^-]+)-(\d+)-([A-Za-z0-9]+)\.nii\.gz$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="IXI root containing IXI-T1/, IXI-T2/, IXI-PD/, ...")
    parser.add_argument("--xls", required=True,
                        help="Path to IXI.xls demographics file.")
    parser.add_argument("--modality", default="T1",
                        help="Modality dir to scan (T1 / T2 / PD).")
    parser.add_argument("--out-dir", default="csv_files",
                        help="Where to write ixi_<split>.csv files.")
    parser.add_argument("--split", type=float, default=0.9,
                        help="Train fraction (rest goes to validation).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- 1. Discover NIfTI files in the chosen modality directory ----------
    mod_dir = os.path.join(args.data_dir, f"IXI-{args.modality}")
    paths = sorted(glob.glob(os.path.join(mod_dir, f"*-{args.modality}.nii.gz")))
    if not paths:
        raise SystemExit(f"No files found under {mod_dir}")

    rows = []
    skipped = 0
    for path in paths:
        m = _FNAME_RE.match(os.path.basename(path))
        if not m:
            skipped += 1
            continue
        ixi_int, site, study, modality = m.groups()
        rows.append({
            "eid": f"IXI{ixi_int.zfill(3)}",
            "ixi_int": int(ixi_int),
            "rel_path": os.path.relpath(path, args.data_dir),
            "site": site,
            "modality": modality,
            "study": study,
        })
    files = pd.DataFrame(rows)
    print(f"Discovered {len(files)} files (skipped {skipped} with unrecognized names).")

    # ---- 2. Join with demographics (IXI.xls) -------------------------------
    xls = pd.read_excel(args.xls)
    sex_col = "SEX_ID (1=m, 2=f)"
    # Re-encode sex to UKB's 0=F / 1=M convention (1->1, 2->0).
    xls["sex"] = (xls[sex_col] == 1).astype(float)
    xls = xls.rename(columns={"AGE": "age"})[["IXI_ID", "age", "sex"]]
    merged = files.merge(xls, left_on="ixi_int", right_on="IXI_ID", how="left")
    n_missing = merged["age"].isna().sum()
    print(f"After join: {len(merged)} rows; {n_missing} missing demographics.")
    merged = merged.drop(columns=["ixi_int", "IXI_ID"])

    # ---- 3. Train / valid split (random, seeded) ---------------------------
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(merged))
    cut = int(len(merged) * args.split)
    train_df = merged.iloc[order[:cut]].sort_values("eid").reset_index(drop=True)
    valid_df = merged.iloc[order[cut:]].sort_values("eid").reset_index(drop=True)

    # ---- 4. Write -----------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, f"ixi_{args.modality}_train.csv")
    valid_path = os.path.join(args.out_dir, f"ixi_{args.modality}_valid.csv")
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    print(f"Wrote {len(train_df)} train rows -> {train_path}")
    print(f"Wrote {len(valid_df)} valid rows -> {valid_path}")


if __name__ == "__main__":
    main()
