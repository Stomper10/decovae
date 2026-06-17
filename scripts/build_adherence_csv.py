"""Build train/val CSVs for the adherence predictors from the pooled manifest.

Emits a ``rel_path`` + target-column CSV that ``downstream.attr_dataset`` /
``brain_age_dataset`` can ingest, with per-target filtering:

  dx  : cohorts with clinical labels only (default adni,oasis) and
        dx in {healthy(=CN), MCI, AD} — tumor / NA dropped. label_map for the
        trainer: '{"healthy":0,"MCI":1,"AD":2}'.
  sex : every volume with a non-missing sex (drops brats, a few ixi/adni).
  age : every volume with a non-missing age (pooled brain-age regressor).

``--path_col`` picks the volume the predictor trains on:
  cache_key (default) → preprocessed ``.npy`` (SAME space the VAE/diffusion sees,
            so the predictor matches generation space) → rel_path = "<cache_key>.npy".
  src_path            → raw source NIfTI → rel_path = "<src_path>" (set data_dir="").

NOTE: train the predictor in the space it will be EVALUATED in (generated volumes
are VAE-decoded ≈ preprocessed space). cache_key is the safer default; see the
adherence-eval normalization note before trusting numbers across .npy↔.nii.gz.
"""
from __future__ import annotations

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="pooled_manifest_{train,valid}.csv")
    p.add_argument("--target", required=True, choices=["dx", "sex", "age"])
    p.add_argument("--out_csv", required=True)
    p.add_argument("--path_col", default="cache_key", choices=["cache_key", "src_path"])
    p.add_argument("--cohorts", default=None,
                   help="comma list to restrict cohorts (default: dx→adni,oasis; else all).")
    p.add_argument("--dx_labels", default="healthy,MCI,AD",
                   help="dx only: kept classes (drops others incl tumor/NA).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)

    cohorts = args.cohorts.split(",") if args.cohorts else (
        ["adni", "oasis"] if args.target == "dx" else None)
    if cohorts:
        df = df[df["cohort"].isin(cohorts)]

    if args.target == "dx":
        keep = args.dx_labels.split(",")
        df = df[df["dx"].isin(keep)]
    elif args.target == "sex":
        df = df[df["sex"].notna()]
    else:  # age
        df = df[df["age"].notna()]

    if args.path_col == "cache_key":
        rel = df["cache_key"].astype(str) + ".npy"
    else:
        rel = df["src_path"].astype(str)

    out = pd.DataFrame({
        "rel_path": rel.values,
        args.target: df[args.target].values,
        "cohort": df["cohort"].values,
        "modality": df["modality"].values,
    })
    out.to_csv(args.out_csv, index=False)

    print(f"[build_adherence_csv] target={args.target} path_col={args.path_col} "
          f"cohorts={cohorts or 'all'} -> {args.out_csv} ({len(out)} rows)")
    if args.target in ("dx", "sex"):
        print("  class distribution:")
        print(out[args.target].value_counts().to_string())
    else:
        print(f"  age range: {out['age'].min():.1f}–{out['age'].max():.1f}  "
              f"mean {out['age'].mean():.1f}  n={len(out)}")


if __name__ == "__main__":
    main()
