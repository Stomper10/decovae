#!/usr/bin/env python3
"""Build a stratified SUBSET of the pooled cache for the GSDS lambda sweep.

The full pooled cache (~862GB, 65k vol) is too large to ship to GSDS. For a
lambda (decorrelation strength) sweep we only need the *relative* ranking of
offdiag_corr-vs-recon across lambda values, which a per-cell stratified subset
estimates fine (the imbalance sampler reshapes the distribution anyway).

This does NOT copy any data. It writes:
  - csv_files/pooled_manifest_train_subset.csv   (<= TRAIN_CAP per cohort|modality)
  - csv_files/pooled_manifest_valid_subset.csv   (<= VALID_CAP per cell)
  - csv_files/pooled_subset_filelist.txt         (cache-root-relative .npy/.json)

Then transfer to GSDS with (user action, credentialed):
  rsync -a --files-from=csv_files/pooled_subset_filelist.txt \
        /data/wonyoungjang/decovae_cache/  <gsds-host>:<gsds-cache-root>/

On GSDS point the train/valid CSV envs at the *_subset.csv files. The subset
valid manifest only references shipped files, so load_manifest_stratified(112)
stays self-consistent.

Usage:
  python3 scripts/build_pooled_subset.py [--train-cap 300] [--valid-cap 16] [--seed 42]
"""
import argparse
import csv
import os
import random
from collections import defaultdict

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_files")
TRAIN_IN = os.path.join(CSV_DIR, "pooled_manifest_train.csv")
VALID_IN = os.path.join(CSV_DIR, "pooled_manifest_valid.csv")
TRAIN_OUT = os.path.join(CSV_DIR, "pooled_manifest_train_subset.csv")
VALID_OUT = os.path.join(CSV_DIR, "pooled_manifest_valid_subset.csv")
FILELIST_OUT = os.path.join(CSV_DIR, "pooled_subset_filelist.txt")
NPY_BYTES = 14_155_904  # one fp16 192^3 .npy (observed)


def stratified(rows, cap, seed):
    """Deterministic per (cohort|modality) cap."""
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["cohort"], r["modality"])].append(r)
    out = []
    for cell in sorted(by_cell):
        bucket = by_cell[cell]
        random.Random(f"{seed}|{cell[0]}|{cell[1]}").shuffle(bucket)
        out.extend(bucket[:cap])
    return out, by_cell


def load(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        return list(rd), rd.fieldnames


def write_manifest(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=300)
    ap.add_argument("--valid-cap", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_rows, header = load(TRAIN_IN)
    valid_rows, _ = load(VALID_IN)

    train_sub, train_cells = stratified(train_rows, args.train_cap, args.seed)
    valid_sub, _ = stratified(valid_rows, args.valid_cap, args.seed)

    write_manifest(TRAIN_OUT, train_sub, header)
    write_manifest(VALID_OUT, valid_sub, header)

    # cache-root-relative file list (.npy + .json per cache_key)
    keys = [r["cache_key"] for r in train_sub] + [r["cache_key"] for r in valid_sub]
    with open(FILELIST_OUT, "w") as f:
        for k in keys:
            f.write(k + ".npy\n")
            f.write(k + ".json\n")

    gb = len(keys) * NPY_BYTES / 1024**3
    print(f"train subset : {len(train_sub):>6} vol (cap {args.train_cap}/cell)")
    print(f"valid subset : {len(valid_sub):>6} vol (cap {args.valid_cap}/cell)")
    print(f"total        : {len(keys):>6} vol  ~{gb:.1f} GB  ({2*len(keys)} files incl .json)")
    print("\nper-cell (train, available -> taken):")
    for cell in sorted(train_cells):
        avail = len(train_cells[cell])
        print(f"  {cell[0]:>6}|{cell[1]:<6} {avail:>6} -> {min(avail, args.train_cap)}")
    print(f"\nwrote:\n  {TRAIN_OUT}\n  {VALID_OUT}\n  {FILELIST_OUT}")


if __name__ == "__main__":
    main()
