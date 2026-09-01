"""Build the shuffled real-reference CSVs for the gFID grid.

WHY THIS EXISTS — the trap it closes:
    ``compute_metric.py`` uses the base CSV for two different things:
      * the REAL reference set   -> adapter.load_manifest(csv, n=num_images)
                                    == the FIRST n rows  (datasets/base.py:48)
      * the GENERATION conditions -> rng.integers(0, len(df), num_images)
                                    == uniform over ALL rows (compute_metric.py:357)
    The pooled manifest is cohort-ordered and UKB-dominated, so on the pooled
    ``all_<modality>`` slices those two disagree completely:

        all_T1 train = 26,146 rows
          first 2500 rows : {ukb: 2500}                       <- would be the reference
          true mix        : {ukb: 20201, adni: 2491, oasis: 1098,
                             brats: 1000, hcp: 891, ixi: 465} <- what gets generated

    i.e. the headline per-modality FID would silently score "pooled T1 generations
    vs pure UKB" instead of "vs pooled T1". Pre-shuffling the CSV makes the head-n
    slice an unbiased sample, so reference and conditions come from one distribution.

    (Single cohort x modality cells are homogeneous and were never affected; this
    matters for the all_* slices, which are the primary reporting granularity.)

REFERENCE SPLIT = train (decided 2026-09-01). The held-out test split is far too
small on the rare cells for a stable FID reference (ixi_T1 test = 58) and FID is
bounded by the SMALLER of the two sets, so raising n_gen alone cannot fix it.
Train also matches the usual FID convention (reference = the data distribution
being modelled). The cost is that a memorising model would be flattered, so the
nearest-neighbour memorisation check on test is mandatory, not optional.

    python scripts/build_gfid_slices.py \
        --manifest csv_files/pooled_manifest_train.csv \
        --out_dir  csv_files/gfid_slices
"""
from __future__ import annotations

import argparse
import os
import zlib

import numpy as np
import pandas as pd

# Primary granularity = per-modality (all cohorts). Our diffusion UNet is the
# B-config (cohort token OFF), so modality is the agreed FID slice; the
# per-cohort x modality cells are the coverage/diagnostic breakdown.
MODALITIES = ["T1", "T2", "FLAIR"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="csv_files/pooled_manifest_train.csv")
    p.add_argument("--out_dir", default="csv_files/gfid_slices")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Evaluation n per slice type. Modality slices carry the arm ranking, so they get
# the large n; the 13 cells are a coverage breakdown and run cheaper.
N_EVAL = {"modality": 2500, "cell": 500}


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.manifest)

    # vae_only rows (brats_T1c) are in the VAE corpus but NOT in the diffusion
    # vocabulary -- T1c exists only in BraTS and is always dx=tumor, so a
    # "healthy T1c" condition is out of distribution (plan sec 4.2). 13 generation
    # cells, not the 14 the reconstruction tables use.
    df = df[df["vae_only"] == 0].copy()

    rows = []
    for mod in MODALITIES:
        sub = df[df["modality"] == mod]
        rows.append((f"all_{mod}", "modality", sub))
        for coh in sorted(sub["cohort"].unique()):
            rows.append((f"{coh}_{mod}", "cell", sub[sub["cohort"] == coh]))

    print(f"{'slice':16s} {'type':9s} {'n_avail':>8s}  cohort mix (head-2500 before/after shuffle)")
    for name, kind, sub in rows:
        # One RNG per slice keyed on the slice name so adding a slice later never
        # reshuffles the ones already evaluated. crc32, not hash(): Python salts
        # str hashing per process (PYTHONHASHSEED), so hash() would give AIBIO and
        # GSDS different shuffles of the same slice and silently different
        # reference sets. This must be reproducible across machines.
        seed = args.seed + (zlib.crc32(name.encode()) % 10_000)
        shuffled = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out = os.path.join(args.out_dir, f"{name}.csv")
        shuffled.to_csv(out, index=False)

        # Null floor (real-vs-real): two DISJOINT halves of the same slice, giving
        # the FID a perfect generator would score at this n. Without it a gFID of
        # 15 cannot be read as good or bad. FID falls with n, so where a slice is
        # too small to yield two full-size halves the null is computed on the
        # largest disjoint pair available -> a CONSERVATIVE (too-high) floor.
        n_eval = N_EVAL[kind]
        n_half = min(n_eval, len(shuffled) // 2)
        shuffled.head(n_half).to_csv(
            os.path.join(args.out_dir, f"{name}_nullA.csv"), index=False)
        shuffled.iloc[n_half:2 * n_half].to_csv(
            os.path.join(args.out_dir, f"{name}_nullB.csv"), index=False)

        n = len(shuffled)
        if kind == "modality":
            before = sub.head(2500)["cohort"].value_counts().to_dict()
            after = shuffled.head(2500)["cohort"].value_counts().to_dict()
            flag = "" if n_half == n_eval else f"  (null n={n_half}, conservative)"
            print(f"{name:16s} {kind:9s} {n:8d}  n_eval={n_eval}{flag}\n{'':16s} {'':9s} {'':8s}  head2500 {before} -> {after}")
        else:
            flag = "" if n_half == n_eval else f"  (null n={n_half}, conservative)"
            print(f"{name:16s} {kind:9s} {n:8d}  n_eval={n_eval}{flag}")

    print(f"\nwrote {len(rows)} slice CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
