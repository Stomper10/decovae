#!/usr/bin/env python3
"""Cohort-restricted pooled slices for the gFID grid (supersedes make_nobrats_slices.py).

WHY THIS IS A RE-SCORE AND NOT A REGENERATION
    compute_metric.py draws generation conditions i.i.d. from BASE_CSV
    (compute_metric.py:363, rng.integers over ALL rows) and writes the FULL token
    dict to each volume's .cond.json sidecar (compute_metric.py:583). That dict
    always carries `cohort` -- datasets/pooled.py:184 emits it unconditionally and
    only the A-variant model config consumes it -- so every generated volume, A or
    B config, records the cohort of the row its condition came from.

    Keeping only the draws whose source row is in the retained cohorts therefore
    yields an i.i.d. sample from the RESTRICTED condition distribution: exactly
    what regenerating with this CSV as BASE_CSV would give, minus a new rng
    realisation. Re-scoring is strictly better here because the restricted number
    and the full-corpus number then come from the SAME volumes, so their
    difference is the restriction and nothing else.

    The cost is n. all_T2 keeps 1,145 of 3,036 rows, so a 2,500-draw generation
    survives at ~943, not 1,145. Regenerating to recover those 200 volumes is not
    worth it: FID's finite-sample bias is the same for every arm at a shared n, so
    it cannot move the ranking, which is the only thing these slices decide.

NULL FILE NAMING. The halves are written as `<slice>_<suffix>_nullA.csv`, i.e. the
restriction suffix comes BEFORE `_nullA`. This is not cosmetic: `null_row` in
scripts/eval_gfid_grid.sh looks for `${slice}_nullA.csv`, so the earlier
`all_T2_nullA_nobrats.csv` spelling was unreachable by the launcher -- which is why
the ex_brats columns ended up quoting the unrestricted floor.

NULL HALVES ARE REBUILT, NOT FILTERED. Filtering the existing _nullA/_nullB gives
unequal halves (all_T2: 564 vs 581) because the cohort mix differs between them by
chance. FID is bounded by the smaller set, so unequal halves quietly measure the
floor at the smaller n. Rebuilding from the filtered slice keeps them equal.

    Where the filtered slice cannot yield two full-size halves the floor is
    computed on the largest disjoint pair available and is therefore CONSERVATIVE
    (too high). A too-high floor flatters the generator, so the reported n_null
    must travel with the number -- see build_gfid_slices.py, same convention.

Usage:
    python3 scripts/make_subset_slices.py --drop brats hcp --suffix 4co
    python3 scripts/make_subset_slices.py --drop brats     --suffix nobrats
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

# Must match build_gfid_slices.N_EVAL["modality"]; the null floor is only
# meaningful at the n the gFID itself was measured at.
N_EVAL_MODALITY = 2500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slice_dir", default="csv_files/gfid_slices")
    p.add_argument("--drop", nargs="+", required=True,
                   help="Cohorts to remove, e.g. --drop brats hcp")
    p.add_argument("--suffix", required=True,
                   help="Output suffix, e.g. 4co -> all_T1_4co.csv")
    p.add_argument("--prefix", default="all_",
                   help="Only pooled slices need this; per-cohort slices are "
                        "already single-cohort and brats_* would come out empty.")
    p.add_argument("--n_gen", type=int, default=N_EVAL_MODALITY,
                   help="n the existing generations used, for the survivor estimate.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    drop = set(args.drop)

    # Accept only the ORIGINAL slices, never a derived one. A source slice is
    # "<prefix><modality>.csv", so after stripping the prefix its stem has no
    # underscore left; every derivative (_nullA, _nobrats, _4co, and any
    # restriction added later) picks one up. Matching on a blocklist of known
    # suffixes instead would silently feed all_T1_4co.csv back in as a source and
    # emit all_T1_4co_nobrats.csv.
    srcs = []
    for f in sorted(glob.glob(os.path.join(args.slice_dir, f"{args.prefix}*.csv"))):
        stem = os.path.basename(f)[len(args.prefix):-len(".csv")]
        if "_" not in stem:
            srcs.append(f)

    print(f"drop = {sorted(drop)}   suffix = _{args.suffix}\n")
    print(f"{'slice':16s} {'n_full':>7s} {'n_kept':>7s} {'kept%':>6s} "
          f"{'n_null':>7s} {'~n_rescore':>10s}  dropped")
    made = 0
    for f in srcs:
        base = os.path.basename(f)[:-len(".csv")]
        d = pd.read_csv(f)
        if "cohort" not in d.columns:
            print(f"  [skip] {base}: no cohort column")
            continue

        n0 = len(d)
        gone = d["cohort"].isin(drop)
        keep = d[~gone].reset_index(drop=True)   # order preserved => still shuffled
        n1 = len(keep)
        if n1 == 0:
            print(f"  [skip] {base}: empty after drop")
            continue

        out = os.path.join(args.slice_dir, f"{base}_{args.suffix}.csv")
        keep.to_csv(out, index=False)

        n_half = min(N_EVAL_MODALITY, n1 // 2)
        keep.head(n_half).to_csv(
            os.path.join(args.slice_dir, f"{base}_{args.suffix}_nullA.csv"), index=False)
        keep.iloc[n_half:2 * n_half].to_csv(
            os.path.join(args.slice_dir, f"{base}_{args.suffix}_nullB.csv"), index=False)

        # How many of the ALREADY GENERATED volumes survive the filter, in
        # expectation: conditions were drawn uniformly over the full slice.
        n_rescore = round(args.n_gen * n1 / n0)
        flag = "" if n_half == N_EVAL_MODALITY else " *"
        mix = ", ".join(f"{c} {v}" for c, v in d.loc[gone, "cohort"].value_counts().items()) or "-"
        print(f"{base:16s} {n0:7d} {n1:7d} {100*n1/n0:5.1f}% "
              f"{n_half:6d}{flag:>1s} {n_rescore:10d}  {mix}")
        made += 1

    print(f"\n{made} slices written (+ null pairs). "
          f"* = slice too small for two {N_EVAL_MODALITY}-row halves; floor is conservative.")
    print("Generated volumes are filtered by the .cond.json `cohort` key, not by "
          "index: generation draws conditions randomly, so gen_XXXX does NOT "
          "correspond to row XXXX of the slice.")


if __name__ == "__main__":
    main()
