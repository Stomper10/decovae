#!/usr/bin/env python3
"""Stratified subsamples of the dx training set, for the data-scaling curve.

WHY NOT --real_limit. train_attr_predictor.py's --real_limit is df.head(n)
(attr_dataset._build_records), which works for the tumour segmentor because its pool
is one cohort. The dx pool is not: build_adherence_csv emits manifest order, which is
grouped by cohort, so head(n) is 100% ADNI up to n=2,500 and OASIS never appears at
all. A curve built that way measures "ADNI only vs ADNI+OASIS", not "less data vs
more data" -- cohort would be confounded with n, and cohort is exactly the axis the
A-config work says the model is sensitive to.

WHAT THIS DOES INSTEAD. Samples each (cohort, dx) stratum in proportion to the full
pool, so every subset is a scale model of the whole: same class balance, same cohort
mix. Subsets are NESTED (a fixed seed and a growing head of one shuffled order), so
the curve varies only in quantity -- disjoint draws would add subset-composition
noise on top of the effect being measured, the same reasoning as the segmentor pack.

Usage:  python3 scripts/make_dx_curve_subsets.py --sizes 250 500 1000 2500
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

ADH = "/data/wonyoungjang/decodata/pooled/downstream/adherence"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{ADH}/dx_clf/adh_dx_train.csv")
    ap.add_argument("--out_dir", default=f"{ADH}/dx_curve")
    ap.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 2500])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    os.makedirs(args.out_dir, exist_ok=True)
    strata = ["cohort", "dx"]

    # One shuffled order per stratum; every size takes a prefix of it, which is what
    # makes the subsets nested.
    order = {k: g.sample(frac=1.0, random_state=args.seed)
             for k, g in df.groupby(strata, dropna=False)}
    total = len(df)

    print(f"pool {total}  strata {len(order)}")
    print(f"{'n':>6} {'actual':>7}  " + "  ".join(f"{d:>8}" for d in ("healthy", "MCI", "AD"))
          + "   cohorts")
    for n in args.sizes + [total]:
        parts = []
        for k, g in order.items():
            take = max(1, round(len(g) * n / total)) if n < total else len(g)
            parts.append(g.head(min(take, len(g))))
        sub = pd.concat(parts).sample(frac=1.0, random_state=args.seed)
        tag = "full" if n >= total else str(n)
        out = os.path.join(args.out_dir, f"dx_train_n{tag}.csv")
        sub.to_csv(out, index=False)
        vc = sub["dx"].value_counts()
        print(f"{tag:>6} {len(sub):>7}  " + "  ".join(f"{vc.get(d,0):>8}" for d in ("healthy", "MCI", "AD"))
              + f"   {dict(sub['cohort'].value_counts())}")
    print(f"\nwrote {args.out_dir}/dx_train_n*.csv")


if __name__ == "__main__":
    main()
