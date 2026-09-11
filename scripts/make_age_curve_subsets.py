#!/usr/bin/env python3
"""Stratified, nested subsamples of the brain-age training set, for the data-scaling curve.

Same design as scripts/make_dx_curve_subsets.py, for the same reasons -- see its
header. In short: --real_limit is df.head(n) and the pool is in manifest order, so
head(n) would be 100% UKB until n is in the tens of thousands, and cohort would be
confounded with n.

CORPUS. The four reported cohorts (ukb, adni, ixi, oasis): 46,038 volumes. Plan
section 6 already restricts the brain-age regressor to cohorts with exact ages; the
4-cohort decision also removes hcp.

STRATA = (cohort, modality, age quintile). Age is the target, so it is stratified
too: a subset that happened to under-sample the old end would change what the curve
measures. Quintile edges come from the whole pool (20 / 56 / 62 / 67 / 71 / 97),
giving 40 strata; the smallest holds 34 volumes, so even n=250 keeps every stratum
(actual 259 -- each stratum contributes at least one).

NESTED: one shuffled order per stratum and every size takes a prefix of it, so the
curve varies only in quantity.

    python3 scripts/make_age_curve_subsets.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

OUT = "/data/wonyoungjang/decodata/pooled/downstream"
COHORTS_4CO = ["ukb", "adni", "ixi", "oasis"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{OUT}/tstr/csv/age_train_pool.csv")
    ap.add_argument("--out_dir", default=f"{OUT}/adherence/age_curve")
    ap.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 2500, 5000, 10000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    df = df[df["cohort"].isin(COHORTS_4CO)].reset_index(drop=True)
    df["age_q"] = pd.qcut(df["age"], 5, labels=False, duplicates="drop")
    os.makedirs(args.out_dir, exist_ok=True)
    strata = ["cohort", "modality", "age_q"]

    order = {k: g.sample(frac=1.0, random_state=args.seed) for k, g in df.groupby(strata)}
    total = len(df)
    print(f"pool {total}  strata {len(order)}  age quintile edges "
          f"{[round(x, 1) for x in df['age'].quantile([0, .2, .4, .6, .8, 1]).tolist()]}")
    print(f"{'n':>6} {'actual':>7} {'age mean':>9} {'age sd':>7}   cohorts")
    for n in args.sizes + [total]:
        parts = []
        for g in order.values():
            take = max(1, round(len(g) * n / total)) if n < total else len(g)
            parts.append(g.head(min(take, len(g))))
        sub = pd.concat(parts).sample(frac=1.0, random_state=args.seed)
        tag = "full" if n >= total else str(n)
        sub[["rel_path", "age", "cohort", "modality"]].to_csv(
            os.path.join(args.out_dir, f"age_train_n{tag}.csv"), index=False)
        print(f"{tag:>6} {len(sub):>7} {sub['age'].mean():>9.2f} {sub['age'].std():>7.2f}   "
              f"{dict(sub['cohort'].value_counts())}")
    print(f"\nwrote {args.out_dir}/age_train_n*.csv")


if __name__ == "__main__":
    main()
