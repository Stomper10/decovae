#!/usr/bin/env python3
"""Training CSVs for TSTR (Train on Synthetic, Test on Real), with a real control
that is actually matched to the synthetic set.

WHY THE LAST ROUND'S REAL CONTROL WAS NOT A CONTROL. It was built with --real_limit,
which is df.head(n), on adherence CSVs that are sorted by cohort. head(1974) of the
dx pool is 100% ADNI (its dx split, MCI 949 / healthy 753 / AD 272, is exactly what
GSDS reported training on), and head(6342) of the age pool is 100% UKB. The synthetic
sets span every cohort in their slice, and the validation set is 31% OASIS for dx.
The comparison was "one cohort, real" against "all cohorts, synthetic" -- which is
how synthetic came out ABOVE real on dx.

WHAT THIS DOES INSTEAD. For each task it reads the synthetic volumes' .cond.json
sidecars (the label is the condition the volume was generated from), keeps the
reported corpus, and counts volumes per stratum:

    age  (cohort, modality)
    sex  (cohort, modality, sex)
    dx   (cohort, modality, dx)

Then every source -- each arm's synthetic set and the real control -- is sampled to
the SAME count in every stratum: the minimum over the arms and the real pool. Every
training set therefore has identical size, cohort mix, modality mix and label
balance, and the only thing that differs is where the volumes came from.

The arms should already agree. compute_metric.py draws conditions with one seed from
one BASE_CSV, so all three arms were generated from the same condition list. A
stratum where they differ means missing volumes; it is reported, and the minimum
absorbs it.

Real rows come from the train split, validation and test from other splits, so the
sets are disjoint by construction.

    python3 scripts/build_tstr_csvs.py \\
        --synth_root /leelabsg/data/.../pooled/stage1 \\
        --arm maisi=pooled-maisi-kl8e4-eff32-s1-Acfg ... \\
        --pool_dir <dir with {age,sex,dx}_train_pool.csv> --out_dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib

import pandas as pd

COHORTS_4CO = ["ukb", "adni", "ixi", "oasis"]

# Which generated cells feed each task. Cells are named by the tags GSDS already
# used for A-config / orig decoder / g=3.0. age and sex share the pooled modality
# slices; dx uses the dementia cells, since the pooled slices carry almost no MCI/AD.
TASKS = {
    "age": {"cohorts": COHORTS_4CO, "labels": None,
            "strata": ["cohort", "modality"],
            "cells": ["g30_03_Acfg_orig_all_T1", "g30_03_Acfg_orig_all_T2",
                      "g30_03_Acfg_orig_all_FLAIR"]},
    "sex": {"cohorts": COHORTS_4CO, "labels": ["M", "F"],
            "strata": ["cohort", "modality", "sex"],
            "cells": ["g30_03_Acfg_orig_all_T1", "g30_03_Acfg_orig_all_T2",
                      "g30_03_Acfg_orig_all_FLAIR"]},
    "dx":  {"cohorts": ["adni", "oasis"], "labels": ["healthy", "MCI", "AD"],
            "strata": ["cohort", "modality", "dx"],
            "cells": ["g30_02_Acfg_orig_adni_T1", "g30_02_Acfg_orig_adni_FLAIR",
                      "g30_02_Acfg_orig_oasis_T1", "g30_02_Acfg_orig_oasis_T2"]},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synth_root", required=True,
                   help="Directory holding the arm run dirs; synthetic rel_paths are relative to it.")
    p.add_argument("--arm", action="append", required=True,
                   help="short=run_dir, e.g. maisi=pooled-maisi-kl8e4-eff32-s1-Acfg (repeat).")
    p.add_argument("--pool_dir", required=True,
                   help="Directory with {age,sex,dx}_train_pool.csv (real train split).")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--real_data_dir", default=None,
                   help="Real cache root. When given, pool rows whose file is absent are dropped "
                        "BEFORE sampling, so a partial cache cannot put a missing volume into the "
                        "real control and crash the run hours in.")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--seed", type=int, default=0,
                   help="Sampling seed. Fixed across training seeds on purpose: the training "
                        "seed should vary initialisation and order, not the data.")
    return p.parse_args()


def _clean(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict]:
    spec = TASKS[task]
    n0 = len(df)
    df = df[df["cohort"].isin(spec["cohorts"])]
    n_cohort = n0 - len(df)
    if spec["labels"] is None:
        df = df.assign(**{task: pd.to_numeric(df[task], errors="coerce")})
        df = df[df[task].notna()]
    else:
        df = df[df[task].isin(spec["labels"])]
    return df.reset_index(drop=True), {"dropped_cohort": n_cohort,
                                       "dropped_label": n0 - n_cohort - len(df)}


def read_synth(synth_root: str, run_dir: str, task: str) -> tuple[pd.DataFrame, dict]:
    rows, missing_vol = [], 0
    for cell in TASKS[task]["cells"]:
        vol_dir = os.path.join(synth_root, run_dir, "cells", cell, "outputs", "volumes")
        sides = sorted(glob.glob(os.path.join(vol_dir, "gen_*.cond.json")))
        if not sides:
            raise SystemExit(f"[FATAL] no gen_*.cond.json under {vol_dir}")
        for side in sides:
            vol = side[: -len(".cond.json")] + ".nii.gz"
            if not os.path.isfile(vol):
                missing_vol += 1
                continue
            c = json.load(open(side))
            rows.append({"rel_path": os.path.relpath(vol, synth_root), task: c.get(task),
                         "cohort": c.get("cohort"), "modality": c.get("modality")})
    df, info = _clean(pd.DataFrame(rows), task)
    info.update({"read": len(rows), "missing_volume": missing_vol})
    return df, info


def _rng(seed: int, *keys: str) -> int:
    # crc32 rather than hash(): Python salts str hashing per process, and this has
    # to give the same sample on AIBIO and GSDS.
    return seed + zlib.crc32("|".join(keys).encode()) % 100_000


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    arms = dict(a.split("=", 1) for a in args.arm)
    summary = {}

    for task in args.tasks:
        spec, strata = TASKS[task], TASKS[task]["strata"]
        sources, info = {}, {}
        for short, run_dir in arms.items():
            sources[f"synth_{short}"], info[f"synth_{short}"] = read_synth(args.synth_root, run_dir, task)
        real = pd.read_csv(os.path.join(args.pool_dir, f"{task}_train_pool.csv"))
        real_df, real_info = _clean(real, task)
        if args.real_data_dir:
            ok = real_df["rel_path"].map(lambda r: os.path.isfile(os.path.join(args.real_data_dir, r)))
            real_info["missing_file"] = int((~ok).sum())
            if real_info["missing_file"]:
                print(f"[warn] {task}: {real_info['missing_file']} real pool rows have no file under "
                      f"{args.real_data_dir}; excluded from sampling")
            real_df = real_df[ok].reset_index(drop=True)
        sources["real_ctrl"], info["real_ctrl"] = real_df, real_info

        counts = pd.DataFrame({s: d.groupby(strata).size() for s, d in sources.items()}).fillna(0).astype(int)
        synth_cols = [c for c in counts.columns if c.startswith("synth_")]
        disagree = counts[synth_cols].nunique(axis=1) > 1
        target = counts.min(axis=1)

        print(f"\n=== {task}: per-stratum counts (target = min) ===")
        print(counts.assign(target=target).to_string())
        if disagree.any():
            print(f"[warn] {int(disagree.sum())} strata differ between arms "
                  f"(missing volumes?) -- the minimum absorbs it:")
            print(counts[disagree].to_string())

        for src, df in sources.items():
            parts = []
            for key, g in df.groupby(strata):
                n = int(target.get(key, 0))
                if n:
                    parts.append(g.sample(n=n, random_state=_rng(args.seed, task, src, str(key))))
            out = pd.concat(parts).sample(frac=1.0, random_state=_rng(args.seed, task, src))
            out = out[["rel_path", task, "cohort", "modality"]]
            fp = os.path.join(args.out_dir, f"train_{src}_{task}.csv")
            out.to_csv(fp, index=False)
            info[src]["written"] = len(out)
        sizes = {s: info[s]["written"] for s in sources}
        assert len(set(sizes.values())) == 1, sizes
        print(f"[{task}] every source = {next(iter(sizes.values()))} volumes | " +
              " | ".join(f"{s}: read {info[s].get('read', len(sources[s]))}, "
                         f"-cohort {info[s]['dropped_cohort']}, -label {info[s]['dropped_label']}"
                         + (f", -missing {info[s]['missing_volume']}" if 'missing_volume' in info[s] else "")
                         + (f", -nofile {info[s]['missing_file']}" if 'missing_file' in info[s] else "")
                         for s in sources))
        summary[task] = {"n_per_source": next(iter(sizes.values())),
                         "strata": strata, "sources": info,
                         "arms_disagree_strata": int(disagree.sum())}

    with open(os.path.join(args.out_dir, "tstr_csv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(f"\nwrote {args.out_dir}/train_<source>_<task>.csv + tstr_csv_summary.json")


if __name__ == "__main__":
    main()
