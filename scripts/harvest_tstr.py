#!/usr/bin/env python3
"""Collect TSTR runs (scripts/train_TSTR.sh) into one results CSV.

One row per run, named <source>_<task>_s<seed>. Reported metrics come from
test_metrics.json -- the held-out test set scored once with best.pt. Validation
appears only as the value at the selected epoch, for reference; it is the maximum
over epochs and must not be reported as the result.

Refuses to silently mix incomplete runs: a run with fewer than --epochs unique
epochs or no test_metrics.json is listed as a problem and left out. A resumed run
can repeat the epoch that was interrupted, so history is de-duplicated by epoch
(last occurrence wins).

Also checks the two invariants that make the comparison valid: within a task every
source trained on the same n, and every run was scored on the same test n.

    python3 scripts/harvest_tstr.py --runs_dir <tstr>/runs --out_csv results_tstr_v2.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

PRIMARY = {"age": ("test_mae", "lower"), "sex": ("test_bacc", "higher"), "dx": ("test_bacc", "higher")}


def _gpus(run_dir: str):
    logs = sorted(glob.glob(os.path.join(run_dir, "logs", "*.log")), key=os.path.getmtime)
    for fp in reversed(logs):
        m = re.findall(r"gpus=(\d+)", open(fp, errors="ignore").read())
        if m:
            return int(m[-1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    rows, problems = [], []
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "*_s[0-9]*"))):
        name = os.path.basename(d)
        try:
            source, task, s = name.rsplit("_", 2)
            seed = int(s[1:])
        except ValueError:
            continue
        meta_fp, hist_fp, test_fp = (os.path.join(d, f) for f in
                                     ("run_meta.json", "history.jsonl", "test_metrics.json"))
        if not (os.path.isfile(meta_fp) and os.path.isfile(hist_fp)):
            problems.append(f"{name}: not started"); continue
        hist = {}
        for line in open(hist_fp):
            if line.strip():
                h = json.loads(line); hist[h["epoch"]] = h
        if len(hist) < args.epochs:
            problems.append(f"{name}: {len(hist)}/{args.epochs} epochs -- resubmit the same TASK/SOURCE/SEED"); continue
        if not os.path.isfile(test_fp):
            problems.append(f"{name}: finished but no test_metrics.json -- resubmit (it resumes and scores)"); continue
        meta, test = json.load(open(meta_fp)), json.load(open(test_fp))
        hb, hl = hist[test["from_epoch"]], hist[max(hist)]
        row = {"task": task, "source": source, "seed": seed, "gpus": _gpus(d),
               "n_train": meta["n_train"], "n_valid": meta["n_valid"], "n_test": test["n"],
               "epochs_run": len(hist), "best_epoch": test["from_epoch"]}
        if task == "age":
            row.update(test_mae=test["mae"], test_rmse=test["rmse"], test_r2=test["r2"],
                       val_mae_at_best=hb["mae"],
                       train_mae_at_best=hb["train_mae"], train_mae_final=hl["train_mae"])
        else:
            row.update(test_acc=test["acc"], test_bacc=test["balanced_acc"],
                       val_bacc_at_best=hb["balanced_acc"],
                       train_loss_at_best=hb["train_loss"], train_loss_final=hl["train_loss"])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["task", "source", "seed"])
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}: {len(df)} complete runs")

    for task, g in df.groupby("task"):
        for col in ("n_train", "n_test", "gpus"):
            if g[col].nunique() > 1:
                problems.append(f"{task}: {col} differs across runs {sorted(g[col].unique())} -- not comparable")
        metric, _ = PRIMARY[task]
        summ = g.groupby("source")[metric].agg(["mean", "std", "count"]).round(4)
        print(f"\n{task}: {metric} (test), mean / std over seeds\n{summ.to_string()}")

    if problems:
        print("\nPROBLEMS:"); print("\n".join(f"  - {p}" for p in problems))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
