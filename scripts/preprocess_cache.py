#!/usr/bin/env python3
"""Full-batch offline preprocessing cache writer for the pooled corpus.

Reads csv_files/pooled_manifest_{split}.csv, runs the canonical pipeline
(scripts/preproc_pipeline.py) on each volume, and writes:
    {cache_root}/{cache_key}.npy   — fp16 192^3 in [0,1]
    {cache_root}/{cache_key}.json  — typed conditioning tokens for that volume

Resumable (skips existing .npy), shardable (--shard i --nshards K), and
CPU-parallel (--workers W; antspynet forced to CPU so no GPU-job-cap conflict).
Failures are logged to {cache_root}/_failures_{split}.csv (QC flag), never abort.

Usage:
    python scripts/preprocess_cache.py --split train --cache_root /data/.../cache \
        --workers 48 --reg rigid [--shard 0 --nshards 4] [--limit N]
"""
import os
# force antspynet/TF onto CPU BEFORE any tf import (avoids GPU-cap). Give each
# worker PP_THREADS threads: the antspynet skull-strip is a TF inference and is
# MUCH faster multi-threaded, so 1-thread workers cripple throughput. Tune so
# workers * PP_THREADS ~= cpus-per-task (set in the sbatch).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_T = os.environ.get("PP_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "TF_NUM_INTRAOP_THREADS"):
    os.environ.setdefault(_v, _T)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import argparse, csv, json
import numpy as np
from multiprocessing import Pool

import preproc_pipeline as P

CSV = "/data/wonyoungjang/decovae/csv_files"
TOKEN_COLS = ["modality", "age", "sex", "dx", "cdrsb"]  # severity = cdrsb


def token_dict(row):
    """Typed tokens: present value or None (=token not emitted)."""
    d = {"cohort": row["cohort"], "site": row.get("site", "")}
    for c in TOKEN_COLS:
        v = row.get(c, "")
        if v is None or str(v).strip() == "" or str(v).lower() == "nan":
            d[c] = None
        else:
            d[c] = v if c in ("modality", "sex", "dx") else float(v)
    return d


def process_one(args):
    row, cache_root, reg, mni_path = args
    key = row["cache_key"]
    npy = os.path.join(cache_root, key + ".npy")
    js = os.path.join(cache_root, key + ".json")
    if os.path.exists(npy) and os.path.exists(js):
        return (key, "skip", "")
    os.makedirs(os.path.dirname(npy), exist_ok=True)
    try:
        arr = P.preprocess_to_192(row["src_path"], row["modality"], row["cohort"],
                                  mni_path, reg=reg)
        # sanity: brain present, finite, in range
        nz = float((arr > 1e-6).mean())
        if not np.isfinite(arr).all() or nz < 0.01 or nz > 0.95:
            return (key, "qc_fail", f"nz={nz:.3f}")
        np.save(npy, arr.astype(np.float16))
        with open(js, "w") as f:
            json.dump(token_dict(row), f)
        return (key, "ok", f"nz={nz:.3f}")
    except Exception as e:
        return (key, "error", repr(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reg", default="rigid", choices=["rigid", "crop"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.cache_root, exist_ok=True)
    mni_path = P.build_mni_brain(os.path.join(a.cache_root, "mni152_1mm_brain.nii.gz"))

    rows = list(csv.DictReader(open(f"{CSV}/pooled_manifest_{a.split}.csv")))
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[{a.split}] shard {a.shard}/{a.nshards}: {len(rows)} volumes, "
          f"workers={a.workers}, reg={a.reg}", flush=True)

    tasks = [(r, a.cache_root, a.reg, mni_path) for r in rows]
    n_ok = n_skip = n_fail = 0
    fails = []
    with Pool(a.workers) as pool:
        for i, (key, status, msg) in enumerate(pool.imap_unordered(process_one, tasks, chunksize=4)):
            if status == "ok": n_ok += 1
            elif status == "skip": n_skip += 1
            else:
                n_fail += 1; fails.append([key, status, msg])
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rows)}  ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)

    fpath = f"{a.cache_root}/_failures_{a.split}_shard{a.shard}.csv"
    with open(fpath, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cache_key", "status", "msg"]); w.writerows(fails)
    print(f"DONE [{a.split} shard {a.shard}]: ok={n_ok} skip={n_skip} fail={n_fail} "
          f"-> failures: {fpath}", flush=True)


if __name__ == "__main__":
    main()
