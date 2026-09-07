#!/usr/bin/env python3
"""Macro-average the per-cell sheets (00, 02) into 05 and 06 for trend reading.

WHY THIS IS A SEPARATE SHEET AND NOT A ROW IN 01/03. A macro-average of per-cell
FIDs is NOT the pooled-slice FID that 01 and 03 report, and the two must never be
compared. gFID all_T1 is 10.58 as one FID over one pooled sample against 25.24 as
the mean of its six cell FIDs — the pooled sample lets a model cover the mixture
without covering any single cell, while the macro-average charges it for every cell
equally. Both are legitimate; they answer different questions. Column names here
carry the `macro_` prefix so a reader cannot mistake one for the other.

WHY `cell_set` HAS AN ex_brats ROW. BraTS is the only pathological cohort, and it
is an outlier in GENERATION but not in RECONSTRUCTION — brats gen inception FID is
75.8 (T1) against ukb_T1's 8.9, while brats RECON beats the corpus average on every
perceptual metric (PSNR 33.7-34.0 vs 32.2-32.4, LPIPS 0.024-0.026 vs 0.025-0.027).
So the VAE encodes and decodes tumours fine and the diffusion prior cannot generate
them. An all-cell macro-average buries that: one cell at 75.8 moves a 13-cell mean
by ~5 points on its own. The ex_brats row isolates the healthy-brain regime so the
data-quantity trend is readable, and the pair of rows makes the pathology gap a
number rather than a caveat.

DO NOT report ex_brats alone. Dropping the only pathological cohort because it is
hard is the same move as picking whichever decoder wins, and the tumour-segmentation
augmentation arm depends on exactly the volumes this would hide.

Usage:  python3 scripts/build_cell_averages.py
"""
from __future__ import annotations

import pandas as pd

JP = "journal_plan"
KEYS_RECON = ["exp", "ckpt", "decoder", "eval_split"]
KEYS_GEN = ["exp", "ckpt", "decoder", "unet_ckpt", "cond_config", "guidance", "eval_split"]
# Per-plane columns ride along with their Avg. The three planes are not
# interchangeable — a 3D volume's xy, yz and zx slices differ in resolution and in
# how much anatomy a slice contains — so a model can be uniformly mediocre or
# strong on one plane and weak on another, and only the Avg is quoted in the paper.
# Keeping the planes here makes that visible without a second file.
FID_BLOCK = ["rad_fid_xy", "rad_fid_yz", "rad_fid_zx", "rad_fid_avg",
             "inc_fid_xy", "inc_fid_yz", "inc_fid_zx", "inc_fid_avg"]
METRICS_RECON = ["lpips", "psnr", "ssim", "ssim_fg"] + FID_BLOCK + ["swav_fid", "dino_fid_avg"]
METRICS_GEN = FID_BLOCK + ["swav_fid", "dino_fid_avg", "inc_fid_null",
                           "adh_modality_bacc", "adh_sex_bacc", "adh_dx_bacc",
                           "adh_age_mae", "adh_age_r2"]


def macro(src: str, keys: list[str], metrics: list[str], out: str) -> None:
    df = pd.read_csv(f"{JP}/{src}")
    have = [m for m in metrics if m in df.columns]
    rows = []
    for label, sub in (("all", df), ("ex_brats", df[df["cohort"] != "brats"])):
        # Rows whose metric block is entirely empty are separate measurement sets
        # (00 keeps LPIPS/PSNR rows apart from FID rows because their checkpoints
        # differ), so average each metric over the rows that actually carry it
        # rather than dropping a row for missing a column it never had.
        g = sub.groupby(keys, dropna=False)
        agg = g[have].mean(numeric_only=True)
        agg.insert(0, "cell_set", label)
        agg["n_cells"] = g["cell"].nunique()
        rows.append(agg.reset_index())
    res = pd.concat(rows, ignore_index=True)
    res = res.rename(columns={m: f"macro_{m}" for m in have})
    res.insert(0, "source", "scripts/build_cell_averages.py")
    res = res[["cell_set"] + keys + ["n_cells"] +
              [f"macro_{m}" for m in have] + ["source"]]
    res = res.sort_values(["cell_set"] + keys).round(4)
    res.to_csv(f"{JP}/{out}", index=False)
    print(f"wrote {JP}/{out}  ({len(res)} rows)")


if __name__ == "__main__":
    macro("00_results_pooled_recon_percell.csv", KEYS_RECON, METRICS_RECON,
          "05_results_pooled_recon_cellavg.csv")
    macro("02_results_pooled_gen_percell.csv", KEYS_GEN, METRICS_GEN,
          "06_results_pooled_gen_cellavg.csv")
