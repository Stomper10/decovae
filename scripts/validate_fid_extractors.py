#!/usr/bin/env python3
"""Which FID extractors actually discriminate ARMS, using recon as ground truth.

WHY THIS EXISTS. The generation sheets have no ground truth -- that is the whole
point of FID -- so an extractor's verdict there cannot be checked. The
reconstruction sheet does: LPIPS, PSNR and SSIM are measured against the input
volume, so for every (arm x decoder x cell) we know which configuration is
genuinely better. Run each candidate extractor's rFID against that and keep only
the ones that agree.

WHAT SEPARATES A MEASUREMENT FROM A CONSTANT. Over the 28 (cell x decoder-block)
points, swav names maisi 28 times, dino 27, rad names sid 24. Their answer does not
depend on the cell or on the block -- they are constants wearing a metric's
clothes. Inception names vad 17 and maisi 9, and it SWITCHES between blocks in the
direction the perceptual truth switches (orig truth leans vad, dft truth leans
maisi, and inception follows). That switching is the signature of measurement.

A constant scores well whenever the truth happens to be constant too, which is why
a single block proves nothing: dino reaches rho +0.785 on the dft block purely
because that block's truth is maisi 12/14. It is a stopped clock, not a sensor.
Pooling all 6 configurations per cell hides this even better -- swav and dino come
out at 0.49 and 0.58, but that is the DFT-vs-orig contrast, which is large enough
that naming maisi is nearly free.

THE TEST IS PAIRED. Cells differ enormously in difficulty, so a raw correlation
over all points would measure "is this cell easy" and not "can this extractor tell
configurations apart". Everything is centred within cell first.

    python3 scripts/validate_fid_extractors.py
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import spearmanr

FIDS = ["rad_fid_avg", "inc_fid_avg", "swav_fid", "dino_fid_avg"]


def quality_rank(df: pd.DataFrame) -> pd.Series:
    """Perceptual consensus rank within each cell; 1 = best.

    Three metrics rather than one because they disagree at the margin and no
    single one is authoritative. SSIM in particular is unreliable in this regime
    (lambda sweep, 2026-06-10), so it votes but does not decide.
    """
    g = df.groupby("cell")
    return (g["lpips"].rank(ascending=True)
            + g["psnr"].rank(ascending=False)
            + g["ssim"].rank(ascending=False)) / 3


def centred_rho(df: pd.DataFrame, col: str) -> tuple[float, float]:
    z_f = df[col] - df.groupby("cell")[col].transform("mean")
    z_q = df["q"] - df.groupby("cell")["q"].transform("mean")
    r = spearmanr(z_q, z_f)
    return r.statistic, r.pvalue


def report(df: pd.DataFrame, label: str) -> None:
    df = df.assign(q=quality_rank(df))
    n_cfg = df["cell"].value_counts().iloc[0]
    print(f"\n{label}")
    print(f"  {len(df)} points, {df['cell'].nunique()} cells, {n_cfg} configs/cell")
    print(f"  {'extractor':14s} {'rho':>7s} {'p':>10s}")
    for f in FIDS:
        if f not in df.columns:
            continue
        rho, p = centred_rho(df, f)
        print(f"  {f:14s} {rho:7.3f} {p:10.2e}")


def winners(df: pd.DataFrame, label: str) -> None:
    df = df.assign(q=quality_rank(df))
    cols = {"perceptual": df.loc[df.groupby("cell")["q"].idxmin(), "arm"].value_counts()}
    for f in FIDS:
        if f in df.columns:
            cols[f] = df.loc[df.groupby("cell")[f].idxmin(), "arm"].value_counts()
    print(f"\nPer-cell winner -- {label}")
    print(pd.DataFrame(cols).fillna(0).astype(int).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="journal_plan/00_results_pooled_recon_percell.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.sheet)
    d["cell"] = d["cohort"] + "_" + d["modality"]
    d["arm"] = (d["exp"].str.replace("pooled-", "", regex=False)
                        .str.replace("-kl8e4-eff32-s1", "", regex=False))

    report(d, "ALL configs (arms x decoders) -- decoder contrast INCLUDED, misleading")
    for dec in sorted(d["decoder"].unique()):
        report(d[d["decoder"] == dec], f"decoder={dec} only -- the arm contrast")
    for dec in sorted(d["decoder"].unique()):
        winners(d[d["decoder"] == dec], f"decoder={dec}")

    print("""
READING IT. An extractor may rank arms only if it is positive and significant on
EVERY arm-only block, and only if its per-cell winner column changes between blocks
the way the perceptual column does. One arm sweeping every cell in both blocks is a
constant, whatever its rho says on the block where the truth happens to agree.""")


if __name__ == "__main__":
    main()
