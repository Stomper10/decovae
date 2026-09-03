#!/usr/bin/env python3
"""Per-modality latent geometry for the pooled stage1 arms — CPU only.

WHY: generation degrades far more on T2 than on T1/FLAIR, and the degradation
ordering (maisi +31% < vad +83% < sid +141%) is inverse to posterior sigma. Two
mechanisms explain that, and they imply different fixes:

  (1) GLOBAL SHARPENING — the aux regularizers make the latent more
      deterministic everywhere, so p(z) is uniformly harder to model and the
      scarce modality simply cannot pay for it. Fix: pick kl/lambda for
      generation instead of reconstruction.

  (2) MODALITY FRAGMENTATION — cov/var/cor are computed on BATCH-AGGREGATE
      channel statistics, and ~82% of every batch is T1+FLAIR. The latent
      geometry is then shaped for the majority modalities and T2 inherits a
      geometry tuned elsewhere. Fix: stratify the regularizer per modality.

This script tests (2) directly. It measures, per modality, exactly the
quantities the regularizers act on — channel correlation (cor_loss), channel
covariance off-diagonal (cov_loss), and per-channel variance (var_loss) — and
compares T2 against T1 WITHIN each arm.

THE CONTROL IS maisi. It has no aux regularizer, so its per-modality levels are
the natural ones; a regularized arm is scored by how much of the gap DOWN FROM
that control it closed in each modality. Two traps this avoids: a raw per-arm
number confounds "T2 latents differ" with "the penalty missed T2", and a raw
T2/T1 ratio is meaningless once the penalty works, because it divides one
near-zero by another (sid T1 0.00563 vs T2 0.00686 reads as "+22%" while the
absolute gap is 0.0012 against a 0.149 drop from the control).

WHAT IS AND IS NOT MEASURABLE HERE: the saved <sid>_emb.nii.gz is the SAMPLED
latent z = z_mu + eps*z_sigma (extract_emb.py:224), not z_mu, and the per-volume
mu/sigma .npy files were cleaned up after extraction. So posterior sigma — the
quantity mechanism (1) is about — cannot be recovered without re-encoding on a
GPU. Sigma contributes only ~0.25-1% of the latent variance at the measured
sigma/std of 5-10%, so it is negligible for the geometry read here. Hence this
script can CONFIRM (2); it can only disfavour it, not confirm (1), by elimination.

Usage:
  python3 scripts/analyze_latent_by_modality.py --n 400 --workers 12
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import numpy as np

STAGE1 = "/data/wonyoungjang/decodata/pooled/stage1"
ARMS = [
    ("maisi", "pooled-maisi-kl8e4-eff32-s1"),
    ("sid-cor50", "pooled-sid-cor50-kl8e4-eff32-s1"),
    ("vad-cov1var1", "pooled-vad-cov1var1-kl8e4-eff32-s1"),
]
MODALITIES = ["T1", "T2", "FLAIR"]


def _suffstats(args):
    """(n, sum_c, sum_outer) over voxels for one chunk of latent files.

    Sufficient statistics rather than raw voxels: a 4x48^3 latent is 442 KB, and
    accumulating (4,) and (4,4) per file keeps memory flat regardless of --n.
    """
    import nibabel as nib

    paths = args
    n = 0
    s = np.zeros(4, dtype=np.float64)
    S = np.zeros((4, 4), dtype=np.float64)
    bad = 0
    for p in paths:
        try:
            arr = np.asarray(nib.load(p).dataobj, dtype=np.float64)  # (48,48,48,4)
        except Exception:
            bad += 1
            continue
        X = arr.reshape(-1, arr.shape[-1])
        n += X.shape[0]
        s += X.sum(0)
        S += X.T @ X
    return n, s, S, bad


def effective_rank(cov: np.ndarray) -> float:
    """exp(entropy of the normalised eigenspectrum) — 4.0 = all channels used
    equally, 1.0 = the latent collapsed onto one direction."""
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return float("nan")
    p = w / w.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def geometry(n, s, S):
    mean = s / n
    cov = S / n - np.outer(mean, mean)
    var = np.diag(cov)
    sd = np.sqrt(np.clip(var, 1e-12, None))
    corr = cov / np.outer(sd, sd)
    off = ~np.eye(4, dtype=bool)
    return {
        "std": float(np.sqrt(var.mean())),          # RMS per-channel std
        "mean": float(mean.mean()),
        "var_spread": float(var.max() / max(var.min(), 1e-12)),
        "offdiag_corr": float(np.abs(corr[off]).mean()),   # cor_loss target
        "offdiag_cov": float(np.abs(cov[off]).mean()),     # cov_loss target
        "erank": effective_rank(cov),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="volumes sampled per modality")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="journal_plan/results_pooled_latent_by_modality.csv")
    args = ap.parse_args()

    import multiprocessing as mp

    # The SAME subject list for every arm: filenames are identical across arms, so
    # sampling once removes subject-sampling noise from the arm comparison entirely.
    ref = os.path.join(STAGE1, ARMS[0][1], "embeddings")
    by_mod = defaultdict(list)
    for f in os.listdir(ref):
        if not f.endswith("_emb.nii.gz"):
            continue
        for m in MODALITIES:
            if f.endswith(f"_{m}_emb.nii.gz"):
                by_mod[m].append(f)
                break
    rng = random.Random(args.seed)
    picked = {m: rng.sample(v, min(args.n, len(v))) for m, v in by_mod.items()}
    print("sampled:", {m: f"{len(v)}/{len(by_mod[m])}" for m, v in picked.items()}, flush=True)

    rows = []
    for label, arm in ARMS:
        emb = os.path.join(STAGE1, arm, "embeddings")
        for m in MODALITIES:
            files = [os.path.join(emb, f) for f in picked[m]]
            chunks = [files[i::args.workers] for i in range(args.workers)]
            with mp.Pool(args.workers) as pool:
                res = pool.map(_suffstats, chunks)
            n = sum(r[0] for r in res)
            s = sum(r[1] for r in res)
            S = sum(r[2] for r in res)
            bad = sum(r[3] for r in res)
            g = geometry(n, s, S)
            g.update(arm=label, modality=m, n_vol=len(files), n_vox=n, unreadable=bad)
            rows.append(g)
            print(f"  {label:13s} {m:6s} std={g['std']:.4f} "
                  f"offdiag_corr={g['offdiag_corr']:.5f} "
                  f"offdiag_cov={g['offdiag_cov']:.5f} erank={g['erank']:.3f}"
                  + (f"  [{bad} unreadable]" if bad else ""), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)[
        ["arm", "modality", "n_vol", "n_vox", "std", "mean", "var_spread",
         "offdiag_corr", "offdiag_cov", "erank", "unreadable"]
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # ---- the actual test: how much of the gap to the unregularised control did
    # the penalty close, per modality? ------------------------------------------
    # NOT a raw T2/T1 ratio: after a successful penalty both numbers are near zero,
    # so their ratio amplifies a negligible absolute difference into a large-looking
    # percentage. (sid T1 0.00563 vs T2 0.00686 reads as "+22%" while the absolute
    # gap is 0.0012 against a 0.149 drop from the control.) Fraction-of-gap-closed
    # is scale-free in the right way and answers the actual question: did the
    # regularizer reach T2 as well as it reached T1?
    piv = df.set_index(["arm", "modality"])
    print("\n=== how much of the gap to maisi did the penalty close? ===")
    print(f"{'arm':14s}{'metric':16s}" + "".join(f"{m:>10s}" for m in MODALITIES))
    for a, _ in ARMS:
        if a == "maisi":
            continue
        for k in ["offdiag_corr", "offdiag_cov"]:
            cells = []
            for m in MODALITIES:
                ctrl = piv.loc[("maisi", m), k]
                cells.append(f"{100 * (1 - piv.loc[(a, m), k] / ctrl):9.1f}%")
            print(f"{a:14s}{k:16s}" + "".join(cells))
    print("\n=== raw levels for reference (control first) ===")
    for k in ["std", "offdiag_corr", "erank"]:
        print(f"  {k}")
        for a, _ in ARMS:
            print(f"    {a:14s}" + "".join(f"{piv.loc[(a, m), k]:10.4f}" for m in MODALITIES))
    print("\nRead: T2 closing nearly as much of the gap as T1 means the regularizer"
          "\nreached the minority modality — mechanism (2) disfavoured, and (1) is left"
          "\nstanding by elimination (posterior sigma itself needs a GPU re-encode).")


if __name__ == "__main__":
    main()
