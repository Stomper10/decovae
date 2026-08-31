#!/bin/bash
# AIBIO-side: stage the interim-trajectory VAE checkpoints (see GRID below) into a flat
# `eval-<grp>-ck<step>` layout for transfer to GSDS, where extract_emb.sh /
# compute_metric.sh treat each as its own experiment dir (no per-ckpt output
# collision — the launchers namespace outputs by EXP_NAME only).
#
# Hardlinks within the same filesystem (no 2.4GB double-copy); falls back to cp.
# After staging, rsync the staging dir to GSDS, preserving the same tree:
#   rsync -av <STAGE_OUT>/  <gsds>:<POOLED_OUTPUT_ROOT>/stage1/
#
# Usage:  bash scripts/stage_interim_ckpts.sh [STAGE_OUT_DIR]
set -euo pipefail

SRC_ROOT=/data/wonyoungjang/decodata/pooled/stage1
STAGE_OUT="${1:-${SRC_ROOT}/_xfer_interim}"

# --- grid (edit here; group names are prefixed into eval-<grp>-ck<step>) ---------
# kl8e4/eff32 best-ckpt search over the two HEALTHY models (converged valid recon
# + in-band): maisi (sf 0.9466) and sid-cor10 (sf 0.9047 — band EDGE, only 0.005
# of margin, so the band trajectory across ckpts is the point of this sweep).
# VAD is excluded: neither vad-cov50 (converged but sf 1.2114 = out of band) nor
# vad-cov1var1 (in band but valid recon never converged, SSIM 0.628) is usable —
# it is being refit (job 229609), then re-run this sweep with GROUPS/SRC repointed.
# Group names carry -kl8e4 so outputs don't collide with the earlier kl5e3-eff32
# trajectory (eval-maisi-ck80000, ...) already harvested into journal_plan/*.csv.
# NB: do NOT name this array GROUPS — that is a bash builtin holding the caller's
# Unix GIDs; assigning to it is silently ignored and the loop yields GID numbers.
MODELS=(maisi-kl8e4 sid-cor10-kl8e4)
declare -A SRC=(
  [maisi-kl8e4]=pooled-maisi-kl8e4-eff32-s1
  [sid-cor10-kl8e4]=pooled-sid-cor10-kl8e4-eff32-s1
)
# Candidates = top-5 by VALID RECON LOSS (recovered from the training logs'
# "at step N: <loss>" lines, merged across self-resubmits), restricted to steps
# that actually have a saved ckpt, + 320000 as a like-for-like anchor.
# Why not a uniform grid: the loss curve is a free prior and it shows both models
# BOTTOM OUT near 240-260k and then DEGRADE — 320k is ~5% worse than best for both
# (maisi 0.014994 vs 0.014372 @240k; sid10 0.015376 vs 0.014971 @260k), and 320k
# makes neither top-5. A uniform grid would have missed both optima.
# NB: valid loss exists only at multiples of 20k (validation every 4k INTERSECT
# ckpt every 10k), so there are 16 measurable candidate steps per model.
# CAVEAT: valid recon loss is L1/L2-based and so rewards mean-reversion (the same
# blind spot that let vad1 post the best PSNR with SSIM 0.628) — it is used here
# ONLY to shortlist; the recon eval (which includes SSIM) does the actual picking.
declare -A CAND=(
  [maisi-kl8e4]="160000 200000 240000 260000 280000 320000"
  [sid-cor10-kl8e4]="180000 220000 240000 260000 280000 320000"
)
RUNS=()
for g in "${MODELS[@]}"; do for s in ${CAND[$g]}; do RUNS+=("${g}:${s}"); done; done

for r in "${RUNS[@]}"; do
  grp="${r%%:*}"; step="${r##*:}"
  src="${SRC_ROOT}/${SRC[$grp]}/weights/vae/checkpoint-${step}/model.pt"
  dst="${STAGE_OUT}/eval-${grp}-ck${step}/weights/vae/checkpoint-${step}"
  if [[ ! -f "${src}" ]]; then echo "[MISSING] ${src}"; continue; fi
  mkdir -p "${dst}"
  ln -f "${src}" "${dst}/model.pt" 2>/dev/null || cp "${src}" "${dst}/model.pt"
  echo "staged eval-${grp}-ck${step}  <- ${SRC[$grp]}/checkpoint-${step}"
done

echo
echo "Staged under: ${STAGE_OUT}"
echo "Transfer to GSDS (preserve tree):"
echo "  rsync -av ${STAGE_OUT}/  <gsds>:${SRC_ROOT}/"
