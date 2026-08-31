#!/bin/bash
# GSDS-side driver: interim-trajectory latent extraction + analysis + recon metric.
#
# For each staged checkpoint (GROUPS x STEPS, see grid below) it submits TWO
# independent sbatch jobs:
#   (1) extract_emb.sh   STAGES=extract,geometry,stat  -> embeddings + latent_geometry.csv
#                                                         + latent_stats.csv (scale_factor)
#   (2) compute_metric.sh EVAL_MODE=real_vs_recon DETERMINISTIC=1 -> rFID + SSIM (z=z_mu)
# = 2 jobs per ckpt (24 for the 12-ckpt kl8e4 grid); SLURM throttles to GSDS's
# 4-job cap automatically. Note recon here is whole-valid-set (NUM_IMAGES=2500)
# -> ONE rFID + SSIM per ckpt, not the 14-cell breakdown; that is deliberate,
# this is a screen. Run the full per-cell eval only on the final pick. The two
# pipelines are independent (recon does not consume extract's output), so no
# ordering barrier is needed.
#
# Prereqs on GSDS:
#   - checkpoints staged at $STAGE_ROOT/eval-<grp>-ck<step>/weights/vae/checkpoint-<step>/model.pt
#     (see scripts/stage_interim_ckpts.sh on AIBIO + rsync)
#   - pooled VALID-manifest .npy cache present (~6493 vols; NOT the full 862GB train cache)
#   - RadImageNet feature extractor (FEATURE_EXTRACTOR_PATH in env.local.sh)
#   - MONAI >= 1.4 is sufficient (real_vs_recon + extract_emb don't need 1.5)
#
# Usage:  bash scripts/eval_interim_trajectory.sh
# Override: EXTRACT_CSV=<csv> NUM_IMAGES=<n> bash scripts/eval_interim_trajectory.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/env.local.sh"

export DATASET=pooled
export STAGE=stage1
STAGE_ROOT="${POOLED_OUTPUT_ROOT}/stage1"

# Extraction source: the 6493-vol VALID manifest (bounded; > geometry's 5000 cap,
# robust scale_factor). Override EXTRACT_CSV to use a different / larger set.
: "${EXTRACT_CSV:=${POOLED_VALID_CSV}}"
: "${NUM_IMAGES:=2500}"   # recon FID sample count from the valid set

# Tiny valid stub so extract_emb's "validation" pass is trivial — the real
# extraction + geometry + stat all key off train_label_dir (= EXTRACT_CSV).
TINY_VALID="${STAGE_ROOT}/_tiny_valid.csv"
mkdir -p "${STAGE_ROOT}"
head -n 5 "${EXTRACT_CSV}" > "${TINY_VALID}"

# --- grid: MUST match scripts/stage_interim_ckpts.sh (MODELS/CAND) --------------
# NB: do NOT name this array GROUPS — bash builtin (caller's Unix GIDs).
# Candidates = top-5 by valid recon loss + 320k anchor; see stage_interim_ckpts.sh.
MODELS=(maisi-kl8e4 sid-cor10-kl8e4)
declare -A CAND=(
  [maisi-kl8e4]="160000 200000 240000 260000 280000 320000"
  [sid-cor10-kl8e4]="180000 220000 240000 260000 280000 320000"
)
RUNS=()
for g in "${MODELS[@]}"; do for s in ${CAND[$g]}; do RUNS+=("${g}:${s}"); done; done

# Stage gating. Default = recon ONLY: it is the fast, decisive signal (SSIM/LPIPS/
# rFID over 2500 valid vols), whereas extract_emb encodes the full 6493-vol valid
# manifest AND writes embeddings to disk = much slower. The band gate (scale_factor)
# is NOT skipped, just DEFERRED: rerun with DO_EXTRACT=1 on the 1-2 finalists only.
: "${DO_RECON:=1}"
: "${DO_EXTRACT:=0}"

for r in "${RUNS[@]}"; do
  grp="${r%%:*}"; step="${r##*:}"
  exp="eval-${grp}-ck${step}"
  ck="${STAGE_ROOT}/${exp}/weights/vae/checkpoint-${step}/model.pt"
  if [[ ! -f "${ck}" ]]; then echo "[skip] missing ${ck}"; continue; fi

  what=""; [[ "${DO_RECON}" == 1 ]] && what+="recon "; [[ "${DO_EXTRACT}" == 1 ]] && what+="extract"
  echo "[submit] ${exp}  (${what:-nothing})"

  # (1) extract + geometry + stat over the valid manifest  [deferred by default]
  if [[ "${DO_EXTRACT}" == 1 ]]; then
    EXP_NAME="${exp}" \
    WORK_DIR="${STAGE_ROOT}/${exp}" \
    CKPT="${ck}" \
    TRAIN_CSV="${EXTRACT_CSV}" \
    VALID_CSV="${TINY_VALID}" \
    STAGES="extract,geometry,stat" \
      sbatch "${SCRIPT_DIR}/extract_emb.sh"
  fi

  # (2) recon metric: rFID + SSIM, deterministic decode (z=z_mu)
  if [[ "${DO_RECON}" == 1 ]]; then
    EXP_NAME="${exp}" \
    VAE_CKPT_NAME="checkpoint-${step}" \
    EVAL_MODE="real_vs_recon" \
    DETERMINISTIC=1 \
    NUM_IMAGES="${NUM_IMAGES}" \
      sbatch "${SCRIPT_DIR}/compute_metric.sh"
  fi
done

echo
echo "Submitted. Watch: squeue -u \$USER ; tail logs under ${STAGE_ROOT}/eval-*/logs/"
echo "Results:"
echo "  scale_factor / latent stats : ${STAGE_ROOT}/eval-*/analysis/latent_stats.csv"
echo "  latent geometry             : ${STAGE_ROOT}/eval-*/analysis/latent_geometry.csv"
echo "  rFID / SSIM                  : ${STAGE_ROOT}/eval-*/outputs/  (+ stdout in logs/)"
