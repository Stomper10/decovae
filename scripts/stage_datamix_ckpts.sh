#!/bin/bash
# AIBIO-side: stage the datamix + pooled interim VAE checkpoints into the flat
# `eval-<grp>-ck<step>` layout that scripts/eval_datamix_cells.sh (GSDS) expects.
# Hardlinks model.pt within the same filesystem (no double-copy); falls back to cp.
#
# GRID (4 models x 6 steps = 24 ckpts):
#   grp        <- source experiment dir (SRC_ROOT/...)              cells measured
#   ukbT1      <- ukbT1-maisi-kl8e4-eff32-s1        (specialist)    ukb_T1
#   ukbT1flair <- ukbT1ukbFLAIR-maisi-kl8e4-eff32-s1 (cross-modal)  ukb_T1, ukb_FLAIR
#   ukbT1adni  <- ukbT1adniT1-maisi-kl8e4-eff32-s1  (cross-cohort)  ukb_T1, adni_T1
#   pooled     <- pooled-maisi-kl8e4-eff32-s1       (14-cell)       ukb_T1, ukb_FLAIR, adni_T1 (rFID only)
# Steps: 100000 120000 140000 160000 200000 240000
#
# Usage:  bash scripts/stage_datamix_ckpts.sh [STAGE_OUT_DIR]
# Then (your part): rsync -av <STAGE_OUT>/  <gsds>:<POOLED_OUTPUT_ROOT>/stage1/
set -euo pipefail

SRC_ROOT=/data/wonyoungjang/decodata/pooled/stage1
STAGE_OUT="${1:-${SRC_ROOT}/_xfer_datamix}"

declare -A SRC=(
  [ukbT1]=ukbT1-maisi-kl8e4-eff32-s1
  [ukbT1flair]=ukbT1ukbFLAIR-maisi-kl8e4-eff32-s1
  [ukbT1adni]=ukbT1adniT1-maisi-kl8e4-eff32-s1
  [pooled]=pooled-maisi-kl8e4-eff32-s1
)
MODELS=(ukbT1 ukbT1flair ukbT1adni pooled)
: "${STEPS:=100000 120000 140000 160000 200000 240000}"   # override to stage early ckpts, e.g. STEPS="10000 20000 ... 100000"
: "${MODELS_OVR:=}"; [[ -n "${MODELS_OVR}" ]] && MODELS=(${MODELS_OVR})

n_ok=0 n_miss=0
for grp in "${MODELS[@]}"; do
  for step in ${STEPS}; do
    src="${SRC_ROOT}/${SRC[$grp]}/weights/vae/checkpoint-${step}/model.pt"
    dst="${STAGE_OUT}/eval-${grp}-ck${step}/weights/vae/checkpoint-${step}"
    if [[ ! -f "${src}" ]]; then echo "[MISSING] ${src}"; n_miss=$((n_miss+1)); continue; fi
    mkdir -p "${dst}"
    ln -f "${src}" "${dst}/model.pt" 2>/dev/null || cp "${src}" "${dst}/model.pt"
    echo "staged eval-${grp}-ck${step}  <- ${SRC[$grp]}/checkpoint-${step}"
    n_ok=$((n_ok+1))
  done
done

echo
echo "Staged ${n_ok} ckpts (${n_miss} missing) under: ${STAGE_OUT}"
echo "Transfer to GSDS (preserve tree):"
echo "  rsync -av ${STAGE_OUT}/  <gsds>:\${POOLED_OUTPUT_ROOT}/stage1/"
