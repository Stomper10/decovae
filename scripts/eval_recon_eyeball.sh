#!/bin/bash
# Qualitative recon A/B across models at SPECIFIC checkpoints (AIBIO or GSDS).
#
# Why this exists: recon images are written as {i:04d}_recon_xyz.png /
# recon_{i:04d}.nii.gz with no checkpoint in the name, so the trajectory sweep
# (11 checkpoints -> one cells/ukb_T1_test dir per arm) left only whichever
# checkpoint's job happened to finish LAST. Those PNGs cannot be attributed to a
# checkpoint. This driver gives every (arm, step) its own output dir via OUT_TAG.
#
# PHASE=generate only: writes base/recon PNGs + volumes and the paired metrics.
# No FID -- the eyeball A/B does not need it and the r=1.0 default is excluded.
#
# Subject i is the same volume in every arm (deterministic per-cell filelist +
# DETERMINISTIC=1), so the columns of the comparison grid are apples-to-apples.
#
# Usage:
#   DRYRUN=1 bash scripts/eval_recon_eyeball.sh
#   bash scripts/eval_recon_eyeball.sh
#   PAIRS="pooled-maisi-kl8e4-eff32-s1:240000 pooled-vad-cov1var1-kl8e4-eff32-s1:320000" \
#     NUM_IMAGES=32 bash scripts/eval_recon_eyeball.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CM="${SCRIPT_DIR}/compute_metric.sh"
STAGE_ROOT="${STAGE_ROOT:-${POOLED_OUTPUT_ROOT:?set POOLED_OUTPUT_ROOT or STAGE_ROOT}/stage1}"

# arm:step -- the two plateau representatives we want to compare by eye.
PAIRS="${PAIRS:-pooled-maisi-kl8e4-eff32-s1:240000 pooled-vad-cov1var1-kl8e4-eff32-s1:320000}"
CELL="${CELL:-ukb_T1}"
SPLIT="${SPLIT:-test}"
# 16 subjects is plenty to eyeball blur; the paired metrics this prints are on
# n=16 and are NOT comparable to the n=500 numbers in results_triad_trajectory.csv.
NUM_IMAGES="${NUM_IMAGES:-16}"
DRYRUN="${DRYRUN:-0}"

# AIBIO normal QoS allows 1 concurrent GPU job, so chain instead of fanning out.
CHAIN="${CHAIN:-1}"

dep=""; n=0
declare -a DIRS=()
for pair in ${PAIRS}; do
    arm="${pair%%:*}"; step="${pair##*:}"
    ck="${STAGE_ROOT}/${arm}/weights/vae/checkpoint-${step}/model.pt"
    if [[ ! -f "${ck}" ]]; then echo "[skip] missing ${ck}"; continue; fi
    tag="ck$((step/1000))k"
    DIRS+=("${arm}|${tag}")
    env_args=(
      EXP_NAME="${arm}" STAGE=stage1 DATASET=pooled
      VAE_CKPT_NAME="checkpoint-${step}"
      EVAL_MODE=real_vs_recon PHASE=generate DETERMINISTIC=1
      NUM_IMAGES="${NUM_IMAGES}" CELL="${CELL}" SPLIT="${SPLIT}"
      OUT_TAG="${tag}" POSTFIX="eye_${tag}"
      SSIM_FG=1 SSIM_FG_ERODE=3
    )
    if [[ "${DRYRUN}" == "1" ]]; then
        echo "[dryrun] ${dep:+$dep }env ${env_args[*]} sbatch ${CM}"
        jid=$(( RANDOM + 100000 ))
    else
        jid=$(env "${env_args[@]}" sbatch --parsable ${dep:+$dep} "${CM}")
    fi
    echo "[job] ${arm} ck${step}  ->  cells/${CELL}_${SPLIT}_${tag}   jid=${jid}"
    n=$((n+1))
    [[ "${CHAIN}" == "1" ]] && dep="--dependency=afterany:${jid}"
done

echo
echo "submitted=${n}  (DRYRUN=${DRYRUN}, CHAIN=${CHAIN})"
[[ ${n} -eq 0 ]] && exit 0

echo
echo "When both finish, build the comparison grid (no GPU, reads PNGs only):"
echo
printf '  python scripts/recon_compare.py \\\n'
printf '    --out recon_cmp_%s_maisi240k_vs_vad320k.png --subjects 0 1 2 3 \\\n' "${CELL}"
for d in "${DIRS[@]}"; do
    arm="${d%%|*}"; tag="${d##*|}"
    label="$(sed -e 's/^pooled-//' -e 's/-kl8e4-eff32-s1$//' <<<"${arm}")-${tag}"
    printf '    --model "%s:%s/%s/cells/%s_%s_%s/outputs/slices" \\\n' \
        "${label}" "${STAGE_ROOT}" "${arm}" "${CELL}" "${SPLIT}" "${tag}"
done
echo
echo "(the 'real' column is taken from the first --model's *_base_xyz.png;"
echo " it is identical across arms because the per-cell filelist is deterministic)"
