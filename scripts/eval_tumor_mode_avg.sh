#!/bin/bash
# §3b — SLURM array wrapper for eval_tumor_mode_avg.py.
# One task = one volume source (real or one gen cell).
#
# Manifest (TSV): label  vol_src  gen_dir_or_-  max_cases
#   label      short id used in the output CSV filename
#   vol_src    'real' or 'gen'
#   gen_dir    for gen: <cell>/outputs/volumes ; for real: '-'
#   max_cases  0 = all; else clamp
#
# Usage:
#   sbatch --array=0-9%5 --export=ALL,MANIFEST=/path/to/tumor_mode_manifest.tsv \
#     scripts/eval_tumor_mode_avg.sh
#
#SBATCH --job-name=tumor_mode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=P2
#SBATCH --exclude=b00,b06,b07,b08,b10,b12,b13,b14,b15,b16,b17,b18,b24,b26,b28,b31
#SBATCH --time=0-04:00:00
#SBATCH --mem=64GB
#SBATCH --signal=B:SIGUSR1@30
#SBATCH -o /leelabsg/data/wonyoungjang/decovae/journal_plan/tumor_mode_%A_%a.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

: "${MANIFEST:?set MANIFEST}"
: "${SLURM_ARRAY_TASK_ID:?must run as SLURM array}"

LINE=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' "${MANIFEST}")
[[ -z "${LINE}" ]] && { echo "no manifest line for TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }

IFS=$'\t' read -r LABEL VOL_SRC GEN_DIR MAX_CASES <<<"${LINE}"

: "${BRATS_ROOT:=/leelabsg/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train}"
: "${SEG_CKPT:=/leelabsg/data/wonyoungjang/decodata/brats/downstream/tumor_seg/real_only_FLAIR/weights/best.pt}"
OUT="${SCRIPT_DIR}/journal_plan/tumor_mode/${LABEL}.csv"

echo "[array $(date +%H:%M:%S)] TASK=${SLURM_ARRAY_TASK_ID}"
echo "  LABEL    : ${LABEL}"
echo "  VOL_SRC  : ${VOL_SRC}"
echo "  GEN_DIR  : ${GEN_DIR}"
echo "  MAX_CASES: ${MAX_CASES}"
echo "  OUT      : ${OUT}"

ARGS=(--vol_src "${VOL_SRC}" --seg_ckpt "${SEG_CKPT}" --out "${OUT}" --max_cases "${MAX_CASES}")
if [[ "${VOL_SRC}" == "real" ]]; then
  ARGS+=(--brats_root "${BRATS_ROOT}")
else
  ARGS+=(--gen_dir "${GEN_DIR}")
fi

python3 "${SCRIPT_DIR}/scripts/eval_tumor_mode_avg.py" "${ARGS[@]}"
