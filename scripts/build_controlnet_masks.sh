#!/bin/bash
# BraTS tumour masks in the pooled preprocessing space (scripts/build_controlnet_masks.py).
# CPU-only, like preprocess_cache.sh -> does not touch the GPU job cap. Resumable
# (skips existing masks) and shardable (--array + NSHARDS).
#
#   sbatch scripts/build_controlnet_masks.sh
#   LIMIT=4 WORKERS=4 bash scripts/build_controlnet_masks.sh     # local smoke test
#
#SBATCH --job-name=cn_masks
#SBATCH --partition=cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH -o cn_masks_%A_%a.log
#SBATCH --open-mode=append
set -eo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# ants lives in the `deco` env, not the training env (same as preprocess_cache.sh).
source /opt/ohpc/pub/anaconda3/bin/activate 2>/dev/null || source ~/miniconda3/bin/activate
conda activate deco

OUT_ROOT="${OUT_ROOT:-/data/wonyoungjang/decodata/pooled/controlnet_masks}"
CACHE_ROOT="${CACHE_ROOT:-/data/wonyoungjang/decovae_cache}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export PP_THREADS="${PP_THREADS:-4}"
WORKERS="${WORKERS:-$(( CPUS / PP_THREADS ))}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
LIMIT="${LIMIT:-0}"

cd "${SCRIPT_DIR}"
echo "[cn_masks] out=${OUT_ROOT} workers=${WORKERS}x${PP_THREADS} shard=${SHARD}/${NSHARDS} limit=${LIMIT}  $(date)"
PYTHONPATH="${SCRIPT_DIR}/scripts:${PYTHONPATH}" python3 scripts/build_controlnet_masks.py \
    --cache_root "${CACHE_ROOT}" --out_root "${OUT_ROOT}" --workers "${WORKERS}" \
    --shard "${SHARD}" --nshards "${NSHARDS}" --limit "${LIMIT}"
