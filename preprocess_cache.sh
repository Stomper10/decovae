#!/bin/bash
# Offline preprocessing cache for the pooled corpus (scripts/preprocess_cache.py).
# CPU-only multiprocessing job (antspynet forced to CPU) -> does NOT consume the
# 1-GPU-per-user cap. Resumable (skips existing .npy) and shardable.
#
# === SBATCH options are cluster-specific. Edit partition/account for your site,
# === or override at submit time. This is a high-core CPU job, NOT a GPU job.
#SBATCH --job-name=preproc_cache
#SBATCH --partition=cpu-standard     # node01-20, 112 CPU / 503G each (7-day)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112          # full idle node (node12/13/15-20 are 0/112) → no contention
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH -o preproc_cache_%A_%a.log
#SBATCH --open-mode=append
# Optional: shard across array tasks (e.g. `sbatch --array=0-3 preprocess_cache.sh`)
# with NSHARDS matching the array width. Default (no array) = single shard.

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# preprocessing needs the `deco` env (ants/antspynet/monai) — NOT env.local.sh's
# training env. Activate via the system anaconda (compute nodes lack the
# ~/.conda/etc conda.sh path); deco lives in ~/.conda/envs and is discoverable.
source /opt/ohpc/pub/anaconda3/bin/activate
conda activate deco

# --- knobs (override via env at submit time) ---------------------------------
SPLIT="${SPLIT:-train}"                               # train | valid | test
CACHE_ROOT="${CACHE_ROOT:-/data/wonyoungjang/decovae_cache}"
CPUS="${SLURM_CPUS_PER_TASK:-112}"
export PP_THREADS="${PP_THREADS:-4}"                  # threads per worker (antspynet strip is multi-threaded)
WORKERS="${WORKERS:-$(( CPUS / PP_THREADS ))}"        # workers * PP_THREADS ~= cpus
REG="${REG:-rigid}"                                   # rigid (decided) | crop
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
LIMIT="${LIMIT:-0}"                                   # 0 = all; >0 = first N (validation)

echo "[preproc] split=${SPLIT} cache=${CACHE_ROOT} workers=${WORKERS}x${PP_THREADS}thr reg=${REG} shard=${SHARD}/${NSHARDS} limit=${LIMIT}"
cd "${SCRIPT_DIR}"
python3 scripts/preprocess_cache.py \
    --split "${SPLIT}" \
    --cache_root "${CACHE_ROOT}" \
    --workers "${WORKERS}" \
    --reg "${REG}" \
    --shard "${SHARD}" \
    --nshards "${NSHARDS}" \
    --limit "${LIMIT}"

# Examples:
#   SPLIT=valid sbatch preprocess_cache.sh
#   SPLIT=train sbatch --array=0-3 preprocess_cache.sh      # 4-way shard
#   # local smoke test (no SLURM):
#   SPLIT=valid WORKERS=8 CACHE_ROOT=/tmp/cache bash preprocess_cache.sh
