#!/bin/bash
# Phase 3 launcher: extract 3D-MedDiffusion PatchVolume latents for UKB c=4.
#
# Single-GPU inference job (encode-only). Override at submission time:
#   SPLIT=val sbatch generate_3d_meddiff_latents.sh
#   MAX_SAMPLES=16 sbatch generate_3d_meddiff_latents.sh   # smoke test
#
# gpu-4farm QoS (gpu4) enforces a per-job minimum of 4 GPUs (1-GPU requests sit
# in QOSMinGRES forever). We therefore grab a full 4-GPU node and shard the
# encode across all 4 GPUs (4x faster, no wasted allocation).
#SBATCH --job-name=3dmd_latent
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi

source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc
conda activate 3d_meddiff

: "${DATA_JSON:=configs/3d_meddiff/data_ukb.json}"
: "${SPLITS:=train val}"   # space-separated; processed sequentially in one job
: "${EXP_NAME:=ukb_c4}"
: "${AE_CKPT:=/data/wonyoungjang/decodata/3d_meddiff/ukb_c4/my_model/version_2/checkpoints/epoch=1685-step=300000-train/recon_loss=0.15.ckpt}"
: "${OUT_DIR:=/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}/latents}"

EXP_ROOT="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}"
LOGS_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOGS_DIR}"
EXP_LOG="${LOGS_DIR}/3dmd_latent_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

echo "3D MedDiff Phase 3 — latent extraction"
echo "  data_json : ${DATA_JSON}"
echo "  splits    : ${SPLITS}"
echo "  ae_ckpt   : ${AE_CKPT}"
echo "  out_dir   : ${OUT_DIR}"
echo "  job_id    : ${SLURM_JOB_ID}"
echo "  max_samp  : ${MAX_SAMPLES:-(all)}"

MAX_ARG=""
if [[ -n "${MAX_SAMPLES}" ]]; then
    MAX_ARG="--max-samples ${MAX_SAMPLES}"
fi

: "${NUM_SHARDS:=4}"   # one process per GPU on the 4-GPU allocation

# Process each split sequentially; within a split, shard across all 4 GPUs.
# CUDA_VISIBLE_DEVICES pins each shard process to its own GPU.
for split in ${SPLITS}; do
    echo "=== split ${split} @ $(date) ==="
    for ((s = 0; s < NUM_SHARDS; s++)); do
        CUDA_VISIBLE_DEVICES="${s}" python scripts/extract_3d_meddiff_latents.py \
            --data-json "${DATA_JSON}" \
            --split "${split}" \
            --ae-ckpt "${AE_CKPT}" \
            --out-dir "${OUT_DIR}" \
            --shard-index "${s}" \
            --num-shards "${NUM_SHARDS}" \
            ${MAX_ARG} &
    done
    wait
done
echo "=== ALL SPLITS DONE @ $(date) ==="
exit 0
