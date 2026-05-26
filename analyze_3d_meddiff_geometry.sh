#!/bin/bash
# Run the 3D MedDiff latent geometry/stat REFERENCE analysis. Quick (~2 min),
# but gpu-4farm QoS forces a 4-GPU minimum, so we grab 4 and use one.
#
#SBATCH --job-name=3dmd_geom
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
[[ -f "${SCRIPT_DIR}/env.local.sh" ]] && source "${SCRIPT_DIR}/env.local.sh"
source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc
conda activate 3d_meddiff

: "${EXP_NAME:=ukb_c4}"
: "${LATENT_DIR:=/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}/latents/train}"
: "${MAX_SAMPLES:=5000}"
: "${OUT_GEOM:=journal_plan/results_ukb_geometry_3dmd.csv}"
: "${OUT_STAT:=journal_plan/results_ukb_stats_3dmd.csv}"

LOGS_DIR="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}/logs"
mkdir -p "${LOGS_DIR}"
exec >> "${LOGS_DIR}/3dmd_geom_${SLURM_JOB_ID}.log" 2>&1

echo "3D MedDiff latent geometry/stat (reference) — job ${SLURM_JOB_ID}"
echo "  latent_dir : ${LATENT_DIR}"
echo "  max_samples: ${MAX_SAMPLES}"

CUDA_VISIBLE_DEVICES=0 python scripts/analyze_3d_meddiff_latent_geometry.py \
    --latent-dir "${LATENT_DIR}" \
    --run-name 3d_meddiff \
    --max-samples "${MAX_SAMPLES}" \
    --out-geometry "${OUT_GEOM}" \
    --out-stat "${OUT_STAT}"
echo "=== GEOM DONE @ $(date) ==="
exit 0
