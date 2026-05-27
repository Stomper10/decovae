#!/bin/bash
# 3D-MedDiffusion reconstruction-metric eval, equivalent to compute_metric.sh
# real_vs_recon for MAISI/L_SID/L_VAD. Single 4-GPU job, two conda envs:
#   stage 1 (3d_meddiff): base_/recon_ volume generation (4-way GPU shard)
#   stage 2 (deco_v15)  : LPIPS/PSNR/SSIM with compute_metric's exact defs
#   stage 3 (deco_v15)  : rFID via compute_metric.py --phase fid (needs RadImageNet)
#
#SBATCH --job-name=3dmd_metric
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
[[ -f "${SCRIPT_DIR}/env.local.sh" ]] && source "${SCRIPT_DIR}/env.local.sh"

: "${EXP_NAME:=ukb_c4}"
: "${DATASET_CFG:=configs/ukb_20252/dataset.json}"
: "${BASE_CSV:=${UKB_VALID_CSV:-csv_files/valid.csv}}"
: "${DATA_DIR:=${UKB_DATA_DIR:-/data/wonyoungjang/20252_unzip}}"
: "${AE_CKPT:=/data/wonyoungjang/decodata/3d_meddiff/ukb_c4/my_model/version_2/checkpoints/epoch=1685-step=300000-train/recon_loss=0.15.ckpt}"
: "${NUM_IMAGES:=2500}"
: "${NUM_SHARDS:=4}"
# Intensity percentile for the base/recon volumes:
#   NORM_LOWER=0.5 -> paired SSIM/PSNR/LPIPS (3D MedDiff native, §8)
#   NORM_LOWER=0.0 -> rFID/gFID on the baseline-matched FID scale (compute_metric.sh)
# Use a distinct EXP_NAME (e.g. ukb_c4_lower0) when changing this so the two
# volume/feature/CSV sets don't clobber each other.
: "${NORM_LOWER:=0.5}"
: "${NORM_UPPER:=99.5}"

EVAL_EXP="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}/recon_eval"
VOL_DIR="${EVAL_EXP}/outputs/volumes"
LOGS_DIR="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}/logs"
mkdir -p "${VOL_DIR}" "${LOGS_DIR}"
exec >> "${LOGS_DIR}/3dmd_metric_${SLURM_JOB_ID:-$$}.log" 2>&1

echo "=== 3D MedDiff recon-metric eval (job ${SLURM_JOB_ID}) @ $(date) ==="
echo "  ae_ckpt   : ${AE_CKPT}"
echo "  base_csv  : ${BASE_CSV}  data_dir: ${DATA_DIR}"
echo "  num_images: ${NUM_IMAGES}   vol_dir: ${VOL_DIR}"

source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc

# -------- stage 1: base + recon generation (3d_meddiff env, 4-GPU shard) -----
conda activate 3d_meddiff
echo "=== stage 1: recon generation @ $(date) ==="
for ((s = 0; s < NUM_SHARDS; s++)); do
    CUDA_VISIBLE_DEVICES="${s}" python scripts/eval_3d_meddiff_recon.py \
        --dataset-config "${DATASET_CFG}" \
        --base-csv "${BASE_CSV}" \
        --data-dir "${DATA_DIR}" \
        --ae-ckpt "${AE_CKPT}" \
        --out-dir "${VOL_DIR}" \
        --num-images "${NUM_IMAGES}" \
        --norm-lower "${NORM_LOWER}" \
        --norm-upper "${NORM_UPPER}" \
        --shard-index "${s}" \
        --num-shards "${NUM_SHARDS}" &
done
wait
echo "  base/recon counts: base=$(ls ${VOL_DIR}/base_*.nii.gz 2>/dev/null | wc -l) recon=$(ls ${VOL_DIR}/recon_*.nii.gz 2>/dev/null | wc -l)"

# -------- stage 2: LPIPS / PSNR / SSIM (deco_v15 env) -------------------------
conda deactivate 2>/dev/null
conda activate deco_v15
echo "=== stage 2: LPIPS/PSNR/SSIM @ $(date) ==="
: "${RUN_NAME:=3d_meddiff}"
: "${QUALITY_CSV:=journal_plan/results_ukb_quality_3dmd.csv}"
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_recon_metrics \
    --volumes-dir "${VOL_DIR}" \
    --out-csv "${QUALITY_CSV}" \
    --run-name "${RUN_NAME}"

# -------- stage 3: rFID (needs RadImageNet weight) ---------------------------
echo "=== stage 3: rFID @ $(date) ==="
if [[ -n "${FEATURE_EXTRACTOR_PATH}" && -f "${FEATURE_EXTRACTOR_PATH}" ]]; then
    echo "  RadImageNet weight found -> computing rFID via compute_metric --phase fid"
    # compute_metric --phase fid reads filelist_{real,recon}_${NUM_IMAGES}.txt
    # (basenames relative to volumes/). The --phase generate step normally writes
    # these; we generate recon volumes ourselves in stage 1, so build them here.
    ( cd "${VOL_DIR}" && ls base_*.nii.gz  | sort > "${EVAL_EXP}/filelist_real_${NUM_IMAGES}.txt" \
                      && ls recon_*.nii.gz | sort > "${EVAL_EXP}/filelist_recon_${NUM_IMAGES}.txt" )
    echo "  filelists: real=$(wc -l < ${EVAL_EXP}/filelist_real_${NUM_IMAGES}.txt) recon=$(wc -l < ${EVAL_EXP}/filelist_recon_${NUM_IMAGES}.txt)"
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=$((10000 + RANDOM % 50000))
    torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d \
        --rdzv_endpoint=127.0.0.1:${MASTER_PORT} \
        compute_metric.py \
          --exp_dir "${EVAL_EXP}" \
          --dataset_config_path "${DATASET_CFG}" \
          --config_path "${DATASET_CFG}" \
          --eval_mode real_vs_recon \
          --phase fid \
          --num_images "${NUM_IMAGES}" \
          --base_label_dir "${BASE_CSV}" \
          --data_dir "${DATA_DIR}" \
          --feature_extractor_path "${FEATURE_EXTRACTOR_PATH}"
else
    echo "  [SKIP] FEATURE_EXTRACTOR_PATH not set / not found on AIBIO."
    echo "         Transfer RadImageNet-ResNet50_notop.pth from GSDS, set FEATURE_EXTRACTOR_PATH"
    echo "         in env.local.sh, and re-run stage 3 (volumes are already in ${VOL_DIR})."
fi
echo "=== DONE @ $(date) ==="
exit 0
