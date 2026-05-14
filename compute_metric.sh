#!/bin/bash
# Unified metric launcher: volume generation + 2.5D FID under torchrun DDP.
# Switch experiments / modes by editing the EXPERIMENT CONFIG block below, or
# override at sbatch time via env vars (e.g. `EXP_NAME=vad1e1 EVAL_MODE=real_vs_recon sbatch compute_metric.sh`).
#
# === SBATCH options below are cluster-specific. Edit for your cluster, =====
# === or override at submission time. ========================================
#SBATCH --job-name=compute_metric
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=04:00:00
#SBATCH -o /dev/null
#SBATCH --open-mode=append

# ----------------------------------------------------------------------
# Per-user environment (conda activate, TMPDIR, default paths, ...)
# See env.local.example.sh for a template. env.local.sh is gitignored.
#
# Under SLURM the script is copied to a spool dir before execution, so
# ${BASH_SOURCE[0]} no longer points at the repo. We try $SLURM_SUBMIT_DIR
# first (the directory you invoked `sbatch` from), then fall back to the
# script's apparent directory for direct shell execution.
# ----------------------------------------------------------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
else
    echo "[WARN] env.local.sh not found at ${SCRIPT_DIR}/env.local.sh — running with system defaults." >&2
fi

# ======================================================================
# EXPERIMENT CONFIG — edit here, or override at sbatch time via env vars
# ======================================================================
: "${EXP_NAME:=vad1e1}"
: "${STAGE:=stage1}"                 # VAE stage that this experiment came from
: "${EVAL_MODE:=real_vs_recon}"        # real_vs_real | real_vs_recon | real_vs_gen
: "${PHASE:=all}"                    # generate | fid | all
: "${NUM_IMAGES:=2500}"
: "${POSTFIX:=30step}"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT:-./outputs/ukb_20252}/${STAGE}}"
: "${VAE_CKPT_NAME:=checkpoint-40000}"
: "${UNET_CKPT_NAME:=checkpoint-10000}"
: "${BASE_CSV:=${VALID_CSV:-./data/valid.csv}}"
: "${OTHER_CSV:=${TRAIN_CSV:-./data/train.csv}}"
: "${DATASET_CFG:=configs/ukb_20252/dataset.json}"
: "${NUM_GPUS:=4}"

# ----------------------------------------------------------------------
# Experiment directory tree + log
# ----------------------------------------------------------------------
EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${LOGS_DIR}" \
         "${EXP_DIR}/outputs/volumes" \
         "${EXP_DIR}/outputs/slices" \
         "${EXP_DIR}/outputs/features"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_metric_${SLURM_JOB_ID:-$$}.log"
exec >> "${EXP_LOG}" 2>&1

# ----------------------------------------------------------------------
# DDP rendezvous
# ----------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

echo "$(date +"%Y-%m-%d %H-%M-%S") :: compute_metric"
echo "  EXP_NAME   : ${EXP_NAME}"
echo "  EVAL_MODE  : ${EVAL_MODE}"
echo "  PHASE      : ${PHASE}"
echo "  NUM_IMAGES : ${NUM_IMAGES}"
echo "  POSTFIX    : ${POSTFIX}"
echo "  EXP_DIR    : ${EXP_DIR}"
echo "  EXP_LOG    : ${EXP_LOG}"
echo "  master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  num_gpus   : ${NUM_GPUS}"

# ----------------------------------------------------------------------
# Compose CLI args
# ----------------------------------------------------------------------
VAE_PATH="${EXP_DIR}/weights/vae/${VAE_CKPT_NAME}"
UNET_PATH="${EXP_DIR}/weights/unet/${UNET_CKPT_NAME}"
CONFIG_PATH="${EXP_DIR}/weights/unet/config.json"

srun --cpu-bind=none,v --accel-bind=g torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    compute_metric.py \
      --exp_dir "${EXP_DIR}" \
      --dataset_config_path "${DATASET_CFG}" \
      --config_path "${CONFIG_PATH}" \
      --pretrained_vae_path "${VAE_PATH}" \
      --pretrained_unet_path "${UNET_PATH}" \
      --eval_mode "${EVAL_MODE}" \
      --phase "${PHASE}" \
      --num_images "${NUM_IMAGES}" \
      --postfix "${POSTFIX}" \
      --base_label_dir "${BASE_CSV}" \
      --other_label_dir "${OTHER_CSV}" \
      --data_dir "${DATA_DIR}" \
      --feature_extractor_path "${FEATURE_EXTRACTOR_PATH}" \
      --save_volume \
      --save_real &
wait
exit 0
