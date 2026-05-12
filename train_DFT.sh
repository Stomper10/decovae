#!/bin/bash
# Unified launcher for train_DFT.py (Decoder Fine-tuning).
# Switch experiments by editing the EXPERIMENT CONFIG block below.
# DFT runs are managed in their own experiment directories under
# OUTPUT_DIR_BASE, separate from VAE runs (see decodata/.../stage1/).
#
# === SBATCH options below are cluster-specific. Edit for your cluster, =====
# === or override at submission time (e.g. `sbatch -p mypart train_DFT.sh`). =
#SBATCH --job-name=train_DFT
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=P2
#SBATCH --exclude=b00,b06,b07,b08,b09,b12,b15,b24,b30
#SBATCH --time=0-12:00:00
#SBATCH --mem=200GB
#SBATCH --signal=B:SIGUSR1@180
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
# EXPERIMENT CONFIG — edit here, or override at sbatch time via --export
# ======================================================================
: "${EXP_NAME:=vad1e1_decft_noise1.0}"
: "${STAGE:=stage1}"   # DFT fine-tunes the VAE decoder from a given stage
: "${DATASET_CFG:=configs/ukb_20252/dataset.json}"
: "${MODEL_CFG:=configs/ukb_20252/model_fm.json}"
: "${TRAIN_CFG:=configs/ukb_20252/vae_decft_${STAGE}.json}"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT:-./outputs/ukb_20252}/${STAGE}}"
: "${PRETRAINED_PATH:=${OUTPUT_DIR_BASE}/vad1e1/weights/vae/checkpoint-40000/model.pt}"
: "${NOISE_SCALE:=1.0}"
: "${PLOSS_MODEL:=squeeze}"
: "${RESUME:=1}"   # 1 = pass --resume, 0 = fresh start

# ----------------------------------------------------------------------
# Experiment directory tree + log
# ----------------------------------------------------------------------
EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${LOGS_DIR}" "${EXP_DIR}/outputs" "${EXP_DIR}/weights/vae"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_dft_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

# ----------------------------------------------------------------------
# DDP rendezvous
# ----------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

echo "UKB - Starting Decoder Fine-tuning MULTI-NODE DDP training..."
echo "  job_name        : ${EXP_NAME}"
echo "  nodes           : ${SLURM_NNODES}  gpus_per_node : 4"
echo "  master          : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  dataset_cfg     : ${DATASET_CFG}"
echo "  model_cfg       : ${MODEL_CFG}"
echo "  train_cfg       : ${TRAIN_CFG}"
echo "  output_dir      : ${OUTPUT_DIR_BASE}"
echo "  exp_dir         : ${EXP_DIR}"
echo "  exp_log         : ${EXP_LOG}"
echo "  pretrained_path : ${PRETRAINED_PATH}"
echo "  noise_scale     : ${NOISE_SCALE}"
echo "  ploss_model     : ${PLOSS_MODEL}"
echo "  resume          : ${RESUME}"

# ----------------------------------------------------------------------
# Auto-requeue on SIGUSR1 (sent 180s before SLURM time limit)
# ----------------------------------------------------------------------
max_restarts=1000
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

function resubmit()
{
    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximum restarts limit"
        exit 1
    fi
}

trap 'resubmit' SIGUSR1

# ----------------------------------------------------------------------
# Compose CLI args
# ----------------------------------------------------------------------
RESUME_FLAG=""
if [[ "${RESUME}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

srun --cpu-bind=none,v --accel-bind=g torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_DFT.py \
      --dataset_config_path "${DATASET_CFG}" \
      --model_config_path "${MODEL_CFG}" \
      --train_config_path "${TRAIN_CFG}" \
      --output_dir "${OUTPUT_DIR_BASE}" \
      --run_name "${EXP_NAME}" \
      --cpus_per_task "${SLURM_CPUS_PER_TASK}" \
      --pretrained_model_path "${PRETRAINED_PATH}" \
      --latent_noise_scale "${NOISE_SCALE}" \
      --ploss_model "${PLOSS_MODEL}" \
      ${RESUME_FLAG} &
wait
exit 0
