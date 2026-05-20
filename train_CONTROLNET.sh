#!/bin/bash
# Launcher for train_CONTROLNET.py (BraTS mask-conditional latent diffusion).
#
# Trains a ControlNetMaisi on top of a *frozen* base UNet that was previously
# trained by train_UNET.sh. The base UNet checkpoint path is the **only**
# extra knob compared to train_UNET.sh — everything else mirrors the
# train_UNET.sh contract so launches feel identical.
#
# Typical invocation (after extract_emb.py has populated embeddings/ and the
# base UNet has been trained):
#
#   DATASET=brats EXP_NAME=brats_controlnet \
#   TRAINED_DIFFUSION_PATH=/.../weights/unet/best-checkpoint/model.pt \
#       sbatch train_CONTROLNET.sh
#
#SBATCH --job-name=train_CONTROLNET
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
else
    echo "[WARN] env.local.sh not found at ${SCRIPT_DIR}/env.local.sh — running with system defaults." >&2
fi

# ======================================================================
# EXPERIMENT CONFIG
# ======================================================================
: "${EXP_NAME:=brats_controlnet}"
: "${STAGE:=stage1}"
: "${DATASET:=brats}"  # default BraTS; mask-conditional is BraTS-only for now
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"
: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"
: "${MODEL_CFG:=configs/${DATASET}/model_fm.json}"
: "${TRAIN_CFG:=configs/${DATASET}/controlnet_train.json}"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT}/${STAGE}}"
: "${RESUME:=1}"

# Path to the *trained* base UNet that ControlNet will freeze and condition.
# Required — script aborts inside python if unset/missing.
: "${TRAINED_DIFFUSION_PATH:=}"

# ----------------------------------------------------------------------
# Experiment directory tree + log
# ----------------------------------------------------------------------
EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/analysis" "${EXP_DIR}/embeddings" "${LOGS_DIR}" "${EXP_DIR}/outputs" "${EXP_DIR}/weights/controlnet"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_controlnet_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

# ----------------------------------------------------------------------
# DDP rendezvous
# ----------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

echo "BraTS - Starting ControlNet (mask-conditional) training..."
echo "  exp_name             : ${EXP_NAME}"
echo "  dataset              : ${DATASET}"
echo "  nodes                : ${SLURM_NNODES}  gpus_per_node : ${NPROC_PER_NODE}"
echo "  master               : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  dataset_cfg          : ${DATASET_CFG}"
echo "  model_cfg            : ${MODEL_CFG}"
echo "  train_cfg            : ${TRAIN_CFG}"
echo "  trained_diffusion_path: ${TRAINED_DIFFUSION_PATH}"
echo "  output_dir           : ${OUTPUT_DIR_BASE}"
echo "  exp_dir              : ${EXP_DIR}"
echo "  exp_log              : ${EXP_LOG}"
echo "  resume               : ${RESUME}"

if [[ -z "${TRAINED_DIFFUSION_PATH}" ]]; then
    echo "[FATAL] TRAINED_DIFFUSION_PATH unset. Set it to the base UNet checkpoint path:" >&2
    echo "        TRAINED_DIFFUSION_PATH=/.../weights/unet/best-checkpoint/model.pt sbatch ..." >&2
    exit 2
fi

# ----------------------------------------------------------------------
# Auto-requeue on SIGUSR1
# ----------------------------------------------------------------------
max_restarts=1000
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

function resubmit() {
    if [[ ${restarts} -lt ${max_restarts} ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    fi
    echo "Restart limit reached"
    exit 1
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
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_CONTROLNET.py \
      --dataset_config_path "${DATASET_CFG}" \
      --model_config_path "${MODEL_CFG}" \
      --train_config_path "${TRAIN_CFG}" \
      --output_dir "${OUTPUT_DIR_BASE}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --run_name "${EXP_NAME}" \
      --cpus_per_task "${SLURM_CPUS_PER_TASK}" \
      --trained_diffusion_path "${TRAINED_DIFFUSION_PATH}" \
      ${RESUME_FLAG} &
wait
exit 0
