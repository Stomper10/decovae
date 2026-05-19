#!/bin/bash
# Unified launcher for train_VAE.py.
# Switch experiments by editing the EXPERIMENT CONFIG block below — the four
# legacy variants reduce to one of these settings:
#
#   Base (no aux loss)  : LAMBDA_COV=0.0  LAMBDA_COR=0.0  LAMBDA_VAR=0.0
#   +Cov                : LAMBDA_COV>0    LAMBDA_COR=0.0  LAMBDA_VAR=0.0
#   +Cor (SID)          : LAMBDA_COV=0.0  LAMBDA_COR>0    LAMBDA_VAR=0.0
#   +Cov+Var (VAD)      : LAMBDA_COV>0    LAMBDA_COR=0.0  LAMBDA_VAR>0
#
# === SBATCH options below are cluster-specific. Edit for your cluster, =====
# === or override at submission time (e.g. `sbatch -p mypart train_VAE.sh`). =
#SBATCH --job-name=train_VAE
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
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
: "${EXP_NAME:=vad1e-0}"
: "${STAGE:=stage1}"     # switch stages: STAGE=stage2 sbatch train_VAE.sh
: "${DATASET:=ukb_20252}"  # switch dataset: DATASET=ixi sbatch train_VAE.sh
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"
: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"
: "${MODEL_CFG:=configs/${DATASET}/model_fm.json}"
: "${TRAIN_CFG:=configs/${DATASET}/vae_train_${STAGE}.json}"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT}/${STAGE}}"
: "${LAMBDA_COV:=1.0}"
: "${LAMBDA_COR:=0.0}"
: "${LAMBDA_VAR:=1.0}"
: "${TARGET_VAR:=1.0}"
: "${RESUME:=1}"   # 1 = pass --resume, 0 = fresh start

# ----------------------------------------------------------------------
# Experiment directory tree + log
# ----------------------------------------------------------------------
EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/embeddings" "${LOGS_DIR}" "${EXP_DIR}/outputs" "${EXP_DIR}/weights/vae"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_vae_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

# ----------------------------------------------------------------------
# DDP rendezvous
# ----------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

echo "UKB - Starting unified MAISI VAE MULTI-NODE DDP training..."
echo "  job_name   : ${EXP_NAME}"
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
echo "  nodes      : ${SLURM_NNODES}  gpus_per_node : ${NPROC_PER_NODE}"
echo "  master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  dataset_cfg: ${DATASET_CFG}"
echo "  model_cfg  : ${MODEL_CFG}"
echo "  train_cfg  : ${TRAIN_CFG}"
echo "  output_dir : ${OUTPUT_DIR_BASE}"
echo "  exp_dir    : ${EXP_DIR}"
echo "  exp_log    : ${EXP_LOG}"
echo "  lambdas    : cov=${LAMBDA_COV} cor=${LAMBDA_COR} var=${LAMBDA_VAR} (target_var=${TARGET_VAR})"
echo "  resume     : ${RESUME}"

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
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_VAE.py \
      --dataset_config_path "${DATASET_CFG}" \
      --model_config_path "${MODEL_CFG}" \
      --train_config_path "${TRAIN_CFG}" \
      --output_dir "${OUTPUT_DIR_BASE}" \
      --data_dir "${DATA_DIR}" \
      --train_label_dir "${TRAIN_CSV}" \
      --valid_label_dir "${VALID_CSV}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --run_name "${EXP_NAME}" \
      --cpus_per_task "${SLURM_CPUS_PER_TASK}" \
      --lambda_cov "${LAMBDA_COV}" \
      --lambda_cor "${LAMBDA_COR}" \
      --lambda_var "${LAMBDA_VAR}" \
      --target_var "${TARGET_VAR}" \
      ${RESUME_FLAG} &
wait
exit 0
