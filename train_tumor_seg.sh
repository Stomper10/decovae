#!/bin/bash
# Tumor segmentor (SegResNet) — BraTS downstream synthetic-aug eval.
#
# Real-only (default):
#   sbatch train_tumor_seg.sh
#
# Real + synthetic mix (after stage2 mask-conditional VAE+UNet produces volumes):
#   SYNTH_CSV=/path/to/synth_seg_index.csv SYNTH_DIR=/path/to/synth_volumes \
#   EXP_NAME=real_plus_decoVAE sbatch train_tumor_seg.sh
#
# Data-regime ablation:
#   REAL_LIMIT=100 EXP_NAME=real_only_n100 sbatch train_tumor_seg.sh
#
#SBATCH --job-name=tumor_seg
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
fi

: "${EXP_NAME:=real_only}"
: "${DATASET:=brats}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"

: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"
: "${SEG_CFG:=configs/${DATASET}/tumor_seg.json}"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT}/downstream/tumor_seg}"
: "${SYNTH_CSV:=}"
: "${SYNTH_DIR:=}"
: "${REAL_LIMIT:=}"
: "${SYNTH_LIMIT:=}"

EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${LOGS_DIR}" "${EXP_DIR}/weights"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_tumor_seg_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

read EPOCHS BS NW LR WD <<EOF
$(python -c "import json,sys;c=json.load(open('${SEG_CFG}'));print(c['epochs'],c['batch_size'],c['num_workers'],c['lr'],c['weight_decay'])")
EOF

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

echo "Tumor-seg SegResNet training"
echo "  exp_name      : ${EXP_NAME}"
echo "  dataset       : ${DATASET}"
echo "  dataset_cfg   : ${DATASET_CFG}"
echo "  seg_cfg       : ${SEG_CFG}"
echo "  output_dir    : ${EXP_DIR}"
echo "  real_csv      : ${TRAIN_CSV}"
echo "  synth_csv     : ${SYNTH_CSV:-(none — real-only)}"
echo "  real_limit    : ${REAL_LIMIT:-(none)}"
echo "  gpus / nproc  : ${NPROC_PER_NODE}"

max_restarts=1000
restarts=$(scontrol show job ${SLURM_JOB_ID} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
function resubmit() {
    if [[ ${restarts} -lt ${max_restarts} ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    fi
    exit 1
}
trap 'resubmit' SIGUSR1

EXTRA_FLAGS=""
[[ -n "${SYNTH_CSV}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --synthetic_csv ${SYNTH_CSV}"
[[ -n "${SYNTH_DIR}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --synthetic_dir ${SYNTH_DIR}"
[[ -n "${REAL_LIMIT}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --real_limit ${REAL_LIMIT}"
[[ -n "${SYNTH_LIMIT}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --synth_limit ${SYNTH_LIMIT}"

srun --cpu-bind=none,v --accel-bind=g torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m downstream.train_tumor_seg \
      --dataset_config_path "${DATASET_CFG}" \
      --seg_config_path "${SEG_CFG}" \
      --train_csv "${TRAIN_CSV}" \
      --valid_csv "${VALID_CSV}" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${OUTPUT_DIR_BASE}" \
      --run_name "${EXP_NAME}" \
      --batch_size "${BS}" \
      --num_workers "${NW}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --weight_decay "${WD}" \
      ${EXTRA_FLAGS} &
wait
exit 0
