#!/bin/bash
# Brain-age regressor (SFCN) — downstream synthetic-aug eval entry point.
#
# Real-only (default):
#   sbatch train_brain_age.sh
#
# Real + synthetic mix (after stage2 weights produce synthetic volumes):
#   SYNTH_CSV=/path/to/synth_index.csv SYNTH_DIR=/path/to/synth_volumes \
#   EXP_NAME=real_plus_decoVAE sbatch train_brain_age.sh
#
# Data-regime ablation:
#   REAL_LIMIT=1000 EXP_NAME=real_only_n1000 sbatch train_brain_age.sh
#
#SBATCH --job-name=brain_age
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi

: "${EXP_NAME:=real_only}"
: "${DATASET:=ukb_20252}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"

: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"
: "${BRAIN_AGE_CFG:=configs/${DATASET}/brain_age.json}"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT}/downstream/brain_age}"
: "${SYNTH_CSV:=}"
: "${SYNTH_DIR:=}"
: "${REAL_LIMIT:=}"
: "${SYNTH_LIMIT:=}"

EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${LOGS_DIR}" "${EXP_DIR}/weights"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_brain_age_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

# Parse hyperparams from JSON (lightweight — avoids adding a dep on jq).
# Each value is env-overrideable: `EPOCHS=2 BS=2 sbatch train_brain_age.sh` etc.
read CFG_EPOCHS CFG_BS CFG_NW CFG_LR CFG_WD CFG_DROPOUT <<EOF
$(python -c "import json,sys;c=json.load(open('${BRAIN_AGE_CFG}'));print(c['epochs'],c['batch_size'],c['num_workers'],c['lr'],c['weight_decay'],c['dropout'])")
EOF
: "${EPOCHS:=${CFG_EPOCHS}}"
: "${BS:=${CFG_BS}}"
: "${NW:=${CFG_NW}}"
: "${LR:=${CFG_LR}}"
: "${WD:=${CFG_WD}}"
: "${DROPOUT:=${CFG_DROPOUT}}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

echo "Brain-age SFCN training"
echo "  exp_name      : ${EXP_NAME}"
echo "  dataset       : ${DATASET}"
echo "  dataset_cfg   : ${DATASET_CFG}"
echo "  brain_age_cfg : ${BRAIN_AGE_CFG}"
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

SYNTH_FLAGS=""
[[ -n "${SYNTH_CSV}" ]] && SYNTH_FLAGS="${SYNTH_FLAGS} --synthetic_csv ${SYNTH_CSV}"
[[ -n "${SYNTH_DIR}" ]] && SYNTH_FLAGS="${SYNTH_FLAGS} --synthetic_dir ${SYNTH_DIR}"
[[ -n "${REAL_LIMIT}" ]] && SYNTH_FLAGS="${SYNTH_FLAGS} --real_limit ${REAL_LIMIT}"
[[ -n "${SYNTH_LIMIT}" ]] && SYNTH_FLAGS="${SYNTH_FLAGS} --synth_limit ${SYNTH_LIMIT}"

srun --cpu-bind=none,v --accel-bind=g torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m downstream.train_brain_age \
      --dataset_config_path "${DATASET_CFG}" \
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
      --dropout "${DROPOUT}" \
      ${SYNTH_FLAGS} &
wait
exit 0
