#!/bin/bash
# TSTR (Train on Synthetic, Test on Real) — one training run per submission.
#
#   TASK=age SOURCE=synth_vad SEED=0 sbatch scripts/train_TSTR.sh
#
# TASK    age | sex | dx
# SOURCE  real_ctrl | synth_maisi | synth_sid | synth_vad
# SEED    training seed (init + data order). The training SET is fixed by
#         scripts/build_tstr_csvs.py and does not change with SEED.
#
# Every source of a task trains on a CSV of identical size and identical
# (cohort, modality, label) composition, selects best.pt on the REAL validation set,
# and is scored ONCE on the REAL test set (test_metrics.json). Validation is never
# the reported number: best.pt is its maximum over epochs.
#
# SBATCH resources below are AIBIO defaults; override them at submit time elsewhere.
#SBATCH --job-name=tstr
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
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
: "${DATASET:=pooled}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"     # -> DATA_DIR (real cache), OUTPUT_ROOT

: "${TASK:?set TASK (age|sex|dx)}"
: "${SOURCE:?set SOURCE (real_ctrl|synth_maisi|synth_sid|synth_vad)}"
: "${SEED:?set SEED}"
: "${EPOCHS:=15}"
: "${BS:=4}"
: "${NW:=4}"
: "${LR:=1e-3}"
: "${WD:=1e-4}"
: "${DROPOUT:=0.5}"
: "${TSTR_ROOT:=${OUTPUT_ROOT}/downstream/tstr}"
: "${CSV_DIR:=${TSTR_ROOT}/csv}"
: "${SYNTH_ROOT:=${OUTPUT_ROOT}/stage1}"
: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"

RUN_NAME="${SOURCE}_${TASK}_s${SEED}"
OUT_BASE="${TSTR_ROOT}/runs"
LOGS_DIR="${OUT_BASE}/${RUN_NAME}/logs"
mkdir -p "${LOGS_DIR}"
exec >> "${LOGS_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-local}.log" 2>&1

TRAIN_CSV="${CSV_DIR}/train_${SOURCE}_${TASK}.csv"
VALID_CSV="${CSV_DIR}/${TASK}_valid_4co.csv"
TEST_CSV="${CSV_DIR}/${TASK}_test_4co.csv"
for f in "${TRAIN_CSV}" "${VALID_CSV}" "${TEST_CSV}"; do
  [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}"; exit 1; }
done

# Synthetic rel_paths are relative to SYNTH_ROOT, real ones to DATA_DIR. Validation
# and test are always real.
if [[ "${SOURCE}" == real_ctrl ]]; then TRAIN_ROOT="${DATA_DIR}"; else TRAIN_ROOT="${SYNTH_ROOT}"; fi

case "${TASK}" in
  age) ENTRY=downstream.train_brain_age; TASK_FLAGS="" ;;
  sex) ENTRY=downstream.train_attr_predictor
       TASK_FLAGS="--target sex --label_map {\"M\":0,\"F\":1}" ;;
  dx)  ENTRY=downstream.train_attr_predictor
       TASK_FLAGS="--target dx --label_map {\"healthy\":0,\"MCI\":1,\"AD\":2} --class_weighted" ;;
  *)   echo "[FATAL] TASK=${TASK}"; exit 1 ;;
esac

# Entry points run as modules (-m downstream.X), like train_brain_age.sh: the
# scripts import `downstream.*`, which a bare file path does not put on sys.path.
source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc
conda activate "${CONDA_ENV:-deco_v15}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1
: "${NPROC_PER_NODE:=$(nvidia-smi -L 2>/dev/null | wc -l)}"

echo "TSTR ${RUN_NAME}   job=${SLURM_JOB_ID}   $(date)"
echo "  entry      : ${ENTRY}"
echo "  train      : ${TRAIN_CSV}  (root ${TRAIN_ROOT})"
echo "  valid/test : ${VALID_CSV} / ${TEST_CSV}  (root ${DATA_DIR})"
echo "  epochs=${EPOCHS} bs=${BS} lr=${LR} seed=${SEED} gpus=${NPROC_PER_NODE}"

# A requeue re-enters this script; the trainer resumes from weights/last.pt.
max_restarts=1000
restarts=$(scontrol show job ${SLURM_JOB_ID} 2>/dev/null | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
function resubmit() {
    if [[ ${restarts:-0} -lt ${max_restarts} ]]; then scontrol requeue ${SLURM_JOB_ID}; exit 0; fi
    exit 1
}
trap 'resubmit' SIGUSR1

srun --cpu-bind=none,v --accel-bind=g torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m ${ENTRY} \
    --dataset_config_path "${DATASET_CFG}" \
    --train_csv "${TRAIN_CSV}" \
    --valid_csv "${VALID_CSV}" \
    --test_csv "${TEST_CSV}" \
    --data_dir "${TRAIN_ROOT}" \
    --eval_data_dir "${DATA_DIR}" \
    --output_dir "${OUT_BASE}" \
    --run_name "${RUN_NAME}" \
    --seed "${SEED}" \
    --batch_size "${BS}" --num_workers "${NW}" --epochs "${EPOCHS}" \
    --lr "${LR}" --weight_decay "${WD}" --dropout "${DROPOUT}" \
    ${TASK_FLAGS} &
wait
exit 0
