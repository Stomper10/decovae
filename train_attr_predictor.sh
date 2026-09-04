#!/bin/bash
# SFCN attribute classifier (adherence predictor) — modality / sex / dx.
#
# modality is the PRIMARY adherence axis: it is present on 100% of volumes and is the
# one attribute CFG never drops (cfg_drop_presence keep_idx=(0,)), so it is the only
# condition the model always actually received. T1c is excluded — it is vae_only and
# outside the diffusion modality vocabulary, so the generator could never emit it.
#   DATASET=pooled TARGET=modality LABEL_MAP='{"T1":0,"T2":1,"FLAIR":2}' CLASS_WEIGHTED=1 \
#     EXP_NAME=modality_clf sbatch train_attr_predictor.sh
#
# sex:
#   TARGET=sex LABEL_MAP='{"M":0,"F":1}' EXP_NAME=sex_clf sbatch train_attr_predictor.sh
# dx (AD/MCI/CN; CN==healthy in the manifest; auto-filters to adni,oasis):
#   DATASET=pooled TARGET=dx LABEL_MAP='{"healthy":0,"MCI":1,"AD":2}' CLASS_WEIGHTED=1 \
#     EXP_NAME=dx_clf sbatch train_attr_predictor.sh
#
#SBATCH --job-name=attr_clf
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

: "${DATASET:=pooled}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"

: "${TARGET:?set TARGET (e.g. sex, dx)}"
: "${LABEL_MAP:?set LABEL_MAP JSON (e.g. '{\"M\":0,\"F\":1}')}"
: "${EXP_NAME:=${TARGET}_clf}"
: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT}/downstream/adherence}"
: "${CLASS_WEIGHTED:=0}"
: "${REAL_LIMIT:=}"
: "${COHORTS:=}"                # build_adherence_csv cohort filter (dx defaults to adni,oasis)
: "${DX_LABELS:=healthy,MCI,AD}"  # dx classes (CN == healthy in the manifest)
: "${MODALITY_LABELS:=T1,T2,FLAIR}"  # modality classes (T1c excluded: vae_only)
# Hyperparams (env-overrideable; defaults match the brain-age regressor).
: "${EPOCHS:=60}"
: "${BS:=4}"
: "${NW:=4}"
: "${LR:=1e-3}"
: "${WD:=1e-4}"
: "${DROPOUT:=0.5}"

EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${LOGS_DIR}" "${EXP_DIR}/weights"
EXP_LOG="${LOGS_DIR}/${EXP_NAME}_attr_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

# resolve_dataset.sh hands us the RAW pooled manifest (cache_key + label cols);
# the predictor needs a filtered rel_path+target CSV → build it inline (the
# rel_path points at the .npy cache under DATA_DIR, i.e. generation space).
ADH_TRAIN_CSV="${EXP_DIR}/adh_${TARGET}_train.csv"
ADH_VALID_CSV="${EXP_DIR}/adh_${TARGET}_valid.csv"
COH_FLAG=""; [[ -n "${COHORTS}" ]] && COH_FLAG="--cohorts ${COHORTS}"
python3 scripts/build_adherence_csv.py --manifest "${TRAIN_CSV}" --target "${TARGET}" \
    --out_csv "${ADH_TRAIN_CSV}" --dx_labels "${DX_LABELS}" \
    --modality_labels "${MODALITY_LABELS}" ${COH_FLAG}
python3 scripts/build_adherence_csv.py --manifest "${VALID_CSV}" --target "${TARGET}" \
    --out_csv "${ADH_VALID_CSV}" --dx_labels "${DX_LABELS}" \
    --modality_labels "${MODALITY_LABELS}" ${COH_FLAG}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

CW_FLAG=""; [[ "${CLASS_WEIGHTED}" == "1" ]] && CW_FLAG="--class_weighted"
RL_FLAG=""; [[ -n "${REAL_LIMIT}" ]] && RL_FLAG="--real_limit ${REAL_LIMIT}"

echo "Attribute classifier (adherence predictor)"
echo "  exp_name    : ${EXP_NAME}"
echo "  dataset     : ${DATASET}  (cfg ${DATASET_CFG})"
echo "  target      : ${TARGET}   label_map=${LABEL_MAP}  class_weighted=${CLASS_WEIGHTED}"
echo "  output_dir  : ${EXP_DIR}"
echo "  manifest    : ${TRAIN_CSV}"
echo "  adh_csv     : ${ADH_TRAIN_CSV} / ${ADH_VALID_CSV}"
echo "  data_dir    : ${DATA_DIR}"
echo "  gpus / nproc: ${NPROC_PER_NODE}"

max_restarts=1000
restarts=$(scontrol show job ${SLURM_JOB_ID} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
function resubmit() {
    if [[ ${restarts} -lt ${max_restarts} ]]; then
        scontrol requeue ${SLURM_JOB_ID}; exit 0
    fi
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
    -m downstream.train_attr_predictor \
      --dataset_config_path "${DATASET_CFG}" \
      --train_csv "${ADH_TRAIN_CSV}" \
      --valid_csv "${ADH_VALID_CSV}" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${OUTPUT_DIR_BASE}" \
      --run_name "${EXP_NAME}" \
      --target "${TARGET}" \
      --label_map "${LABEL_MAP}" \
      --batch_size "${BS}" \
      --num_workers "${NW}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --weight_decay "${WD}" \
      --dropout "${DROPOUT}" \
      ${CW_FLAG} ${RL_FLAG} &
wait
exit 0
