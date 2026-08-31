#!/bin/bash
# 4-CELL SUBSET stage1 launcher (capacity-dilution probe).
# Single maisi VAE on 8 GPU, trained on ONLY {ukb,adni} x {T1,FLAIR} (4 cells),
# reproducing the OLD-recipe config of `pooled-maisi-kl5e3-s1` EXACTLY:
#   configs/pooled/vae_train_stage1_kl5e3.json  (bs16 x 8 GPU = eff128, lr 2.8e-4,
#   kl 5e-3, max_train_steps 40800 -> cosine T_max 40800 runs to completion,
#   adv_warmup 4000, patch 64^3, l1).  Only the data manifest changes (14 -> 4 cells).
# Compare per-cell metrics vs 14-cell baseline `pooled-maisi-kl5e3-s1-test`.
#   sbatch scripts/train_VAE_4cell.sh
#SBATCH --job-name=vae_4cell
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# Walltime self-continuation (same mechanism as train_VAE_pack.sh): SLURM sends the
# batch shell SIGTERM 180s before the limit; the trap forwards it so train_VAE.py
# checkpoints gracefully, then the script re-sbatches itself. --requeue alone does NOT
# cover TIMEOUT. The run resumes from its latest checkpoint each time.
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/4cell/4cell_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled

# --- 4-cell manifest override (must be set BEFORE resolve_dataset so its :=default is skipped) ---
export TRAIN_CSV="${SCRIPT_DIR}/csv_files/pooled_manifest_train_4cell.csv"
export VALID_CSV="${SCRIPT_DIR}/csv_files/pooled_manifest_valid_4cell.csv"
source scripts/resolve_dataset.sh   # provides DATA_DIR, OUTPUT_ROOT (keeps our TRAIN/VALID_CSV override)

TRAIN_CFG="configs/pooled/vae_train_stage1_kl5e3.json"   # OLD recipe, faithful to pooled-maisi-kl5e3-s1
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/4cell"; mkdir -p "${RUNDIR}"

NAME="pooled-maisi-kl5e3-4cell-s1"
NG="${SLURM_GPUS_ON_NODE:-8}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))
echo "=== 4cell job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; cfg=${TRAIN_CFG}; run=${NAME} ==="
echo "    TRAIN_CSV=${TRAIN_CSV}"
echo "    VALID_CSV=${VALID_CSV}"

# Per-GPU memory/util heartbeat.
( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

mkdir -p "${OUT}/${NAME}"/{embeddings,logs,outputs,weights/vae}
RESUME_FLAG=""
if compgen -G "${OUT}/${NAME}/weights/vae/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi

# Forward walltime SIGTERM -> graceful checkpoint (train_VAE.py handles TERM/INT).
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  [ -n "${TPID}" ] && kill -TERM "${TPID}" 2>/dev/null
}
trap on_term TERM USR1

WANDB_DIR="${OUT}/${NAME}" \
torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29520 \
  train_VAE.py \
    --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
    --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
    --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
    --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "${SLURM_CPUS_PER_TASK}" --target_var 1.0 \
    --run_name "${NAME}" --lambda_cov 0 --lambda_cor 0 --lambda_var 0 ${RESUME_FLAG} \
  >> "${OUT}/${NAME}/logs/${NAME}_4cell_${SLURM_JOB_ID}.log" 2>&1 &
TPID=$!
echo "launched ${NAME} (pid ${TPID}, maisi cov=0 cor=0 var=0, resume='${RESUME_FLAG}') on 8 GPU"

while kill -0 "${TPID}" 2>/dev/null; do wait "${TPID}" 2>/dev/null || true; done
kill "${SMI}" 2>/dev/null

# Self-resubmit until max_train_steps reached (walltime-safe; run completes at 40800).
MAX_STEPS=$(grep -oE '"max_train_steps"[[:space:]]*:[[:space:]]*[0-9]+' "${TRAIN_CFG}" | grep -oE '[0-9]+$')
MAX_STEPS=${MAX_STEPS:-40800}
last=$(ls -d "${OUT}/${NAME}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
last=${last:-0}
echo "  ${NAME}: last_ckpt=${last} / ${MAX_STEPS}"
if [ "${last}" -lt "${MAX_STEPS}" ]; then
  echo "=== self-resubmit (< ${MAX_STEPS}): sbatch scripts/train_VAE_4cell.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_VAE_4cell.sh"
else
  echo "=== reached ${MAX_STEPS}; training complete, no resubmit ==="
fi
echo "=== 4cell DONE ==="
exit 0
