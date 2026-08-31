#!/bin/bash
# DATA-COMPOSITION 3-into-1 co-resident stage1 launcher (specialist-under-pooled probe).
# ONE sbatch runs 3 train_VAE.py groups co-resident on the 8 GPUs (distinct master_port),
# ALL with the eff32/kl8e4 recipe IDENTICAL to pooled-maisi-kl8e4-eff32-s1 (bs4 x 8 =
# eff32, lr 1.41e-4, kl 8e-4, 320000 steps, adv_warmup 16000, patch 64^3, l1, bf16 amp,
# maisi lambda cov=cor=var=0). ONLY the training-data manifest differs per group:
#   1. ukbT1            — ukb_T1 alone (pure specialist)
#   2. ukbT1_adniT1     — + adni_T1  (cross-cohort, same modality)
#   3. ukbT1_ukbFLAIR   — + ukb_FLAIR (cross-modality, same cohort)
# Isolates what moves the ukb_T1 recon floor (19.29 pooled / 19.28 MAISI zero-shot) when
# cells are added. Same preproc/grid/recipe as pooled -> gap = data composition, not regime.
#   sbatch scripts/train_VAE_datamix_pack.sh
#SBATCH --job-name=vae_datamix
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# Walltime self-continuation: SLURM signals the batch shell SIGTERM 180s before the
# limit (B: = the shell). The trap forwards it so train_VAE.py checkpoints gracefully,
# then the script re-sbatches itself. --requeue only covers preemption, NOT TIMEOUT.
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/datamix/datamix_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh   # provides DATA_DIR, OUTPUT_ROOT, WANDB_ENTITY

TRAIN_CFG="configs/pooled/vae_train_stage1_eff32_kl8e4.json"   # IDENTICAL to pooled-maisi-kl8e4-eff32-s1
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/datamix"; mkdir -p "${RUNDIR}"
CSV="${SCRIPT_DIR}/csv_files"

# Per-group data composition.  EXP_NAME:train_tag:valid_tag  (manifest = pooled_manifest_{train,valid}_<tag>.csv)
RUNS=(
  "ukbT1-maisi-kl8e4-eff32-s1:ukbT1:ukbT1"
  "ukbT1adniT1-maisi-kl8e4-eff32-s1:ukbT1_adniT1:ukbT1_adniT1"
  "ukbT1ukbFLAIR-maisi-kl8e4-eff32-s1:ukbT1_ukbFLAIR:ukbT1_ukbFLAIR"
)
NG="${SLURM_GPUS_ON_NODE:-8}"
NGRP=${#RUNS[@]}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# === co-residency recipe (validated in packprobe 222963: 3/3 groups, ~11 GB/GPU) ===
export NCCL_SHM_DISABLE=1
STAGGER_SEC=90
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then
  sleep 3; echo "[MPS] daemon up"
else
  echo "[MPS] FAILED to start (binary missing or not permitted) — co-residency will likely deadlock"
fi
echo "=== datamix job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP}; cfg=${TRAIN_CFG}; P2P=on SHM=off MPS=on stagger=${STAGGER_SEC}s ==="

# Per-GPU memory/util heartbeat every 5 min.
( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / NGRP ))" --target_var 1.0 \
  --lambda_cov 0 --lambda_cor 0 --lambda_var 0 )

PIDS=(); PORT=29540
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1
for spec in "${RUNS[@]}"; do
  IFS=: read -r name traintag validtag <<< "$spec"
  TRAIN_CSV="${CSV}/pooled_manifest_train_${traintag}.csv"
  VALID_CSV="${CSV}/pooled_manifest_valid_${validtag}.csv"
  if [[ ! -f "${TRAIN_CSV}" || ! -f "${VALID_CSV}" ]]; then
    echo "!! missing manifest for ${name}: ${TRAIN_CSV} / ${VALID_CSV}" ; continue
  fi
  mkdir -p "${OUT}/${name}"/{embeddings,logs,outputs,weights/vae}
  RESUME_FLAG=""
  if compgen -G "${OUT}/${name}/weights/vae/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi
  TORCHINDUCTOR_CACHE_DIR="${OUT}/${name}/torchinductor" \
  TRITON_CACHE_DIR="${OUT}/${name}/triton" \
  WANDB_DIR="${OUT}/${name}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_VAE.py "${COMMON[@]}" --run_name "${name}" \
    --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" ${RESUME_FLAG} \
    >> "${OUT}/${name}/logs/${name}_datamix_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} (pid $!, train=${traintag} valid=${validtag}, resume='${RESUME_FLAG}') port ${PORT}; staggering ${STAGGER_SEC}s"
  PORT=$((PORT+10))
  sleep "${STAGGER_SEC}"
done

for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p" 2>/dev/null || true; done
done
kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"

# Self-resubmit unless every group reached MAX_STEPS.
MAX_STEPS=$(grep -oE '"max_train_steps"[[:space:]]*:[[:space:]]*[0-9]+' "${TRAIN_CFG}" | grep -oE '[0-9]+$')
MAX_STEPS=${MAX_STEPS:-320000}
need_more=0
for spec in "${RUNS[@]}"; do
  IFS=: read -r name _ _ <<< "$spec"
  last=$(ls -d "${OUT}/${name}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  echo "  ${name}: last_ckpt=${last} / ${MAX_STEPS}"
  [ "${last}" -lt "${MAX_STEPS}" ] && need_more=1
done
if [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (a group < ${MAX_STEPS}): sbatch scripts/train_VAE_datamix_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_VAE_datamix_pack.sh"
else
  echo "=== all groups reached ${MAX_STEPS}; training complete, no resubmit ==="
fi
echo "=== datamix pack DONE ==="
exit 0
