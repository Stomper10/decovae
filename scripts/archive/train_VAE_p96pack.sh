#!/bin/bash
# PATCH-LEVER (64->96) 3-into-1 co-resident stage1 launcher (⑥ MIUA-recipe A/B).
# ONE sbatch runs 3 train_VAE.py groups co-resident on the 8 GPUs (distinct master_port),
# ALL eff32 (bs4 x 8 = eff32, lr 1.41e-4, 320000 steps, adv_warmup 16000, l1, bf16 amp,
# maisi lambda cov=cor=var=0). ONLY patch=96^3 (vs the p64 baselines) + per-group data/kl:
#   A1 ukbT1  p96 kl8e-4  -> specialist patch-lever (vs existing ukbT1 p64 kl8e-4)
#   A2 pooled p96 kl8e-4  -> pooled patch-lever + pooling-free check @p96 (vs A1)
#   A3 ukbT1  p96 kl5e-4  -> MIUA recipe on ukb_T1/native (patch96+kl5e-4); A3 vs MIUA @r0.4
#                           = pure native-vs-MNI residual. kl-lever @p96 = A1 vs A3.
# num_splits stays 1 (autoencoder_def, model_fm.json) — memory knob, NOT a science lever.
# NOTE: p96 ~3.4x activation memory vs p64; adv turns on at step 16000 (memory jump).
# If 3-into-1 OOMs at ~16k, the STALL guard below stops the resubmit loop (no infinite
# OOM-resubmit on the cap=1 slot) — then bump autoencoder_def.num_splits (1->2) and resume.
#   sbatch scripts/train_VAE_p96pack.sh
#SBATCH --job-name=vae_p96pack
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/p96pack/p96pack_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh   # provides DATA_DIR, OUTPUT_ROOT, WANDB_ENTITY

DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"     # autoencoder_def.num_splits=1 (memory knob)
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/p96pack"; mkdir -p "${RUNDIR}"
CSV="${SCRIPT_DIR}/csv_files"

# Per-group:  EXP_NAME|train_config|train_manifest|valid_manifest
RUNS=(
  "ukbT1-maisi-kl8e4-eff32-p96-s1|configs/pooled/vae_train_stage1_eff32_kl8e4_p96.json|pooled_manifest_train_ukbT1.csv|pooled_manifest_valid_ukbT1.csv"
  "pooled-maisi-kl8e4-eff32-p96-s1|configs/pooled/vae_train_stage1_eff32_kl8e4_p96.json|pooled_manifest_train.csv|pooled_manifest_valid.csv"
  "ukbT1-maisi-kl5e4-eff32-p96-s1|configs/pooled/vae_train_stage1_eff32_kl5e4_p96.json|pooled_manifest_train_ukbT1.csv|pooled_manifest_valid_ukbT1.csv"
)
NG="${SLURM_GPUS_ON_NODE:-8}"
NGRP=${#RUNS[@]}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# === co-residency recipe (validated packprobe 222963 @p64: 3/3 groups) ===
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
echo "=== p96pack job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP}; P2P=on SHM=off MPS=on stagger=${STAGGER_SEC}s ==="

# Per-GPU memory/util heartbeat every 5 min (watch this around step 16000 = adv-on).
( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / NGRP ))" --target_var 1.0 \
  --lambda_cov 0 --lambda_cor 0 --lambda_var 0 )

PIDS=(); PORT=29560
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name traincfg trainman validman <<< "$spec"
  TRAIN_CFG="${traincfg}"
  TRAIN_CSV="${CSV}/${trainman}"
  VALID_CSV="${CSV}/${validman}"
  if [[ ! -f "${TRAIN_CFG}" ]]; then echo "!! missing config for ${name}: ${TRAIN_CFG}"; continue; fi
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
    train_VAE.py "${COMMON[@]}" --train_config_path "${TRAIN_CFG}" --run_name "${name}" \
    --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" ${RESUME_FLAG} \
    >> "${OUT}/${name}/logs/${name}_p96pack_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} (pid $!, cfg=${TRAIN_CFG}, train=${trainman}, resume='${RESUME_FLAG}') port ${PORT}; staggering ${STAGGER_SEC}s"
  PORT=$((PORT+10))
  sleep "${STAGGER_SEC}"
done

for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p" 2>/dev/null || true; done
done
kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"

# Self-resubmit unless every group reached MAX_STEPS.  STALL GUARD: if total progress did
# not advance vs the previous submission, this is a crash loop (e.g. OOM at adv-on) — STOP
# instead of infinitely resubmitting on the cap=1 slot.
MAX_STEPS=320000
STAMP="${RUNDIR}/.p96_progress_stamp"
prev_total=$(cat "${STAMP}" 2>/dev/null || echo -1)
need_more=0; cur_total=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name _ _ _ <<< "$spec"
  last=$(ls -d "${OUT}/${name}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  cur_total=$((cur_total + last))
  echo "  ${name}: last_ckpt=${last} / ${MAX_STEPS}"
  [ "${last}" -lt "${MAX_STEPS}" ] && need_more=1
done
echo "${cur_total}" > "${STAMP}"
if [ "${need_more}" -eq 0 ]; then
  echo "=== all groups reached ${MAX_STEPS}; training complete, no resubmit ==="
elif [ "${prev_total}" -ge 0 ] && [ "${cur_total}" -le "${prev_total}" ]; then
  echo "=== STALL: no progress since last submit (cur=${cur_total} <= prev=${prev_total}) — likely OOM/crash loop."
  echo "=== NOT resubmitting. Fix (e.g. bump autoencoder_def.num_splits 1->2 in configs/pooled/model_fm.json), then: sbatch scripts/train_VAE_p96pack.sh ==="
else
  echo "=== self-resubmit (progress ${prev_total}->${cur_total}, a group < ${MAX_STEPS}): sbatch scripts/train_VAE_p96pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_VAE_p96pack.sh"
fi
echo "=== p96pack DONE ==="
exit 0
