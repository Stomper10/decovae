#!/bin/bash
# PRODUCTION N-into-1 co-resident stage1 launcher (pair_launcher_policy: stage1
# ablation set only). ONE sbatch (cap=1-safe) runs N train_VAE.py groups co-resident
# on the node's GPUs (distinct master_port). eff32 recipe. Per-group auto-resume so
# SLURM requeue continues from each group's latest checkpoint.
#   sbatch scripts/train_VAE_pack.sh
# Edit RUNS below to 2 groups if the 3-way memory probe (packprobe) shows 3 won't fit.
#SBATCH --job-name=vae_pack
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# Walltime self-continuation: SLURM signals the batch shell SIGTERM 180s before the
# limit (B: = the shell, not job steps). The trap below forwards it to each group so
# train_VAE.py checkpoints gracefully, then the script re-sbatches itself. (--requeue
# only covers preemption/node-failure, NOT TIMEOUT — hence this explicit path.)
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/pack/pack_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

TRAIN_CFG="configs/pooled/vae_train_stage1_eff32.json"   # eff32: bs/gpu=4, lr1.41e-4, 320K steps, adv_warmup16k, kl5e-3
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/pack"
mkdir -p "${RUNDIR}"

# stage1 ablation set co-resident.  EXP_NAME:lambda_cov:lambda_cor:lambda_var
RUNS=(
  "pooled-maisi-kl5e3-eff32-s1:0:0:0"
  "pooled-sid-cor50-kl5e3-eff32-s1:0:50:0"
  "pooled-vad-cov50-kl5e3-eff32-s1:50:0:50"
)
NG="${SLURM_GPUS_ON_NODE:-8}"
NGRP=${#RUNS[@]}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# === co-residency recipe (validated in packprobe 222963: 3/3 groups, ~11 GB/GPU) ===
# Keep P2P (NVLink) ON, disable only host-SHM to remove the cross-group SHM-segment
# collision; stagger launches so each group finishes NCCL init before the next starts.
export NCCL_SHM_DISABLE=1
STAGGER_SEC=90
# CUDA MPS: routes all co-resident processes through one server so their kernels run
# concurrently on the SMs — lets NCCL collectives from multiple comms on a shared GPU
# make progress (without it they block-spin and deadlock).
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then
  sleep 3; echo "[MPS] daemon up"
else
  echo "[MPS] FAILED to start (binary missing or not permitted) — co-residency will likely deadlock"
fi
echo "=== pack job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP}; cfg=${TRAIN_CFG}; P2P=on SHM=off MPS=on stagger=${STAGGER_SEC}s ==="

# Per-GPU TOTAL memory + util every 5 min (catch creep/OOM over the multi-day run).
( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / NGRP ))" --target_var 1.0 )

PIDS=(); PORT=29510
# Forward walltime SIGTERM to every group → train_VAE.py saves a checkpoint and exits
# (SIGTERM/SIGINT handler in train_VAE.py). TERMED guards against double-firing.
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1
for spec in "${RUNS[@]}"; do
  IFS=: read -r name lcov lcor lvar <<< "$spec"
  mkdir -p "${OUT}/${name}"/{embeddings,logs,outputs,weights/vae}
  # Auto-resume: if this group already has checkpoints, continue from them (requeue-safe).
  RESUME_FLAG=""
  if compgen -G "${OUT}/${name}/weights/vae/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi
  # Per-group WANDB_DIR + TORCHINDUCTOR/TRITON cache dirs: co-resident groups sharing one
  # wandb/ dir hang rank-0 before the first dist.barrier, and sharing one inductor autotune
  # cache race on writes → corrupted JSON. Isolate both per group (validated in 222963).
  TORCHINDUCTOR_CACHE_DIR="${OUT}/${name}/torchinductor" \
  TRITON_CACHE_DIR="${OUT}/${name}/triton" \
  WANDB_DIR="${OUT}/${name}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_VAE.py "${COMMON[@]}" --run_name "${name}" \
    --lambda_cov "${lcov}" --lambda_cor "${lcor}" --lambda_var "${lvar}" ${RESUME_FLAG} \
    >> "${OUT}/${name}/logs/${name}_pack_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} (pid $!, cov=${lcov} cor=${lcor} var=${lvar}, resume='${RESUME_FLAG}') port ${PORT}; staggering ${STAGGER_SEC}s"
  PORT=$((PORT+10))
  sleep "${STAGGER_SEC}"   # let this group finish NCCL init before the next starts
done

# Robust wait: `wait` returns early when on_term fires, so loop until each child is
# truly gone (lets the post-SIGTERM checkpoint save complete before we resubmit).
for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p" 2>/dev/null || true; done
done
kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"

# Self-resubmit unless every group has reached MAX_STEPS (training complete).
MAX_STEPS=$(grep -oE '"max_train_steps"[[:space:]]*:[[:space:]]*[0-9]+' "${TRAIN_CFG}" | grep -oE '[0-9]+$')
MAX_STEPS=${MAX_STEPS:-320000}
need_more=0
for spec in "${RUNS[@]}"; do
  IFS=: read -r name _ _ _ <<< "$spec"
  last=$(ls -d "${OUT}/${name}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  echo "  ${name}: last_ckpt=${last} / ${MAX_STEPS}"
  [ "${last}" -lt "${MAX_STEPS}" ] && need_more=1
done
if [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (a group < ${MAX_STEPS}): sbatch ${SCRIPT_DIR}/scripts/train_VAE_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_VAE_pack.sh"
else
  echo "=== all groups reached ${MAX_STEPS}; training complete, no resubmit ==="
fi
echo "=== pack DONE ==="
exit 0
