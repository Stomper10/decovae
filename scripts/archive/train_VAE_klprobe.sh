#!/bin/bash
# kl-PROBE co-resident launcher: 4 maisi(λ=0) groups, IDENTICAL eff32 recipe,
# differing ONLY in kl_weight (per-group TRAIN_CFG). Goal: find the kl whose
# posterior sigma plateaus ~0.45 + scale_factor ∈[0.9,1.1] at the eff32 length
# (kl5e-3 over-regularized: sigma 0.87 @170k). ONE sbatch (cap=1-safe), all 4
# co-resident on 8 GPUs via MPS, per-group auto-resume + walltime self-resubmit.
#   sbatch scripts/train_VAE_klprobe.sh
# After picking the winner kl: scancel this, then run the real 3-model pack
# (maisi/sid/vad) at the winning kl — the winning maisi run here can CONTINUE
# (its name already encodes the kl; auto-resume reuses its checkpoints).
#
# NOTE: 4 co-resident groups (validated set was 3 @ ~11GB/GPU in packprobe 222963;
# 4 → ~44GB/GPU, under 80 and under the cap-5 contention limit). Watch the first
# [smi] line for per-GPU memory/util before trusting the run.
#SBATCH --job-name=vae_klprobe
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/pack/klprobe_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/pack"
mkdir -p "${RUNDIR}"

# kl bracket around the sigma~0.45 target (sigma ≈ 174*kl from the kl5e-3/170k point).
# EXP_NAME:lambda_cov:lambda_cor:lambda_var:TRAIN_CFG   (all maisi → λ=0:0:0)
RUNS=(
  "pooled-maisi-kl1e3-eff32-s1:0:0:0:configs/pooled/vae_train_stage1_eff32_kl1e3.json"
  "pooled-maisi-kl2e3-eff32-s1:0:0:0:configs/pooled/vae_train_stage1_eff32_kl2e3.json"
  "pooled-maisi-kl3e3-eff32-s1:0:0:0:configs/pooled/vae_train_stage1_eff32_kl3e3.json"
  "pooled-maisi-kl4e3-eff32-s1:0:0:0:configs/pooled/vae_train_stage1_eff32_kl4e3.json"
)
NG="${SLURM_GPUS_ON_NODE:-8}"
NGRP=${#RUNS[@]}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# === co-residency recipe (validated in packprobe 222963) ===
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
echo "=== klprobe job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP}; P2P=on SHM=off MPS=on stagger=${STAGGER_SEC}s ==="

# Per-GPU TOTAL memory + util every 5 min.
( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

# train_config_path is now PER-GROUP (the only thing that differs) → not in COMMON.
COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / NGRP ))" --target_var 1.0 )

PIDS=(); PORT=29510
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1
for spec in "${RUNS[@]}"; do
  IFS=: read -r name lcov lcor lvar cfg <<< "$spec"
  mkdir -p "${OUT}/${name}"/{embeddings,logs,outputs,weights/vae}
  RESUME_FLAG=""
  if compgen -G "${OUT}/${name}/weights/vae/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi
  TORCHINDUCTOR_CACHE_DIR="${OUT}/${name}/torchinductor" \
  TRITON_CACHE_DIR="${OUT}/${name}/triton" \
  WANDB_DIR="${OUT}/${name}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_VAE.py "${COMMON[@]}" --train_config_path "${cfg}" --run_name "${name}" \
    --lambda_cov "${lcov}" --lambda_cor "${lcor}" --lambda_var "${lvar}" ${RESUME_FLAG} \
    >> "${OUT}/${name}/logs/${name}_klprobe_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} (pid $!, cfg=${cfg}, resume='${RESUME_FLAG}') port ${PORT}; staggering ${STAGGER_SEC}s"
  PORT=$((PORT+10))
  sleep "${STAGGER_SEC}"
done

for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p" 2>/dev/null || true; done
done
kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"

# Self-resubmit unless every group reached its config's max_train_steps.
need_more=0
for spec in "${RUNS[@]}"; do
  IFS=: read -r name _ _ _ cfg <<< "$spec"
  ms=$(grep -oE '"max_train_steps"[[:space:]]*:[[:space:]]*[0-9]+' "${cfg}" | grep -oE '[0-9]+$'); ms=${ms:-320000}
  last=$(ls -d "${OUT}/${name}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  echo "  ${name}: last_ckpt=${last} / ${ms}"
  [ "${last}" -lt "${ms}" ] && need_more=1
done
if [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (a group not yet at max): sbatch ${SCRIPT_DIR}/scripts/train_VAE_klprobe.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_VAE_klprobe.sh"
else
  echo "=== all groups reached max; no resubmit ==="
fi
echo "=== klprobe DONE ==="
exit 0
