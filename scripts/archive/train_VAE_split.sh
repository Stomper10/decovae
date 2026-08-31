#!/bin/bash
# 4+4 DISJOINT-GPU partition launcher (stage1 ablation pair). ONE sbatch (cap=1-safe).
# Each group gets its OWN GPU subset (no sharing) → independent NCCL groups, NO
# co-residency deadlock (unlike the shared-GPU pack approach). eff32 via per-GPU
# batch 8 on 4 GPUs. Per-group auto-resume (requeue-safe).
#   production : sbatch scripts/train_VAE_split.sh
#   validation : NAME_SUFFIX=-splitval TRAIN_CFG=configs/pooled/vae_train_stage1_probe.json \
#                sbatch scripts/train_VAE_split.sh
#SBATCH --job-name=vae_split
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/split/split_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

: "${TRAIN_CFG:=configs/pooled/vae_train_stage1_eff32_4gpu.json}"   # batch_size 8 → 4 GPU = eff32
: "${NAME_SUFFIX:=}"     # set to e.g. -splitval for a throwaway validation run
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/split"; mkdir -p "${RUNDIR}"

# Two stage1 models on disjoint GPU halves.  EXP_NAME:lambda_cov:lambda_cor:lambda_var
RUNS=(
  "pooled-maisi-kl5e3-eff32-s1:0:0:0"
  "pooled-sid-cor50-kl5e3-eff32-s1:0:50:0"
)
TOTAL_GPU="${SLURM_GPUS_ON_NODE:-8}"
NGRP=${#RUNS[@]}
NG_PER=$(( TOTAL_GPU / NGRP ))
echo "=== split job ${SLURM_JOB_ID} on $(hostname); total_gpu=${TOTAL_GPU}; groups=${NGRP}; gpu/group=${NG_PER}; cfg=${TRAIN_CFG}; suffix='${NAME_SUFFIX}' ==="

( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / NGRP ))" --target_var 1.0 )

PIDS=(); PORT=29510; gpu0=0
for spec in "${RUNS[@]}"; do
  IFS=: read -r base lcov lcor lvar <<< "$spec"
  name="${base}${NAME_SUFFIX}"
  devs=$(seq -s, ${gpu0} $((gpu0+NG_PER-1)))
  mkdir -p "${OUT}/${name}"/{embeddings,logs,outputs,weights/vae}
  RESUME_FLAG=""
  if compgen -G "${OUT}/${name}/weights/vae/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi
  # Per-group WANDB_DIR so the two concurrent wandb.init don't collide on one wandb/ dir
  # (collision/network-retry on rank 0 blocks the first dist.barrier → NCCL store timeout).
  WANDB_DIR="${OUT}/${name}" CUDA_VISIBLE_DEVICES="${devs}" torchrun --nproc_per_node=${NG_PER} --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_VAE.py "${COMMON[@]}" --run_name "${name}" \
    --lambda_cov "${lcov}" --lambda_cor "${lcor}" --lambda_var "${lvar}" ${RESUME_FLAG} \
    >> "${OUT}/${name}/logs/${name}_split_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} on GPUs ${devs} (pid $!, cov=${lcov} cor=${lcor} var=${lvar}, resume='${RESUME_FLAG}') port ${PORT}"
  gpu0=$((gpu0+NG_PER)); PORT=$((PORT+10))
done

rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
kill "${SMI}" 2>/dev/null
echo "=== split DONE last-nonzero-rc=${rc} ==="
exit 0
