#!/bin/bash
# N-into-1 GPU co-residency PROBE (stage1 ablation set — pair_launcher_policy).
# ONE sbatch (cap=1-safe). Launches N train_VAE.py groups co-resident on the SAME
# GPUs (distinct master_port), and snapshots per-GPU TOTAL memory + util with
# nvidia-smi (each process's torch.cuda.max_memory_reserved only sees its own share).
# Throwaway: probe config (bs/gpu=4 = eff32 footprint, adv_warmup=0, 250 steps).
# 4farm = same h100 80GB → per-GPU mem at N procs transfers to 8farm directly.
#   sbatch scripts/train_VAE_packprobe.sh
#SBATCH --job-name=vae_packprobe
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=00:30:00
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/packprobe/packprobe_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

TRAIN_CFG="configs/pooled/vae_train_stage1_probe4.json"   # bs/gpu=4 (eff32 footprint), adv_warmup=0, 250 steps
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/packprobe"
mkdir -p "${RUNDIR}"

# Full stage1 ablation set co-resident: MAISI + SID + VAD.  name:lambda_cov:lambda_cor:lambda_var
RUNS=( "maisi:0:0:0" "sid:0:50:0" "vad:50:0:50" )
NG=4
NGRP=${#RUNS[@]}
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Co-resident NCCL groups on shared GPUs can deadlock during simultaneous init.
# Mirror the intended REAL-run transport: keep P2P (NVLink, the fast path) ON, disable
# only host-SHM (negligible perf on NVSwitch since P2P covers all pairs, but removes the
# cross-group SHM-segment collision) + stagger launches to kill the init race.
export NCCL_SHM_DISABLE=1
STAGGER_SEC=90
# CUDA MPS: routes all co-resident processes through one server so their kernels run
# CONCURRENTLY on the SMs — the missing piece that lets NCCL collectives from multiple
# comms on a shared GPU make progress (without it they block-spin and deadlock).
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then
  sleep 3
  echo "[MPS] daemon up: $(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr '\n' ' ')"
else
  echo "[MPS] FAILED to start nvidia-cuda-mps-control (binary missing or not permitted) — co-residency will likely deadlock"
fi
echo "=== packprobe job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP}; P2P=on SHM=off MPS=on; stagger=${STAGGER_SEC}s ==="
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

# Ground-truth per-GPU TOTAL memory + util (covers all co-resident processes).
( for i in $(seq 1 24); do
    echo "[smi $(date +%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 15
  done ) &
SMI=$!

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / NGRP ))" --target_var 1.0 )

PIDS=(); PORT=29510
for spec in "${RUNS[@]}"; do
  IFS=: read -r name lcov lcor lvar <<< "$spec"
  mkdir -p "${OUT}/packprobe-${name}"/{embeddings,logs,outputs,weights/vae}
  # Per-group WANDB_DIR so concurrent wandb.init don't collide (the real cause of the
  # earlier rank-0 hang before the first dist.barrier — NOT NCCL co-residency).
  # Per-group TORCHINDUCTOR/TRITON cache dirs: co-resident groups sharing one inductor
  # autotune cache race on writes → corrupted JSON ("Extra data") killed vad in 222911.
  # Same class as the wandb-dir collision; isolate the cache dir per group to fix it.
  TORCHINDUCTOR_CACHE_DIR="${OUT}/packprobe-${name}/torchinductor" \
  TRITON_CACHE_DIR="${OUT}/packprobe-${name}/triton" \
  WANDB_DIR="${OUT}/packprobe-${name}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_VAE.py "${COMMON[@]}" --run_name "packprobe-${name}" \
    --lambda_cov "${lcov}" --lambda_cor "${lcor}" --lambda_var "${lvar}" \
    > "${RUNDIR}/${name}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} (pid $!, cov=${lcov} cor=${lcor} var=${lvar}) port ${PORT}; staggering ${STAGGER_SEC}s"
  PORT=$((PORT+10))
  sleep "${STAGGER_SEC}"   # let this group finish NCCL init before the next starts
done

rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"
echo "=== DONE last-nonzero-rc=${rc} ==="
for spec in "${RUNS[@]}"; do IFS=: read -r name _ _ _ <<< "$spec"; echo "--- ${name}.log tail ---"; tail -6 "${RUNDIR}/${name}.log" 2>/dev/null; done
exit 0
