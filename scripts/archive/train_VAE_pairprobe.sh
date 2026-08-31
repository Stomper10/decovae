#!/bin/bash
# 2-into-1 GPU co-residency PROBE (stage1 ablation pair only — pair_launcher_policy).
# ONE sbatch (cap=1-safe). Launches TWO train_VAE.py groups co-resident on the SAME
# GPUs via two torchrun groups (distinct master_port), and snapshots per-GPU TOTAL
# memory with nvidia-smi (each process's torch.cuda.max_memory_reserved only sees its
# own share, so nvidia-smi is the ground truth for the combined footprint + OOM).
# Throwaway: probe config (bs/gpu=8, adv_warmup=0, 250 steps). 4farm = same h100 80GB.
#   sbatch scripts/train_VAE_pairprobe.sh
#SBATCH --job-name=vae_pairprobe
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=00:30:00
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/pairprobe/pairprobe_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

TRAIN_CFG="configs/pooled/vae_train_stage1_probe.json"   # bs/gpu=8, adv_warmup=0, 250 steps
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/pairprobe"
mkdir -p "${RUNDIR}"
for g in A B; do mkdir -p "${OUT}/pairprobe-${g}"/{embeddings,logs,outputs,weights/vae}; done

export CUDA_VISIBLE_DEVICES=0,1,2,3
NG=4
echo "=== pairprobe job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

# Ground-truth per-GPU TOTAL memory + util snapshots (covers both co-resident processes).
( for i in $(seq 1 24); do
    echo "[smi $(date +%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 15
  done ) &
SMI=$!

# Shared args (inline launch — NO command substitution, so torchrun stays a child
# of THIS shell and `wait` works).
COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / 2 ))" --target_var 1.0 )

# Group A = MAISI (no aux loss); Group B = SID (cor=10). The real ablation pair.
torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29510 \
    train_VAE.py "${COMMON[@]}" --run_name pairprobe-A --lambda_cov 0 --lambda_cor 0 --lambda_var 0 \
    > "${RUNDIR}/A.log" 2>&1 &
A=$!
torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29520 \
    train_VAE.py "${COMMON[@]}" --run_name pairprobe-B --lambda_cov 0 --lambda_cor 10 --lambda_var 0 \
    > "${RUNDIR}/B.log" 2>&1 &
B=$!
echo "launched A(pid ${A}, MAISI) + B(pid ${B}, SID) co-resident on ${NG} GPUs"
wait "${A}"; rcA=$?
wait "${B}"; rcB=$?
kill "${SMI}" 2>/dev/null
echo "=== DONE: A rc=${rcA}, B rc=${rcB} ==="
echo "--- A.log tail ---"; tail -8 "${RUNDIR}/A.log" 2>/dev/null
echo "--- B.log tail ---"; tail -8 "${RUNDIR}/B.log" 2>/dev/null
exit 0
