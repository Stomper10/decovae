#!/bin/bash
# CO-RESIDENCY SCALING PROBE — measure the real limit (max # of co-resident groups)
# and where aggregate throughput plateaus. ONE sbatch (cap=1-safe). TESTS RUN ON 4farm
# ONLY (same h100 80GB as 8farm). For N in NLIST, launches N IDENTICAL maisi groups
# (lambda 0:0:0) co-resident on all 4 GPUs (each group nproc=4, batch4 = eff32 per-GPU
# shape), runs the probe (250 steps), and records per-group steady-state it/s + aggregate
# + per-GPU mem/util. The limit is a PER-GPU question (how many ranks share one GPU);
# N groups => N ranks/GPU here, identical to N groups of nproc=8 on 8farm => transfers.
# Identical config across groups => differences are pure contention. Inductor cache is
# keyed by GROUP INDEX and reused across N, so each group compiles only once (amortized).
#   sbatch scripts/train_VAE_scaleprobe.sh
#SBATCH --job-name=vae_scaleprobe
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=06:00:00
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/scaleprobe/scaleprobe_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

TRAIN_CFG="configs/pooled/vae_train_stage1_probe4.json"   # bs/gpu=4 (eff32), adv_warmup=0, 250 steps, no valid/ckpt
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/scaleprobe"; mkdir -p "${RUNDIR}"
SUMMARY="${RUNDIR}/summary_${SLURM_JOB_ID}.csv"
echo "N,launched,survived,agg_it_s,per_group_it_s,peak_mem_gb,peak_util" > "${SUMMARY}"

NG="${SLURM_GPUS_ON_NODE:-8}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))
export WANDB_MODE=offline           # never touch the network (offline) for any group
export NCCL_SHM_DISABLE=1           # P2P/NVLink ON, host-SHM OFF (cross-group collision)
NLIST=(1 2 3 4 5 6 8)               # group counts to sweep
STAGGER_SEC=45                      # init stagger between group launches (kill init race)
STEP_FLOOR=120                      # measure steady-state it/s only from this step on

# Shared MPS server for all co-resident groups (concurrent kernels on the SMs).
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then sleep 3; echo "[MPS] daemon up"; else echo "[MPS] FAILED to start"; fi
echo "=== scaleprobe job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; NLIST=${NLIST[*]} ==="
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --lambda_cov 0 --lambda_cor 0 --lambda_var 0 --target_var 1.0 )

for N in "${NLIST[@]}"; do
  echo "==================== N=${N} groups ===================="
  PIDS=(); PORT=29600; CPG=$(( SLURM_CPUS_PER_TASK / N ))   # cpus per group (dataloader workers)
  # background per-GPU nvidia-smi sampler for THIS N
  SMI_LOG="${RUNDIR}/smi_N${N}_${SLURM_JOB_ID}.log"; : > "${SMI_LOG}"
  ( for s in $(seq 1 80); do
      echo "$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' '|')" >> "${SMI_LOG}"
      sleep 10
    done ) & SMI=$!
  for i in $(seq 0 $((N-1))); do
    GLOG="${RUNDIR}/N${N}-g${i}_${SLURM_JOB_ID}.log"; : > "${GLOG}"
    # inductor/triton cache keyed by group INDEX (reused across N → compile once); wandb per (N,i)
    TORCHINDUCTOR_CACHE_DIR="${RUNDIR}/cache-g${i}/inductor" \
    TRITON_CACHE_DIR="${RUNDIR}/cache-g${i}/triton" \
    WANDB_DIR="${RUNDIR}/wandb-N${N}-g${i}" \
    torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
      train_VAE.py "${COMMON[@]}" --run_name "scaleprobe-N${N}-g${i}" --cpus_per_task "${CPG}" \
      > "${GLOG}" 2>&1 &
    PIDS+=($!)
    echo "  launched N${N}-g${i} (pid $!) port ${PORT} cpus/group=${CPG}; staggering ${STAGGER_SEC}s"
    PORT=$((PORT+10)); sleep "${STAGGER_SEC}"
  done
  surv=0; for p in "${PIDS[@]}"; do wait "$p" && surv=$((surv+1)); done
  kill "${SMI}" 2>/dev/null

  # parse steady-state it/s per group + smi peaks
  python3 - "$N" "$surv" "$SUMMARY" "$STEP_FLOOR" "$SMI_LOG" "$RUNDIR"/N${N}-g*_${SLURM_JOB_ID}.log <<'PY'
import re, sys, statistics
N, surv, summary, floor, smilog = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
logs = sys.argv[6:]
rate_re = re.compile(r'(\d+)/\d+ \[[^\]]*?,\s*([\d.]+)(it/s|s/it)')
per_group=[]
for lg in logs:
    try: txt=open(lg,errors='ignore').read()
    except FileNotFoundError: continue
    rates=[]
    for m in rate_re.finditer(txt.replace('\r','\n')):
        step=int(m.group(1)); val=float(m.group(2)); unit=m.group(3)
        if step>=floor and val>0:
            rates.append(val if unit=='it/s' else 1.0/val)
    if rates: per_group.append(statistics.median(rates))
pg = statistics.median(per_group) if per_group else 0.0
agg = sum(per_group)
# smi peaks
peak_mem=0.0; peak_util=0
for line in open(smilog,errors='ignore') if smilog else []:
    for cell in line.strip().strip('|').split('|'):
        parts=[c.strip() for c in cell.split(',')]
        if len(parts)>=3:
            try:
                mem=float(parts[1].split()[0]); ut=int(parts[2].split()[0].replace('%',''))
                peak_mem=max(peak_mem, mem); peak_util=max(peak_util, ut)
            except: pass
with open(summary,'a') as f:
    f.write(f"{N},{N},{surv},{agg:.3f},{pg:.3f},{peak_mem/1024:.1f},{peak_util}\n")
print(f"  [N={N}] survived {surv}/{N} | per-group {pg:.3f} it/s | AGG {agg:.3f} it/s | peak {peak_mem/1024:.1f} GB/GPU util {peak_util}%")
PY
done

echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"
echo "=== SCALING SUMMARY ==="; cat "${SUMMARY}"
echo "=== scaleprobe DONE ==="
exit 0
