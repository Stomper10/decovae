#!/bin/bash
# kl5e-4 4-run SLOT-SCHEDULED stage1 launcher (3 slots, 4 groups).
# kl is FIXED at 5e-4 across all four; only patch and data vary -> two clean single levers.
#   G1 ukbT1  p96  RESUME 240000 -> 320000   (= A3, the only p96 arm still improving)
#   G2 ukbT1  p64  0 -> 320000                 patch lever vs G1
#   G3 pooled p96  0 -> 320000                 data lever vs G1
#   G4 pooled p64  0 -> 320000                 completes the 2x2; STARTS WHEN A SLOT FREES
#
# SLOT SCHEDULING: at most SLOTS(=3) groups run co-resident. G1 needs only 80k steps, so
# when it completes, G4 is launched into its slot inside the SAME job -- no waiting for the
# next 24h resubmit. Scheduling is STATE-BASED (reads checkpoint-* on disk), so a requeue
# or resubmit re-derives who still needs work; order in RUNS is priority order.
#
# NOTE ON G4: $OUT/pooled-maisi-kl5e4-eff32-s1 ALREADY holds a 60000-step checkpoint from
# the June kl probe (job 225013). That probe used the IDENTICAL spec -- same config file,
# same pooled manifests, lambdas 0 -- so G4 RESUMES from it and saves ~260k/320k of the run.
# To force a from-scratch G4 instead, rename that directory first; this script never deletes.
#
# TRAP: --signal=B:TERM@180 fires 180s before the walltime kill. The trap now RESUBMITS
# FIRST (before touching the children), then TERMs them, then sweeps orphaned model.pt.tmp.
# Job 249470 died because the old trap saved-then-resubmitted and got SIGKILLed in between,
# breaking the chain. Stall-guard still applies on the NORMAL exit path (= children died on
# their own = crash loop), which is the case it was written for.
#   sbatch scripts/train_VAE_kl5e4pack.sh
#SBATCH --job-name=vae_kl5e4pack
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/kl5e4pack/kl5e4pack_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh   # DATA_DIR, OUTPUT_ROOT, WANDB_ENTITY

DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/kl5e4pack"; mkdir -p "${RUNDIR}"
CSV="${SCRIPT_DIR}/csv_files"
MAX_STEPS=320000
SLOTS=3
STAGGER_SEC=90
POLL_SEC=60

# EXP_NAME|train_config|train_manifest|valid_manifest|master_port
RUNS=(
  "ukbT1-maisi-kl5e4-eff32-p96-s1|configs/pooled/vae_train_stage1_eff32_kl5e4_p96.json|pooled_manifest_train_ukbT1.csv|pooled_manifest_valid_ukbT1.csv|29560"
  "ukbT1-maisi-kl5e4-eff32-s1|configs/pooled/vae_train_stage1_eff32_kl5e4.json|pooled_manifest_train_ukbT1.csv|pooled_manifest_valid_ukbT1.csv|29570"
  "pooled-maisi-kl5e4-eff32-p96-s1|configs/pooled/vae_train_stage1_eff32_kl5e4_p96.json|pooled_manifest_train.csv|pooled_manifest_valid.csv|29580"
  "pooled-maisi-kl5e4-eff32-s1|configs/pooled/vae_train_stage1_eff32_kl5e4.json|pooled_manifest_train.csv|pooled_manifest_valid.csv|29590"
)

NG="${SLURM_GPUS_ON_NODE:-8}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))
export NCCL_SHM_DISABLE=1
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then sleep 3; echo "[MPS] daemon up"
else echo "[MPS] FAILED to start — co-residency will likely deadlock"; fi
echo "=== kl5e4pack job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; slots=${SLOTS}; groups=${#RUNS[@]} ==="

( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / SLOTS ))" --target_var 1.0 \
  --lambda_cov 0 --lambda_cor 0 --lambda_var 0 )

last_step() {   # highest checkpoint-N holding a COMPLETE model.pt (ignores *.tmp)
  local d="${OUT}/$1/weights/vae" s=0 c
  for c in "${d}"/checkpoint-*; do
    [[ -f "${c}/model.pt" ]] || continue
    local n="${c##*checkpoint-}"; [[ "${n}" =~ ^[0-9]+$ ]] && (( n > s )) && s="${n}"
  done
  echo "${s}"
}

sweep_tmp() {   # drop checkpoint dirs left holding only a truncated model.pt.tmp
  local d c
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "$spec"
    d="${OUT}/${name}/weights/vae"
    for c in "${d}"/checkpoint-*; do
      [[ -d "$c" && -f "${c}/model.pt.tmp" && ! -f "${c}/model.pt" ]] || continue
      echo "[cleanup] removing truncated ${c}"; rm -rf "$c"
    done
  done
}

work_remains() {
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "$spec"
    (( $(last_step "${name}") < MAX_STEPS )) && return 0
  done
  return 1
}

resubmit() { cd "${SCRIPT_DIR}" && sbatch "scripts/train_VAE_kl5e4pack.sh"; }

declare -a PIDS=() NAMES=()
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T) ==="
  # 1. chain FIRST — a SIGKILL during shutdown must not break the resubmit
  if work_remains; then echo "=== [signal] resubmitting now (work remains) ==="; resubmit
  else echo "=== [signal] all groups at ${MAX_STEPS}; no resubmit ==="; fi
  # 2. then let the children checkpoint
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  for _ in $(seq 1 24); do
    local alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    (( alive )) || break; sleep 5
  done
  sweep_tmp
  echo "=== [signal] shutdown done at $(date +%F_%T) ==="
  exit 0
}
trap on_term TERM USR1

sweep_tmp   # clear anything a previous kill left behind before deciding resume state

launch() {
  local spec="$1" name traincfg trainman validman port
  IFS='|' read -r name traincfg trainman validman port <<< "$spec"
  local TRAIN_CSV="${CSV}/${trainman}" VALID_CSV="${CSV}/${validman}"
  if [[ ! -f "${traincfg}" ]]; then echo "!! missing config ${traincfg} for ${name}"; return 1; fi
  if [[ ! -f "${TRAIN_CSV}" || ! -f "${VALID_CSV}" ]]; then echo "!! missing manifest for ${name}"; return 1; fi
  mkdir -p "${OUT}/${name}"/{embeddings,logs,outputs,weights/vae}
  local RESUME_FLAG="" from
  from=$(last_step "${name}")
  (( from > 0 )) && RESUME_FLAG="--resume"
  TORCHINDUCTOR_CACHE_DIR="${OUT}/${name}/torchinductor" \
  TRITON_CACHE_DIR="${OUT}/${name}/triton" \
  WANDB_DIR="${OUT}/${name}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${port}" \
    train_VAE.py "${COMMON[@]}" --train_config_path "${traincfg}" --run_name "${name}" \
    --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" ${RESUME_FLAG} \
    >> "${OUT}/${name}/logs/${name}_kl5e4pack_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!); NAMES+=("${name}")
  echo "[launch] ${name} (pid $!, from step ${from}, cfg=${traincfg}, train=${trainman}) port ${port}"
  return 0
}

# --- slot scheduler ------------------------------------------------------------
PENDING=()
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name _ _ _ _ <<< "$spec"
  s=$(last_step "${name}")
  if (( s >= MAX_STEPS )); then echo "[skip] ${name} already at ${s} >= ${MAX_STEPS}"
  else PENDING+=("$spec"); echo "[queue] ${name} (at ${s})"; fi
done

while true; do
  while (( ${#PIDS[@]} < SLOTS )) && (( ${#PENDING[@]} > 0 )); do
    spec="${PENDING[0]}"; PENDING=("${PENDING[@]:1}")
    launch "$spec" && sleep "${STAGGER_SEC}"
  done
  (( ${#PIDS[@]} == 0 )) && break
  sleep "${POLL_SEC}"
  live_p=(); live_n=()
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then live_p+=("${PIDS[$i]}"); live_n+=("${NAMES[$i]}")
    else
      wait "${PIDS[$i]}" 2>/dev/null
      echo "[done] ${NAMES[$i]} exited at $(date +%F_%T) (last ckpt $(last_step "${NAMES[$i]}")) — slot freed"
    fi
  done
  PIDS=("${live_p[@]}"); NAMES=("${live_n[@]}")
done

kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"
sweep_tmp

# --- normal exit path: children died on their own -> stall guard applies --------
STAMP="${RUNDIR}/.kl5e4_progress_stamp"
prev_total=$(cat "${STAMP}" 2>/dev/null || echo -1)
need_more=0; cur_total=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name _ _ _ _ <<< "$spec"
  last=$(last_step "${name}"); cur_total=$((cur_total + last))
  echo "  ${name}: last_ckpt=${last} / ${MAX_STEPS}"
  (( last < MAX_STEPS )) && need_more=1
done
echo "${cur_total}" > "${STAMP}"
if (( need_more == 0 )); then
  echo "=== all groups reached ${MAX_STEPS}; training complete, no resubmit ==="
elif (( prev_total >= 0 )) && (( cur_total <= prev_total )); then
  echo "=== STALL: no progress since last submit (cur=${cur_total} <= prev=${prev_total}) — crash loop."
  echo "=== NOT resubmitting. Fix (e.g. autoencoder_def.num_splits 1->2), then: sbatch scripts/train_VAE_kl5e4pack.sh ==="
else
  echo "=== self-resubmit (progress ${prev_total}->${cur_total}) ==="
  resubmit
fi
echo "=== kl5e4pack DONE ==="
exit 0
