#!/bin/bash
# PRODUCTION 3-into-1 co-resident decoder fine-tuning (DFT) launcher.
# ONE sbatch (cap=1-safe) fine-tunes the decoders of all three kl8e4/eff32 stage1 arms
# co-resident on all 8 GPUs (distinct master_port, MPS-shared). Per-group auto-resume,
# and per-group SKIP when a group already reached the step budget — so this is
# idempotent and can be resubmitted freely.
#   sbatch scripts/train_DFT_pack.sh
#   NOISE_SCALE=3.0 sbatch scripts/train_DFT_pack.sh    # the untested alternative arm
#
# WHY A PACK AND NOT THE SINGLE-ARM train_DFT_pooled.sh:
# That launcher fine-tunes only the lambda-ablation winner, which presumed the winner was
# settled. It is not — the generation study puts vad first on Inception but maisi first on
# SwAV/DINOv2 and sid first on RadImageNet. Evaluating arms against each other with a
# DFT'd decoder on one arm and the original decoder on the others confounds the arm
# comparison with decoder quality, so every arm needs the same treatment.
# Co-residency makes that nearly free: solo is 3h31m (measured, job 263197) and three
# groups at the validated ~84% throughput land at ~4h10m in ONE allocation.
#
# MEMORY: the solo run peaked at 13.6 GB of 81.5 GB per GPU, so three groups sit near
# 41 GB (50%). Headroom is ample; contention, not memory, is the co-residency limit.
#
# BUDGET IS FROZEN: train_DFT.py:340 uses CosineAnnealingLR(T_max=total_opt_steps), so
# max_train_steps=40000 cannot be extended later without a schedule confound.
#SBATCH --job-name=dft_pack
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
# Walltime self-continuation: SLURM signals the batch shell SIGTERM 180s before the
# limit (B: = shell, not job steps); the trap forwards it so each group checkpoints,
# then this script re-sbatches itself. (--requeue covers preemption, NOT TIMEOUT.)
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/dft/dftpack_%j.log
#SBATCH --open-mode=append
# NOT `set -u`: env.local.sh runs `conda activate`, which sources /etc/bashrc, which
# expands $BASHRCSOURCED unbound -> the shell exits before the first echo and the job
# dies with a one-line log (killed job 263079 that way on 2026-09-01). resolve_dataset.sh
# also does indirect ${!var} expansion on unset per-dataset vars.
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh   # DATA_DIR, OUTPUT_ROOT, TRAIN_CSV, VALID_CSV, WANDB_ENTITY

SRC_STEP="${SRC_STEP:-320000}"      # the checkpoint the diffusion latents came from
NOISE_SCALE="${NOISE_SCALE:-1.0}"
PLOSS_MODEL="${PLOSS_MODEL:-squeeze}"

TRAIN_CFG="configs/pooled/vae_decft_stage1_eff16.json"
DATASET_CFG="configs/pooled/dataset.json"
MODEL_CFG="configs/pooled/model_fm.json"
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/dft"; mkdir -p "${RUNDIR}"

# Source stage1 arms. Each group's output goes to <arm>-decft-n<NOISE_SCALE>, which is a
# DIFFERENT directory from the arm itself — DFT can never clobber the stage1 weights.
SRC_ARMS=(
  "pooled-maisi-kl8e4-eff32-s1"
  "pooled-sid-cor50-kl8e4-eff32-s1"
  "pooled-vad-cov1var1-kl8e4-eff32-s1"
)
# A full stage1 model.pt is one of these two sizes; anything else is missing/truncated.
# load_pretrained does NOT hard-fail on a partial state dict, so a short file would
# silently fine-tune a randomly initialised decoder.
SZ_OK_A=284762834
SZ_OK_B=284761362

MAX_STEPS=$(grep -oE '"max_train_steps"[[:space:]]*:[[:space:]]*[0-9]+' "${TRAIN_CFG}" | grep -oE '[0-9]+$')
MAX_STEPS=${MAX_STEPS:-40000}

NG="${SLURM_GPUS_ON_NODE:-8}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# ---------------------------------------------------------------------------
# PREFLIGHT — every source checkpoint must exist at full size, and no group may
# write into its own source directory.
# ---------------------------------------------------------------------------
fail=0
declare -a TODO=()
for arm in "${SRC_ARMS[@]}"; do
  pre="${OUT}/${arm}/weights/vae/checkpoint-${SRC_STEP}/model.pt"
  exp="${arm}-decft-n${NOISE_SCALE}"
  dir="${OUT}/${exp}"
  sz=$(stat -c%s "${pre}" 2>/dev/null || echo 0)
  last=$(ls -d "${dir}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  printf "[preflight] %-34s src=%d bytes  done=%6d/%d\n" "${arm}" "${sz}" "${last}" "${MAX_STEPS}"
  if [[ "${sz}" != "${SZ_OK_A}" && "${sz}" != "${SZ_OK_B}" ]]; then
    echo "  !! ${pre} missing or not a stage1 model.pt (expected ${SZ_OK_A} or ${SZ_OK_B})"; fail=1
  fi
  if [[ "${dir}" == "${OUT}/${arm}" ]]; then
    echo "  !! exp dir equals the source arm — DFT would overwrite the stage1 weights"; fail=1
  fi
  if [[ "${last}" -ge "${MAX_STEPS}" ]]; then
    echo "  -- already at ${MAX_STEPS}; skipping this group"
  else
    TODO+=("${arm}")
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "=== PREFLIGHT FAILED — not launching. ==="
  exit 1
fi
if [[ "${#TODO[@]}" -eq 0 ]]; then
  echo "=== all groups already at ${MAX_STEPS}; nothing to do. ==="
  exit 0
fi
echo "[preflight] OK — launching ${#TODO[@]} group(s): ${TODO[*]}"
echo "[preflight] noise_scale=${NOISE_SCALE}  ploss=${PLOSS_MODEL}  src_step=${SRC_STEP}"

# Baseline for the crash-loop guard: wall clock + total steps already on disk. Comparing
# against THIS (not zero) keeps the guard working on resumes.
START_TS=$(date +%s)
START_CKPT=0
for arm in "${TODO[@]}"; do
  d="${OUT}/${arm}-decft-n${NOISE_SCALE}/weights/vae"
  b=$(ls -d "${d}/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  START_CKPT=$(( START_CKPT + ${b:-0} ))
done
echo "[preflight] baseline total steps on disk = ${START_CKPT}"

# === co-residency recipe (validated in packprobe 222963; N=3 sits mid safe-zone) ======
export NCCL_SHM_DISABLE=1          # keep NVLink P2P, drop only the host-SHM collision
STAGGER_SEC=90                     # let each group finish NCCL init before the next
export OMP_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then sleep 3; echo "[MPS] daemon up"
else echo "[MPS] FAILED to start — co-residency will likely deadlock"; fi
echo "=== dftpack job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${#TODO[@]}; cfg=${TRAIN_CFG}; target=${MAX_STEPS}; P2P=on SHM=off MPS=on stagger=${STAGGER_SEC}s ==="

( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" --data_dir "${DATA_DIR}" \
  --train_label_dir "${TRAIN_CSV}" --valid_label_dir "${VALID_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / ${#TODO[@]} ))" \
  --latent_noise_scale "${NOISE_SCALE}" --ploss_model "${PLOSS_MODEL}" )

PIDS=(); PORT=29710
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1

for arm in "${TODO[@]}"; do
  exp="${arm}-decft-n${NOISE_SCALE}"
  dir="${OUT}/${exp}"
  mkdir -p "${dir}"/{logs,outputs,weights/vae}
  RESUME_FLAG=""
  if compgen -G "${dir}/weights/vae/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi
  # Per-group WANDB_DIR + inductor/triton caches: co-resident groups sharing one wandb/
  # dir hang rank-0 before the first barrier, and a shared inductor cache races on writes.
  TORCHINDUCTOR_CACHE_DIR="${dir}/torchinductor" \
  TRITON_CACHE_DIR="${dir}/triton" \
  WANDB_DIR="${dir}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_DFT.py "${COMMON[@]}" \
      --run_name "${exp}" \
      --pretrained_model_path "${OUT}/${arm}/weights/vae/checkpoint-${SRC_STEP}/model.pt" \
      ${RESUME_FLAG} \
      >> "${dir}/logs/${exp}_dftpack_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${exp} (pid $!, resume='${RESUME_FLAG}') port ${PORT}; staggering ${STAGGER_SEC}s"
  PORT=$((PORT+10))
  sleep "${STAGGER_SEC}"
done

# `wait` returns early when on_term fires — loop until each child is truly gone so the
# post-SIGTERM checkpoint save completes before we resubmit.
for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p" 2>/dev/null || true; done
done
kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"

need_more=0; TOTAL_CKPT=0
for arm in "${TODO[@]}"; do
  exp="${arm}-decft-n${NOISE_SCALE}"
  last=$(ls -d "${OUT}/${exp}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  echo "  ${exp}: last_ckpt=${last} / ${MAX_STEPS}"
  TOTAL_CKPT=$(( TOTAL_CKPT + last ))
  [ "${last}" -lt "${MAX_STEPS}" ] && need_more=1
done

# CRASH-LOOP GUARD. The resubmit is meant for walltime preemption, where a long run has
# advanced the checkpoints. If every group died fast and NOTHING advanced, resubmitting
# just re-runs the same crash — the UNet pack burned 5 allocations in 20 min that way on
# 2026-08-28. Require either real progress or a run long enough to have been walltime.
ELAPSED=$(( $(date +%s) - START_TS ))
if [ "${need_more}" -eq 1 ] && [ "${TOTAL_CKPT}" -le "${START_CKPT}" ] && [ "${ELAPSED}" -lt 1800 ]; then
  echo "=== CRASH LOOP: no group advanced past ${START_CKPT} and the job lasted only ${ELAPSED}s."
  echo "=== NOT resubmitting. Read ${OUT}/<arm>-decft-n${NOISE_SCALE}/logs/*_dftpack_${SLURM_JOB_ID}.log for the traceback. ==="
elif [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (a group < ${MAX_STEPS}): sbatch scripts/train_DFT_pack.sh ==="
  cd "${SCRIPT_DIR}" && NOISE_SCALE="${NOISE_SCALE}" SRC_STEP="${SRC_STEP}" PLOSS_MODEL="${PLOSS_MODEL}" \
    sbatch --export=ALL,NOISE_SCALE,SRC_STEP,PLOSS_MODEL "scripts/train_DFT_pack.sh"
else
  echo "=== all groups reached ${MAX_STEPS}; DFT complete, no resubmit ==="
fi
echo "=== dftpack DONE ==="
exit 0
