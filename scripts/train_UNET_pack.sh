#!/bin/bash
# PRODUCTION 3-into-1 co-resident latent-diffusion (UNet) launcher.
# ONE sbatch (cap=1-safe) runs the three kl8e4/eff32 stage1 arms' UNets co-resident
# on all 8 GPUs (distinct master_port, MPS-shared). Per-group auto-resume so a SLURM
# requeue / walltime self-resubmit continues from each group's latest checkpoint.
#   sbatch scripts/train_UNET_pack.sh
#
# WHY CO-RESIDENT AND NOT THE pair PATTERN: pair_launcher_policy forbids packing for
# UNet because SPLITTING the GPUs (CUDA_VISIBLE_DEVICES=0,1 / 2,3) halves the effective
# batch and breaks comparability. Co-residency is a different pattern: every group runs
# nproc=8 over ALL GPUs and shares them via MPS, so batch/LR/schedule are untouched.
# batch_size=2 x 8 GPU = eff 16 == the solo recipe's 4 x 4. Only throughput is traded
# (~84% of solo per group at N=3, measured in coresidency_scaling_limit).
#
# BUDGET IS FROZEN: train_UNET.py:505 uses PolynomialLR(total_iters=total_opt_steps),
# so max_train_steps=250000 cannot be extended later without a schedule confound.
#SBATCH --job-name=unet_pack
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
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/unetpack/unetpack_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh   # DATA_DIR, OUTPUT_ROOT, TRAIN_CSV, WANDB_ENTITY

TRAIN_CFG="configs/pooled/diff_train_inf_pack8.json"   # = diff_train_inf.json but batch_size 2 (8 GPU -> eff 16)
DATASET_CFG="configs/${DATASET}/dataset.json"
MODEL_CFG="configs/${DATASET}/model_fm.json"           # B-variant: no cohort token (A = model_fm_cohort.json, later, winner only)
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/unetpack"; mkdir -p "${RUNDIR}"

# run_name MUST equal the VAE arm name: train_UNET.py reads its latents from
# <output_dir>/<run_name>/embeddings and scale_factor from <run_name>/analysis/latent_stats.csv.
RUNS=(
  "pooled-maisi-kl8e4-eff32-s1"
  "pooled-sid-cor50-kl8e4-eff32-s1"
  "pooled-vad-cov1var1-kl8e4-eff32-s1"
)
EXPECTED_EMB=57536   # pooled train+valid, T1c excluded (51169 + 6367)

NG="${SLURM_GPUS_ON_NODE:-8}"
NGRP=${#RUNS[@]}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# ---------------------------------------------------------------------------
# PREFLIGHT — abort on incomplete latents.
# build_file_list() skips missing files SILENTLY (`if not os.path.exists: continue`),
# so a half-transferred embeddings/ trains on a subset with no warning at all.
# ---------------------------------------------------------------------------
fail=0
for name in "${RUNS[@]}"; do
  d="${OUT}/${name}"
  emb=$(find "${d}/embeddings" -name '*_emb.nii.gz' 2>/dev/null | wc -l)
  jsn=$(find "${d}/embeddings" -name '*_emb.nii.gz.json' 2>/dev/null | wc -l)
  sf="${d}/analysis/latent_stats.csv"
  tj="${d}/train_files.json"; vj="${d}/valid_files.json"
  printf "[preflight] %-34s emb=%6d json=%6d stats=%s lists=%s\n" \
         "${name}" "${emb}" "${jsn}" "$([[ -f ${sf} ]] && echo yes || echo NO)" \
         "$([[ -f ${tj} && -f ${vj} ]] && echo yes || echo NO)"
  [[ "${emb}" -ne "${EXPECTED_EMB}" ]] && { echo "  !! emb ${emb} != expected ${EXPECTED_EMB}"; fail=1; }
  [[ "${emb}" -ne "${jsn}" ]]          && { echo "  !! emb/json count mismatch"; fail=1; }
  [[ -f "${sf}" ]]                     || { echo "  !! missing latent_stats.csv (scale_factor)"; fail=1; }
  # train_UNET.py:402 reads <exp>/train_files.json + valid_files.json. These are written
  # by extract_emb.py at extraction time, so they do NOT come along with an
  # embeddings/-only transfer. Regenerate with scratchpad/gen_filelists.py (pure
  # manifest derivative — extract_subject_id() uses basename only, so the GSDS path
  # prefix inside them is irrelevant).
  [[ -f "${tj}" && -f "${vj}" ]]       || { echo "  !! missing train_files.json / valid_files.json"; fail=1; }
done
if [[ "${fail}" -ne 0 ]]; then
  echo "=== PREFLIGHT FAILED — not launching. Finish the GSDS transfer, then resubmit. ==="
  exit 1
fi
echo "[preflight] OK — all ${NGRP} arms complete at ${EXPECTED_EMB} latents"

# Baseline for the crash-loop guard at the bottom: wall clock + total steps already on
# disk. Comparing against THIS (not against zero) keeps the guard working on resumes,
# where checkpoints from earlier allocations are already present.
START_TS=$(date +%s)
START_CKPT=0
for name in "${RUNS[@]}"; do
  b=$(ls -d "${OUT}/${name}/weights/unet/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  START_CKPT=$(( START_CKPT + ${b:-0} ))
done
echo "[preflight] baseline total steps on disk = ${START_CKPT}"

# === co-residency recipe (validated in packprobe 222963; N=3 sits mid safe-zone) ===
export NCCL_SHM_DISABLE=1          # keep NVLink P2P, drop only the host-SHM collision
STAGGER_SEC=90                     # let each group finish NCCL init before the next
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then sleep 3; echo "[MPS] daemon up"
else echo "[MPS] FAILED to start — co-residency will likely deadlock"; fi
echo "=== unetpack job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP}; cfg=${TRAIN_CFG}; P2P=on SHM=off MPS=on stagger=${STAGGER_SEC}s ==="

( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

# train_UNET.py takes NO --data_dir / --valid_label_dir / --target_var.
COMMON=( --dataset_config_path "${DATASET_CFG}" --model_config_path "${MODEL_CFG}" \
  --train_config_path "${TRAIN_CFG}" --output_dir "${OUT}" \
  --train_label_dir "${TRAIN_CSV}" \
  --wandb_entity "${WANDB_ENTITY}" --cpus_per_task "$(( SLURM_CPUS_PER_TASK / NGRP ))" )

PIDS=(); PORT=29610
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1

for name in "${RUNS[@]}"; do
  mkdir -p "${OUT}/${name}"/{logs,outputs,weights/unet}
  RESUME_FLAG=""
  if compgen -G "${OUT}/${name}/weights/unet/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi
  # Per-group WANDB_DIR + inductor/triton caches: co-resident groups sharing one wandb/
  # dir hang rank-0 before the first barrier, and a shared inductor cache races on writes.
  TORCHINDUCTOR_CACHE_DIR="${OUT}/${name}/torchinductor" \
  TRITON_CACHE_DIR="${OUT}/${name}/triton" \
  WANDB_DIR="${OUT}/${name}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_UNET.py "${COMMON[@]}" --run_name "${name}" ${RESUME_FLAG} \
    >> "${OUT}/${name}/logs/${name}_unetpack_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} (pid $!, resume='${RESUME_FLAG}') port ${PORT}; staggering ${STAGGER_SEC}s"
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

MAX_STEPS=$(grep -oE '"max_train_steps"[[:space:]]*:[[:space:]]*[0-9]+' "${TRAIN_CFG}" | grep -oE '[0-9]+$')
MAX_STEPS=${MAX_STEPS:-250000}
need_more=0; TOTAL_CKPT=0
for name in "${RUNS[@]}"; do
  last=$(ls -d "${OUT}/${name}/weights/unet/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  echo "  ${name}: last_ckpt=${last} / ${MAX_STEPS}"
  TOTAL_CKPT=$(( TOTAL_CKPT + last ))
  [ "${last}" -lt "${MAX_STEPS}" ] && need_more=1
done
# CRASH-LOOP GUARD. The resubmit is meant for walltime preemption, where a long run
# has advanced the checkpoints. If instead every group died fast and NOTHING advanced,
# resubmitting just re-runs the same crash — job 262231->262376->262377->262379->262380
# burned 5 allocations in 20 min that way (cause: missing train_files.json). Require
# either real progress or a run long enough to have been walltime, else stop and report.
ELAPSED=$(( $(date +%s) - START_TS ))
if [ "${need_more}" -eq 1 ] && [ "${TOTAL_CKPT}" -le "${START_CKPT}" ] && [ "${ELAPSED}" -lt 1800 ]; then
  echo "=== CRASH LOOP: no group advanced past ${START_CKPT} and the job lasted only ${ELAPSED}s."
  echo "=== NOT resubmitting. Read ${OUT}/<arm>/logs/<arm>_unetpack_${SLURM_JOB_ID}.log for the traceback. ==="
elif [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (a group < ${MAX_STEPS}): sbatch scripts/train_UNET_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_UNET_pack.sh"
else
  echo "=== all groups reached ${MAX_STEPS}; training complete, no resubmit ==="
fi
echo "=== unetpack DONE ==="
exit 0
