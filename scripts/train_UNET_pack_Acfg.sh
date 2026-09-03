#!/bin/bash
# PRODUCTION 3-into-1 co-resident latent-diffusion (UNet) launcher — A-CONFIG.
# Identical to scripts/train_UNET_pack.sh in EVERY respect except one: the model config
# is model_fm_cohort.json instead of model_fm.json, i.e. the conditioning attribute list
# gains a `cohort` categorical (vocab ukb/ixi/hcp/brats/adni/oasis). This is the
# cohort-token ablation agreed in the 2026-06-11 meeting: A = token ON, B = token OFF.
#   sbatch scripts/train_UNET_pack_Acfg.sh
#
# THE ABLATION IS SINGLE-VARIABLE — verified, not assumed:
#   model_fm_cohort.json differs from model_fm.json ONLY by the added cohort attribute.
#   encoder stays "mlp_concat"; cfg_drop_prob 0.15 / per_token_drop_prob 0.1 unchanged;
#   cohort is appended AFTER modality/sex/dx, so modality keeps categorical index 0 and
#   cfg_drop_presence(keep_idx=(0,)) still keeps modality — the CFG null baseline is the
#   same modality-only state as in B. Conditioning vector 13-D -> 19-D.
#   The imbalance sampler (cohort x modality x dx, tau=0.5) lives in dataset.json, which
#   is shared, so it is byte-identical to B. DO NOT touch it here: changing the sampler
#   at the same time as the token would make A-vs-B unattributable.
#
# WHY THE SYMLINKS BELOW:
#   train_UNET.py:407 derives BOTH the latent source and the checkpoint destination from
#   <output_dir>/<run_name>. Reusing the B run_name would overwrite B's UNet checkpoints;
#   a fresh run_name would find no latents. So each A run gets its own directory whose
#   embeddings/, analysis/, train_files.json and valid_files.json are symlinks back to
#   the B arm, while weights/unet/ is its own. train_UNET.py only READS those four
#   (build_file_list joins paths, load_latent_stats reads the csv) — nothing writes into
#   them — so the source arm cannot be corrupted. Costs 0 bytes: embeddings is 87 GB/arm.
#
# BUDGET IS FROZEN: train_UNET.py:505 uses PolynomialLR(total_iters=total_opt_steps), so
# max_train_steps=250000 cannot be extended later without a schedule confound. It must
# also stay at 250000 to be comparable with B.
#SBATCH --job-name=unet_Acfg
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
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/unetpack/unetAcfg_%j.log
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
source scripts/resolve_dataset.sh   # DATA_DIR, OUTPUT_ROOT, TRAIN_CSV, WANDB_ENTITY

TRAIN_CFG="configs/pooled/diff_train_inf_pack8.json"   # identical to B: batch 2 x 8 GPU = eff 16
DATASET_CFG="configs/${DATASET}/dataset.json"          # identical to B: sampler lives here
MODEL_CFG="configs/${DATASET}/model_fm_cohort.json"    # <-- THE ONLY DIFFERENCE FROM B
OUT="${OUTPUT_ROOT}/stage1"
RUNDIR="${OUT}/unetpack"; mkdir -p "${RUNDIR}"

SUFFIX="${SUFFIX:--Acfg}"
SRC_ARMS=(
  "pooled-maisi-kl8e4-eff32-s1"
  "pooled-sid-cor50-kl8e4-eff32-s1"
  "pooled-vad-cov1var1-kl8e4-eff32-s1"
)
EXPECTED_EMB=57536   # pooled train+valid, T1c excluded (51169 + 6367)

NG="${SLURM_GPUS_ON_NODE:-8}"
NGRP=${#SRC_ARMS[@]}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# ---------------------------------------------------------------------------
# LINK + PREFLIGHT — abort on incomplete latents.
# build_file_list() skips missing files SILENTLY (`if not os.path.exists: continue`),
# so a broken symlink would train on a subset with no warning at all.
# ---------------------------------------------------------------------------
fail=0
declare -a RUNS=()
for src in "${SRC_ARMS[@]}"; do
  run="${src}${SUFFIX}"
  s="${OUT}/${src}"
  d="${OUT}/${run}"
  if [[ "${d}" == "${s}" ]]; then
    echo "  !! SUFFIX is empty — A would overwrite B's checkpoints. Refusing."; fail=1; continue
  fi
  mkdir -p "${d}"/{logs,outputs,weights/unet}
  # -n so re-running replaces the symlink instead of nesting inside it.
  ln -sfn "${s}/embeddings"       "${d}/embeddings"
  ln -sfn "${s}/analysis"         "${d}/analysis"
  ln -sfn "${s}/train_files.json" "${d}/train_files.json"
  ln -sfn "${s}/valid_files.json" "${d}/valid_files.json"

  emb=$(find -L "${d}/embeddings" -name '*_emb.nii.gz' 2>/dev/null | wc -l)
  jsn=$(find -L "${d}/embeddings" -name '*_emb.nii.gz.json' 2>/dev/null | wc -l)
  sf="${d}/analysis/latent_stats.csv"
  tj="${d}/train_files.json"; vj="${d}/valid_files.json"
  printf "[preflight] %-40s emb=%6d json=%6d stats=%s lists=%s\n" \
         "${run}" "${emb}" "${jsn}" \
         "$([[ -r ${sf} ]] && echo yes || echo NO)" \
         "$([[ -r ${tj} && -r ${vj} ]] && echo yes || echo NO)"
  [[ "${emb}" -ne "${EXPECTED_EMB}" ]] && { echo "  !! emb ${emb} != expected ${EXPECTED_EMB}"; fail=1; }
  [[ "${emb}" -ne "${jsn}" ]]          && { echo "  !! emb/json count mismatch"; fail=1; }
  [[ -r "${sf}" ]]                     || { echo "  !! latent_stats.csv unreadable through the symlink"; fail=1; }
  [[ -r "${tj}" && -r "${vj}" ]]       || { echo "  !! train/valid_files.json unreadable through the symlink"; fail=1; }
  RUNS+=("${run}")
done
# The cohort token is worthless if the sidecars do not carry cohort. derive_conditions
# writes it for every volume, but patch_cohort_into_cond.py had to be run once on the
# already-extracted latents — verify rather than assume.
probe=$(find -L "${OUT}/${SRC_ARMS[0]}/embeddings" -name '*_emb.nii.gz.json' | head -1)
if [[ -n "${probe}" ]] && ! grep -q '"cohort"' "${probe}"; then
  echo "  !! sidecar ${probe} has no cohort field — run scripts/patch_cohort_into_cond.py first"; fail=1
fi
if [[ "${fail}" -ne 0 ]]; then
  echo "=== PREFLIGHT FAILED — not launching. ==="
  exit 1
fi
echo "[preflight] OK — ${NGRP} A-config arms, cohort token ON, ${EXPECTED_EMB} latents each"
echo "[preflight] model_cfg=${MODEL_CFG}  (B used model_fm.json; everything else identical)"

# Baseline for the crash-loop guard: wall clock + total steps already on disk. Comparing
# against THIS (not zero) keeps the guard working on resumes.
START_TS=$(date +%s)
START_CKPT=0
for run in "${RUNS[@]}"; do
  b=$(ls -d "${OUT}/${run}/weights/unet/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  START_CKPT=$(( START_CKPT + ${b:-0} ))
done
echo "[preflight] baseline total steps on disk = ${START_CKPT}"

# === co-residency recipe (validated in packprobe 222963; N=3 sits mid safe-zone) ======
export NCCL_SHM_DISABLE=1          # keep NVLink P2P, drop only the host-SHM collision
STAGGER_SEC=90                     # let each group finish NCCL init before the next
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then sleep 3; echo "[MPS] daemon up"
else echo "[MPS] FAILED to start — co-residency will likely deadlock"; fi
echo "=== unetAcfg job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP}; cfg=${TRAIN_CFG}; model=${MODEL_CFG}; P2P=on SHM=off MPS=on stagger=${STAGGER_SEC}s ==="

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

PIDS=(); PORT=29810   # distinct from the B pack (29610) so both can co-exist on a node
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups for graceful checkpoint ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1

for run in "${RUNS[@]}"; do
  RESUME_FLAG=""
  if compgen -G "${OUT}/${run}/weights/unet/checkpoint-*" > /dev/null; then RESUME_FLAG="--resume"; fi
  # Per-group WANDB_DIR + inductor/triton caches: co-resident groups sharing one wandb/
  # dir hang rank-0 before the first barrier, and a shared inductor cache races on writes.
  TORCHINDUCTOR_CACHE_DIR="${OUT}/${run}/torchinductor" \
  TRITON_CACHE_DIR="${OUT}/${run}/triton" \
  WANDB_DIR="${OUT}/${run}" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${PORT}" \
    train_UNET.py "${COMMON[@]}" --run_name "${run}" ${RESUME_FLAG} \
    >> "${OUT}/${run}/logs/${run}_unetAcfg_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${run} (pid $!, resume='${RESUME_FLAG}') port ${PORT}; staggering ${STAGGER_SEC}s"
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
for run in "${RUNS[@]}"; do
  last=$(ls -d "${OUT}/${run}/weights/unet/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
  last=${last:-0}
  echo "  ${run}: last_ckpt=${last} / ${MAX_STEPS}"
  TOTAL_CKPT=$(( TOTAL_CKPT + last ))
  [ "${last}" -lt "${MAX_STEPS}" ] && need_more=1
done
# CRASH-LOOP GUARD. The resubmit is meant for walltime preemption, where a long run has
# advanced the checkpoints. If every group died fast and NOTHING advanced, resubmitting
# just re-runs the same crash — the B pack burned 5 allocations in 20 min that way on
# 2026-08-28 (cause: missing train_files.json). Require progress or a walltime-length run.
ELAPSED=$(( $(date +%s) - START_TS ))
if [ "${need_more}" -eq 1 ] && [ "${TOTAL_CKPT}" -le "${START_CKPT}" ] && [ "${ELAPSED}" -lt 1800 ]; then
  echo "=== CRASH LOOP: no group advanced past ${START_CKPT} and the job lasted only ${ELAPSED}s."
  echo "=== NOT resubmitting. Read ${OUT}/<run>/logs/<run>_unetAcfg_${SLURM_JOB_ID}.log for the traceback. ==="
elif [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (a group < ${MAX_STEPS}): sbatch scripts/train_UNET_pack_Acfg.sh ==="
  cd "${SCRIPT_DIR}" && SUFFIX="${SUFFIX}" sbatch --export=ALL,SUFFIX "scripts/train_UNET_pack_Acfg.sh"
else
  echo "=== all groups reached ${MAX_STEPS}; A-config training complete, no resubmit ==="
fi
echo "=== unetAcfg DONE ==="
exit 0
