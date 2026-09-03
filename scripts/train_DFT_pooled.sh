#!/bin/bash
# Decoder fine-tuning (DFT) for the pooled λ-ablation winner.
#
# WHY THIS AND NOT A STAGE1 RETRAIN OR STAGE2:
#   train_DFT.py:319-324 freezes everything except decoder / quant_conv_d / post_quant,
#   so the ENCODER IS UNTOUCHED and the latents stay bit-identical. That makes DFT the
#   only post-stage1 option compatible with the diffusion UNet already trained on those
#   latents. A 480k stage1 retrain or a 128³ stage2 both retrain the encoder -> new
#   latents -> the whole diffusion run (3 arms x 250k) has to be redone.
#
# BUDGET — epoch-matched to the MIUA DFT precedent (recovered from git, train_DFT.sh):
#   MIUA  : 4 GPU x bs2 = eff 8, 30,000 steps over ukb_20252 (20,201 vol) = 11.88 epochs
#   pooled: 8 GPU x bs2 = eff16, 40,000 steps over 52,169 vol            = 12.27 epochs
#   lr scaled 1e-5 -> 1.414e-5 by sqrt(16/8), the project's standard batch-lr rule.
#   NOTE: train_DFT.py:340 uses CosineAnnealingLR(T_max=total_opt_steps), so this budget
#   is FROZEN once the run starts — extending it later is a new run, not a continuation.
#
# NOISE SCALE:
#   --latent_noise_scale multiplies the sampling noise in DFT's forward
#   (train_DFT.py:205, z = z_mu + noise*z_sigma*scale). Training the decoder on noisier
#   latents makes it robust to the diffusion UNet's latent error, which is what it will
#   actually be fed at generation time. MIUA ran 1.0 for every arm and 3.0 once
#   (cov1e-0_decft_noise3.0) but never compared them. Default 1.0 here; NOISE_SCALE=3.0
#   gives the other arm of that comparison in a separate run.
#
#   sbatch scripts/train_DFT_pooled.sh
#   NOISE_SCALE=3.0 sbatch scripts/train_DFT_pooled.sh     # the untested alternative
#
#SBATCH --job-name=dft_pooled
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/stage1/dft/dft_%j.log
#SBATCH --open-mode=append
# NOT `set -u`: env.local.sh runs `conda activate`, which sources /etc/bashrc,
# which expands $BASHRCSOURCED unbound -> the shell exits before the first echo
# and the job dies with a one-line log. Killed job 263079 that way on 2026-09-01.
# `resolve_dataset.sh` also does indirect ${!var} expansion on unset per-dataset
# vars. Every other launcher here runs without -u for the same reason.
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

# --- what we fine-tune ------------------------------------------------------
SRC_ARM="${SRC_ARM:-pooled-vad-cov1var1-kl8e4-eff32-s1}"   # λ-ablation winner
SRC_STEP="${SRC_STEP:-320000}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
PLOSS_MODEL="${PLOSS_MODEL:-squeeze}"

OUT="${OUTPUT_ROOT}/stage1"
PRETRAINED="${OUT}/${SRC_ARM}/weights/vae/checkpoint-${SRC_STEP}/model.pt"
EXP_NAME="${EXP_NAME:-${SRC_ARM}-decft-n${NOISE_SCALE}}"

TRAIN_CFG="configs/pooled/vae_decft_stage1_eff16.json"
DATASET_CFG="configs/pooled/dataset.json"
MODEL_CFG="configs/pooled/model_fm.json"
RUNDIR="${OUT}/dft"; mkdir -p "${RUNDIR}"
EXP_DIR="${OUT}/${EXP_NAME}"
mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/outputs" "${EXP_DIR}/weights/vae"

NG="${SLURM_GPUS_ON_NODE:-8}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# ---------------------------------------------------------------------------
# PREFLIGHT — the source checkpoint must exist and be a full model.pt.
# DFT writes into its OWN exp dir, so it can never clobber the stage1 weights,
# but a missing/truncated source would silently fine-tune a randomly initialised
# decoder (load_pretrained does not hard-fail on a partial state dict).
# ---------------------------------------------------------------------------
SZ=$(stat -c%s "${PRETRAINED}" 2>/dev/null || echo 0)
echo "[preflight] source     : ${PRETRAINED}"
echo "[preflight] size       : ${SZ} bytes"
echo "[preflight] exp_dir    : ${EXP_DIR}"
echo "[preflight] noise_scale: ${NOISE_SCALE}   ploss: ${PLOSS_MODEL}"
if [[ "${SZ}" != "284762834" && "${SZ}" != "284761362" ]]; then
  echo "=== PREFLIGHT FAILED — ${PRETRAINED} is missing or not a stage1 model.pt"
  echo "=== (expected 284762834 or 284761362 bytes). Not launching. ==="
  exit 1
fi
if [[ "${EXP_DIR}" == "${OUT}/${SRC_ARM}" ]]; then
  echo "=== PREFLIGHT FAILED — EXP_NAME equals the source arm; DFT would overwrite the stage1 weights. ==="
  exit 1
fi
echo "[preflight] OK"

MAX_STEPS=$(grep -oE '"max_train_steps"[[:space:]]*:[[:space:]]*[0-9]+' "${TRAIN_CFG}" | grep -oE '[0-9]+$')
MAX_STEPS=${MAX_STEPS:-40000}
START_TS=$(date +%s)
START_CKPT=$(ls -d "${EXP_DIR}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
START_CKPT=${START_CKPT:-0}

RESUME_FLAG=""
if compgen -G "${EXP_DIR}/weights/vae/checkpoint-*" > /dev/null; then
  RESUME_FLAG="--resume"
  echo "[resume] existing checkpoints found (max ${START_CKPT}) — resuming"
fi

echo "=== dft job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; cfg=${TRAIN_CFG}; target=${MAX_STEPS} ==="

# Per-GPU memory every 5 min — 96³ patches are 3.4x the stage1 volume per sample.
( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

# Forward the walltime SIGTERM so train_DFT.py's graceful_shutdown (l.65-70) saves a
# checkpoint before exiting. TERMED guards against the trap firing twice.
TERMED=0
CHILD=""
on_term() {
  [[ "${TERMED}" == "1" ]] && return
  TERMED=1
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM for graceful checkpoint ==="
  [[ -n "${CHILD}" ]] && kill -TERM "${CHILD}" 2>/dev/null
}
trap on_term TERM USR1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29710
export OMP_NUM_THREADS=1
export WANDB_DIR="${EXP_DIR}"

torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  train_DFT.py \
    --dataset_config_path "${DATASET_CFG}" \
    --model_config_path "${MODEL_CFG}" \
    --train_config_path "${TRAIN_CFG}" \
    --output_dir "${OUT}" \
    --data_dir "${DATA_DIR}" \
    --train_label_dir "${TRAIN_CSV}" \
    --valid_label_dir "${VALID_CSV}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --run_name "${EXP_NAME}" \
    --cpus_per_task "${SLURM_CPUS_PER_TASK}" \
    --pretrained_model_path "${PRETRAINED}" \
    --latent_noise_scale "${NOISE_SCALE}" \
    --ploss_model "${PLOSS_MODEL}" \
    ${RESUME_FLAG} \
    >> "${EXP_DIR}/logs/${EXP_NAME}_dft_${SLURM_JOB_ID}.log" 2>&1 &
CHILD=$!

# `wait` returns early when on_term fires — loop until the child is truly gone so the
# post-SIGTERM checkpoint save finishes before we resubmit.
while kill -0 "${CHILD}" 2>/dev/null; do wait "${CHILD}" 2>/dev/null || true; done
kill "${SMI}" 2>/dev/null

LAST=$(ls -d "${EXP_DIR}/weights/vae/checkpoint-"* 2>/dev/null | sed -E 's#.*checkpoint-##' | sort -n | tail -1)
LAST=${LAST:-0}
ELAPSED=$(( $(date +%s) - START_TS ))
echo "  ${EXP_NAME}: last_ckpt=${LAST} / ${MAX_STEPS}  (elapsed ${ELAPSED}s, started from ${START_CKPT})"

# CRASH-LOOP GUARD — the resubmit exists for walltime preemption, where the run has
# advanced. If it died fast having advanced nothing, resubmitting just re-runs the same
# crash; the UNet pack burned 5 allocations in 20 min that way on 2026-08-28.
if [ "${LAST}" -lt "${MAX_STEPS}" ] && [ "${LAST}" -le "${START_CKPT}" ] && [ "${ELAPSED}" -lt 1800 ]; then
  echo "=== CRASH LOOP: no progress past ${START_CKPT} in ${ELAPSED}s. NOT resubmitting."
  echo "=== Read ${EXP_DIR}/logs/${EXP_NAME}_dft_${SLURM_JOB_ID}.log for the traceback. ==="
elif [ "${LAST}" -lt "${MAX_STEPS}" ]; then
  echo "=== self-resubmit (${LAST} < ${MAX_STEPS}): sbatch scripts/train_DFT_pooled.sh ==="
  cd "${SCRIPT_DIR}" && NOISE_SCALE="${NOISE_SCALE}" SRC_ARM="${SRC_ARM}" SRC_STEP="${SRC_STEP}" \
    sbatch --export=ALL,NOISE_SCALE,SRC_ARM,SRC_STEP "scripts/train_DFT_pooled.sh"
else
  echo "=== reached ${MAX_STEPS}; DFT complete, no resubmit ==="
fi
echo "=== dft DONE ==="
exit 0
