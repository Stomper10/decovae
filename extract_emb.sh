#!/bin/bash
# Unified embedding extraction + latent analysis launcher.
# Switch experiments by editing the EXPERIMENT CONFIG block below, or override
# at sbatch time via env vars (e.g. `EXP_NAME=cov1e-0 sbatch extract_emb.sh`).
#
# === SBATCH options below are cluster-specific. Edit for your cluster, =====
# === or override at submission time. ========================================
#SBATCH --job-name=extract_emb
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH -o /dev/null

# ----------------------------------------------------------------------
# Per-user environment (conda activate, TMPDIR, default paths, ...)
# See env.local.example.sh for a template. env.local.sh is gitignored.
#
# Under SLURM the script is copied to a spool dir before execution, so
# ${BASH_SOURCE[0]} no longer points at the repo. We try $SLURM_SUBMIT_DIR
# first (the directory you invoked `sbatch` from), then fall back to the
# script's apparent directory for direct shell execution.
# ----------------------------------------------------------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
else
    echo "[WARN] env.local.sh not found at ${SCRIPT_DIR}/env.local.sh — running with system defaults." >&2
fi

# ======================================================================
# EXPERIMENT CONFIG — edit here, or override at sbatch time via env vars
# ======================================================================
: "${EXP_NAME:=vad1e-0}"
: "${STAGE:=stage1}"   # which VAE stage's checkpoint to encode embeddings from
: "${DATASET:=ukb_20252}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"
: "${STAGE_ROOT:=${OUTPUT_ROOT}/${STAGE}}"
: "${WORK_DIR:=${STAGE_ROOT}/${EXP_NAME}}"
: "${CKPT:=${WORK_DIR}/weights/vae/checkpoint-100000/model.pt}"
: "${STAGES:=extract,geometry,stat}" # extract,geometry,stat
: "${CODE_DIR:=${SCRIPT_DIR}}"
: "${DATASET_CFG:=${CODE_DIR}/configs/${DATASET}/dataset.json}"
: "${CONFIG_ENV:=${CODE_DIR}/configs/${DATASET}/environment.json}"
: "${CONFIG_DIFF:=${CODE_DIR}/configs/${DATASET}/diff_train_inf.json}"
: "${CONFIG_MODEL_FM:=${CODE_DIR}/configs/${DATASET}/model_fm.json}"
: "${NUM_GPUS:=4}"

# ----------------------------------------------------------------------
# Experiment directory tree + log
# ----------------------------------------------------------------------
LOGS_DIR="${WORK_DIR}/logs"
mkdir -p "${LOGS_DIR}" "${WORK_DIR}/embeddings" "${WORK_DIR}/analysis"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_emb_${SLURM_JOB_ID:-$$}.log"
exec >> "${EXP_LOG}" 2>&1

echo "$(date +"%Y-%m-%d %H-%M-%S") :: extract_emb"
echo "  EXP_NAME : ${EXP_NAME}"
echo "  WORK_DIR : ${WORK_DIR}"
echo "  CKPT     : ${CKPT}"
echo "  STAGES   : ${STAGES}"
echo "  EXP_LOG  : ${EXP_LOG}"

# ----------------------------------------------------------------------
# Stage 1: extract (distributed under torchrun)
# ----------------------------------------------------------------------
if [[ ",${STAGES}," == *",extract,"* ]]; then
  echo "[extract_emb.sh] running stage=extract under torchrun (${NUM_GPUS} GPUs)"
  torchrun --nproc_per_node=${NUM_GPUS} ${CODE_DIR}/extract_emb.py \
      --work_dir="${WORK_DIR}" \
      --dataset_config_path="${DATASET_CFG}" \
      --data_dir="${DATA_DIR}" \
      --trained_autoencoder_path="${CKPT}" \
      --config_environment="${CONFIG_ENV}" \
      --config_diff_train_inf="${CONFIG_DIFF}" \
      --config_model_fm="${CONFIG_MODEL_FM}" \
      --train_label_dir="${TRAIN_CSV}" \
      --valid_label_dir="${VALID_CSV}" \
      --num_gpus=${NUM_GPUS} \
      --stages=extract
fi

# ----------------------------------------------------------------------
# Stage 2: analysis (single process; geometry/stat manage parallelism internally)
# ----------------------------------------------------------------------
ANALYSIS_STAGES=$(echo "${STAGES}" | tr ',' '\n' | grep -E '^(geometry|stat)$' | paste -sd, -)
if [[ -n "${ANALYSIS_STAGES}" ]]; then
  echo "[extract_emb.sh] running analysis stages=${ANALYSIS_STAGES}"
  python3 ${CODE_DIR}/extract_emb.py \
      --work_dir="${WORK_DIR}" \
      --dataset_config_path="${DATASET_CFG}" \
      --train_label_dir="${TRAIN_CSV}" \
      --stages="${ANALYSIS_STAGES}"
fi

# ----------------------------------------------------------------------
# Filelists for downstream FID / eval pipelines
# ----------------------------------------------------------------------
seq -f "base_%04g.nii.gz"       0 2499 > "${WORK_DIR}/filelist_real_2500.txt"
seq -f "recon_%04g.nii.gz"      0 2499 > "${WORK_DIR}/filelist_recon_2500.txt"
seq -f "gen_%04g_30step.nii.gz" 0 2499 > "${WORK_DIR}/filelist_gen_2500.txt"
echo "[extract_emb.sh] wrote filelists under ${WORK_DIR}/"

echo "$(date +"%Y-%m-%d %H-%M-%S") :: done"
