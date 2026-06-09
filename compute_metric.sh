#!/bin/bash
# Unified metric launcher: volume generation + 2.5D FID under torchrun DDP.
# Switch experiments / modes by editing the EXPERIMENT CONFIG block below, or
# override at sbatch time via env vars (e.g. `EXP_NAME=vad1e1 EVAL_MODE=real_vs_recon sbatch compute_metric.sh`).
#
# === SBATCH options below are cluster-specific. Edit for your cluster, =====
# === or override at submission time. ========================================
#SBATCH --job-name=compute_metric
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=04:00:00
#SBATCH -o /dev/null
#SBATCH --open-mode=append

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
: "${EXP_NAME:=vad1e1}"
: "${STAGE:=stage1}"                 # VAE stage that this experiment came from
: "${EVAL_MODE:=real_vs_recon}"        # real_vs_real | real_vs_recon | real_vs_gen
: "${PHASE:=all}"                    # generate | fid | all
: "${NUM_IMAGES:=2500}"
: "${POSTFIX:=30step}"
: "${DATASET:=ukb_20252}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"
: "${OUTPUT_DIR_BASE:=${OUTPUT_ROOT}/${STAGE}}"
: "${VAE_CKPT_NAME:=checkpoint-40000}"
: "${UNET_CKPT_NAME:=checkpoint-10000}"
# UNET_EXP_NAME defaults to EXP_NAME, but can be overridden when VAE and UNet
# live under different experiment dirs (e.g. DFT VAE + plain stage2 UNet).
: "${UNET_EXP_NAME:=${EXP_NAME}}"
: "${BASE_CSV:=${VALID_CSV}}"
: "${OTHER_CSV:=${TRAIN_CSV}}"
: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"
: "${NUM_GPUS:=4}"
# DETERMINISTIC=1 -> real_vs_recon decodes z=z_mu (no posterior sampling noise)
: "${DETERMINISTIC:=0}"
DET_ARG=""; [[ "${DETERMINISTIC}" == "1" ]] && DET_ARG="--deterministic_recon"
# D2: evaluation precision. fp32 + no-amp = reproducible FID (default).
: "${WEIGHT_DTYPE:=fp32}"
: "${NO_AMP:=1}"
AMP_ARG=""; [[ "${NO_AMP}" == "1" ]] && AMP_ARG="--no_amp"
# CELL (pooled only): restrict the real/conditioning set to one cohort×modality
# cell "cohort_modality" (e.g. adni_FLAIR) → per-(modality×cohort) FID. Outputs
# are namespaced under cells/<CELL> so independent cells don't collide; run one
# sbatch per cell (loop). Empty = whole base CSV (single pooled FID).
: "${CELL:=}"

# ----------------------------------------------------------------------
# Experiment directory tree + log
# ----------------------------------------------------------------------
# Weights / latent-stats always live under the main experiment dir; per-cell
# evaluation only namespaces the OUTPUT dir (volumes/features/filelists).
MAIN_EXP_DIR="${OUTPUT_DIR_BASE}/${EXP_NAME}"
if [[ -n "${CELL}" ]]; then
    # write a per-cell base CSV (filter cohort×modality) and isolate outputs
    CELL_COHORT="${CELL%_*}"; CELL_MOD="${CELL##*_}"
    CELL_CSV="${MAIN_EXP_DIR}/cells/${CELL}/base_${CELL}.csv"
    mkdir -p "$(dirname "${CELL_CSV}")"
    python3 - <<PY
import pandas as pd
df = pd.read_csv("${BASE_CSV}")
df = df[(df["cohort"]=="${CELL_COHORT}") & (df["modality"]=="${CELL_MOD}")]
df.to_csv("${CELL_CSV}", index=False)
print(f"[compute_metric.sh] CELL ${CELL}: {len(df)} real volumes -> ${CELL_CSV}")
PY
    BASE_CSV="${CELL_CSV}"
    POSTFIX="${POSTFIX}_${CELL}"
    EXP_DIR="${MAIN_EXP_DIR}/cells/${CELL}"
else
    EXP_DIR="${MAIN_EXP_DIR}"
fi
LOGS_DIR="${EXP_DIR}/logs"
mkdir -p "${LOGS_DIR}" \
         "${EXP_DIR}/outputs/volumes" \
         "${EXP_DIR}/outputs/slices" \
         "${EXP_DIR}/outputs/features"

EXP_LOG="${LOGS_DIR}/${EXP_NAME}_metric_${SLURM_JOB_ID:-$$}.log"
exec >> "${EXP_LOG}" 2>&1

# ----------------------------------------------------------------------
# DDP rendezvous
# ----------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

echo "$(date +"%Y-%m-%d %H-%M-%S") :: compute_metric"
echo "  EXP_NAME   : ${EXP_NAME}"
echo "  UNET_EXP   : ${UNET_EXP_NAME}"
echo "  EVAL_MODE  : ${EVAL_MODE}"
echo "  PHASE      : ${PHASE}"
echo "  NUM_IMAGES : ${NUM_IMAGES}"
echo "  POSTFIX    : ${POSTFIX}"
echo "  EXP_DIR    : ${EXP_DIR}"
echo "  EXP_LOG    : ${EXP_LOG}"
echo "  master     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  num_gpus   : ${NUM_GPUS}"

# ----------------------------------------------------------------------
# Compose CLI args
# ----------------------------------------------------------------------
UNET_EXP_DIR="${OUTPUT_DIR_BASE}/${UNET_EXP_NAME}"
VAE_PATH="${MAIN_EXP_DIR}/weights/vae/${VAE_CKPT_NAME}"
UNET_PATH="${UNET_EXP_DIR}/weights/unet/${UNET_CKPT_NAME}"

# compute_metric.py expects --config_path to contain top-level model defs
# (autoencoder_def, noise_scheduler, diffusion_unet_def) plus inference
# scalars (num_inference_steps, scale_factor, global_mean) for real_vs_gen.
# Build a merged config under EXP_DIR/logs/ from:
#   - configs/${DATASET}/model_fm.json     (model defs)
#   - configs/${DATASET}/diff_train_inf.json -> num_inference_steps
#   - ${UNET_EXP_DIR}/analysis/latent_stats.csv -> scale_factor / global_mean
# The latent stats live under UNET_EXP_DIR because DFT only re-tunes the
# decoder; encoder embeddings (and therefore latent stats) are shared with the
# plain stage2 run.
MERGED_CONFIG_DIR="${LOGS_DIR}"
CONFIG_PATH="${MERGED_CONFIG_DIR}/merged_config_${SLURM_JOB_ID:-$$}.json"
LATENT_STATS_CSV="${UNET_EXP_DIR}/analysis/latent_stats.csv"
python3 - <<PY
import json, os, sys
m = json.load(open("${SCRIPT_DIR}/configs/${DATASET}/model_fm.json"))
# Disable MAISI's norm_float16 training-memory trick for eval: it hard-casts
# GroupNorm output to fp16, which crashes the next conv under fp32+no_amp.
if isinstance(m.get("autoencoder_def"), dict):
    m["autoencoder_def"]["norm_float16"] = False
if "${EVAL_MODE}" == "real_vs_gen":
    d = json.load(open("${SCRIPT_DIR}/configs/${DATASET}/diff_train_inf.json"))
    # Flatten all diffusion_unet_inference keys to top-level so compute_metric.py
    # can access them via args.<key> (e.g. num_inference_steps, stochastic_scale).
    for k, v in d["diffusion_unet_inference"].items():
        m[k] = v
    stats_csv = "${LATENT_STATS_CSV}"
    if not os.path.exists(stats_csv):
        sys.exit(f"[compute_metric.sh] latent_stats.csv missing: {stats_csv}")
    import pandas as pd
    s = pd.read_csv(stats_csv).iloc[0]
    # column name in the CSV is 'scaling_factor'; compute_metric.py reads 'scale_factor'
    m["scale_factor"] = float(s["scaling_factor"])
    m["global_mean"] = float(s["global_mean"])
json.dump(m, open("${CONFIG_PATH}", "w"), indent=2)
print(f"[compute_metric.sh] wrote merged config: ${CONFIG_PATH}")
PY

srun --cpu-bind=none,v --accel-bind=g torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    compute_metric.py \
      --exp_dir "${EXP_DIR}" \
      --dataset_config_path "${DATASET_CFG}" \
      --config_path "${CONFIG_PATH}" \
      --pretrained_vae_path "${VAE_PATH}" \
      --pretrained_unet_path "${UNET_PATH}" \
      --eval_mode "${EVAL_MODE}" \
      --phase "${PHASE}" \
      --num_images "${NUM_IMAGES}" \
      ${DET_ARG} \
      ${AMP_ARG} \
      --weight_dtype "${WEIGHT_DTYPE}" \
      --postfix "${POSTFIX}" \
      --base_label_dir "${BASE_CSV}" \
      --other_label_dir "${OTHER_CSV}" \
      --data_dir "${DATA_DIR}" \
      --feature_extractor_path "${FEATURE_EXTRACTOR_PATH}" \
      --save_volume \
      --save_real &
wait
exit 0
