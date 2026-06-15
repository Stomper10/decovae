#!/bin/bash
# 3D-MedDiffusion reconstruction-metric eval (rFID + LPIPS/PSNR/SSIM).
#
# Scores the 3DMD PatchVolume AE against the SAME real volumes as MAISI/L_SID/
# L_VAD: pooled manifest -> {cache_key}.npy cache (= what PooledAdapter feeds
# compute_metric.py), same fid_resolution / intensity norm, same valid|test
# split. So ONLY the model differs, not the eval data (fair rFID).
#
# Loops over MULTIPLE checkpoints in ONE allocation (cap=1 -> avoid 6 serial
# jobs). Each ckpt's volumes/features/logs are namespaced so they don't collide;
# stage 1 skips already-generated volumes, so a requeue resumes cleanly.
#
# Two conda envs:
#   stage 1 (3d_meddiff): base_/recon_ volume generation (4-GPU shard)
#   stage 2 (deco_v15)  : LPIPS/PSNR/SSIM
#   stage 3 (deco_v15)  : rFID via compute_metric.py --phase fid (RadImageNet)
#
# Usage (pooled, candidate selection on valid, whole-pooled rFID):
#   AE_CKPTS="/abs/a.ckpt /abs/b.ckpt ..." SPLIT=valid sbatch eval_3d_meddiff_metric.sh
# Per-cell final (after best ckpt chosen), one cell per submission:
#   AE_CKPTS="/abs/best.ckpt" SPLIT=test CELL=adni_FLAIR sbatch eval_3d_meddiff_metric.sh
#
#SBATCH --job-name=3dmd_metric
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
[[ -f "${SCRIPT_DIR}/env.local.sh" ]] && source "${SCRIPT_DIR}/env.local.sh"

# ======================================================================
# CONFIG (override at sbatch time via env)
# ======================================================================
: "${DATASET:=pooled}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"   # -> DATA_DIR / VALID_CSV / OUTPUT_ROOT (pooled = .npy cache)
: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"

# valid = ckpt selection ; test = final reported number (test never used for selection).
: "${SPLIT:=valid}"
# resolve_dataset.sh has no TEST_CSV; derive from VALID_CSV (…_valid.csv -> …_test.csv) or POOLED_TEST_CSV.
: "${TEST_CSV:=${POOLED_TEST_CSV:-${VALID_CSV/_valid.csv/_test.csv}}}"
if [[ "${SPLIT}" == "test" ]]; then BASE_CSV_SRC="${TEST_CSV}"; else BASE_CSV_SRC="${VALID_CSV}"; fi

# Space-separated list of AE checkpoint paths (Lightning .ckpt). REQUIRED.
: "${AE_CKPTS:=}"
: "${NUM_IMAGES:=2500}"      # match compute_metric.sh; lower (e.g. 1000) for a faster selection sweep
: "${NUM_SHARDS:=4}"
# FID scale: lower=0.0 = dataset intensity_norm_metric = baseline-matched (compute_metric.sh). 0.5 = paired only.
: "${NORM_LOWER:=0.0}"
: "${NORM_UPPER:=99.5}"
# NO_CLAMP=1: save recon unclamped (MAISI/compute_metric convention; §11). All baselines MUST share this.
: "${NO_CLAMP:=1}"
NOCLAMP_ARG=""; [[ "${NO_CLAMP}" == "1" ]] && NOCLAMP_ARG="--no-clamp"
# CELL (pooled): "cohort_modality" (e.g. adni_FLAIR) -> per-cohort×modality FID. Empty = whole-pooled FID.
: "${CELL:=}"
: "${RESULTS_CSV:=journal_plan/results_pooled_3dmd_cells.csv}"

if [[ -z "${AE_CKPTS}" ]]; then echo "[FATAL] AE_CKPTS is empty — pass a space-separated list of .ckpt paths." >&2; exit 1; fi

EVAL_ROOT="${OUTPUT_ROOT}/recon_eval_3dmd"
mkdir -p "${EVAL_ROOT}/logs"
RUN_LOG="${EVAL_ROOT}/logs/3dmd_metric_${SLURM_JOB_ID:-$$}.log"
exec >> "${RUN_LOG}" 2>&1

echo "=== 3DMD recon-metric eval (job ${SLURM_JOB_ID:-$$}) @ $(date) ==="
echo "  dataset=${DATASET} split=${SPLIT} base_csv=${BASE_CSV_SRC}"
echo "  data_dir=${DATA_DIR}  num_images=${NUM_IMAGES}  cell=${CELL:-<whole-pooled>}"
echo "  ckpts: ${AE_CKPTS}"

source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc

# --- optional CELL filter: write a filtered base CSV (cohort×modality) --------
BASE_CSV="${BASE_CSV_SRC}"
CELL_TAG="pooled"; CELL_COHORT="all"; CELL_MOD="all"
if [[ -n "${CELL}" ]]; then
    CELL_COHORT="${CELL%_*}"; CELL_MOD="${CELL##*_}"; CELL_TAG="${CELL}"
    BASE_CSV="${EVAL_ROOT}/base_${CELL}_${SPLIT}.csv"
    python3 - "$BASE_CSV_SRC" "$BASE_CSV" "$CELL_COHORT" "$CELL_MOD" <<'PY'
import sys, pandas as pd
src, dst, coh, mod = sys.argv[1:5]
df = pd.read_csv(src); df = df[(df["cohort"]==coh) & (df["modality"]==mod)]
df.to_csv(dst, index=False); print(f"[CELL] {coh}_{mod}: {len(df)} real volumes -> {dst}")
PY
fi

# ======================================================================
# Loop over checkpoints (one namespaced eval per ckpt)
# ======================================================================
for AE_CKPT in ${AE_CKPTS}; do
    # step-tagged milestone ckpts live in '…-step=N-train/recon_loss=X.ckpt'
    # subdirs (the '/' in the PL filename template) so basename collides ->
    # prefer the step token from the path; fall back to basename (latest_checkpoint*).
    LABEL=$(echo "${AE_CKPT}" | grep -oE "step=[0-9]+" | head -1)
    [[ -z "${LABEL}" ]] && LABEL=$(basename "${AE_CKPT}" .ckpt)
    LABEL=$(echo "${LABEL}" | tr -c 'A-Za-z0-9._-' '_')
    EXP_NAME="3dmd_${SPLIT}_${CELL_TAG}_${LABEL}"
    EVAL_EXP="${EVAL_ROOT}/${EXP_NAME}"
    VOL_DIR="${EVAL_EXP}/outputs/volumes"
    mkdir -p "${VOL_DIR}"
    echo ""; echo ">>> CKPT ${AE_CKPT}  ->  ${EXP_NAME}  @ $(date)"
    if [[ ! -f "${AE_CKPT}" ]]; then echo "  [SKIP] ckpt not found"; continue; fi

    # -------- stage 1: base + recon generation (3d_meddiff env, 4-GPU shard) --
    conda activate 3d_meddiff
    echo "  -- stage 1: recon generation @ $(date)"
    for ((s = 0; s < NUM_SHARDS; s++)); do
        CUDA_VISIBLE_DEVICES="${s}" python scripts/eval_3d_meddiff_recon.py \
            --dataset-config "${DATASET_CFG}" \
            --base-csv "${BASE_CSV}" \
            --data-dir "${DATA_DIR}" \
            --ae-ckpt "${AE_CKPT}" \
            --out-dir "${VOL_DIR}" \
            --num-images "${NUM_IMAGES}" \
            --norm-lower "${NORM_LOWER}" \
            --norm-upper "${NORM_UPPER}" \
            ${NOCLAMP_ARG} \
            --shard-index "${s}" \
            --num-shards "${NUM_SHARDS}" &
    done
    wait
    NB=$(ls ${VOL_DIR}/base_*.nii.gz 2>/dev/null | wc -l); NR=$(ls ${VOL_DIR}/recon_*.nii.gz 2>/dev/null | wc -l)
    echo "  base=${NB} recon=${NR}"
    conda deactivate 2>/dev/null

    # -------- stage 2: LPIPS / PSNR / SSIM (deco_v15 env) ---------------------
    conda activate deco_v15
    echo "  -- stage 2: LPIPS/PSNR/SSIM @ $(date)"
    QUALITY_CSV="${EVAL_EXP}/quality.csv"
    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_recon_metrics \
        --volumes-dir "${VOL_DIR}" \
        --out-csv "${QUALITY_CSV}" \
        --run-name "${EXP_NAME}"

    # -------- stage 3: rFID (compute_metric.py --phase fid) -------------------
    echo "  -- stage 3: rFID @ $(date)"
    if [[ -n "${FEATURE_EXTRACTOR_PATH}" && -f "${FEATURE_EXTRACTOR_PATH}" ]]; then
        ( cd "${VOL_DIR}" && ls base_*.nii.gz  | sort > "${EVAL_EXP}/filelist_real_${NUM_IMAGES}.txt" \
                          && ls recon_*.nii.gz | sort > "${EVAL_EXP}/filelist_recon_${NUM_IMAGES}.txt" )
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=$((10000 + RANDOM % 50000))
        torchrun --nproc_per_node="${NUM_SHARDS}" --nnodes=1 --rdzv_backend=c10d \
            --rdzv_endpoint=127.0.0.1:${MASTER_PORT} \
            compute_metric.py \
              --exp_dir "${EVAL_EXP}" \
              --dataset_config_path "${DATASET_CFG}" \
              --config_path "${DATASET_CFG}" \
              --eval_mode real_vs_recon \
              --phase fid \
              --num_images "${NUM_IMAGES}" \
              --base_label_dir "${BASE_CSV}" \
              --data_dir "${DATA_DIR}" \
              --feature_extractor_path "${FEATURE_EXTRACTOR_PATH}"
    else
        echo "  [SKIP] FEATURE_EXTRACTOR_PATH not set/found — LPIPS/PSNR/SSIM only (volumes kept in ${VOL_DIR})."
    fi
    conda deactivate 2>/dev/null
    echo "  <<< DONE ${EXP_NAME} @ $(date)  (quality: ${QUALITY_CSV}; fid in ${EVAL_EXP})"
done

echo ""
echo "=== ALL CKPTS DONE @ $(date) ==="
echo "  rFID per ckpt is logged above (search 'FID' / compute_metric output)."
echo "  Collate into ${RESULTS_CSV} (schema = results_pooled_maisi_cells.csv) after picking the best ckpt."
exit 0
