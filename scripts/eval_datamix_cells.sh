#!/bin/bash
# GSDS-side driver: per-CELL recon metric (paired + 4-extractor rFID) for the
# datamix specialist models + the pooled anchor, across 6 interim checkpoints.
#
# For each (model, cell, step) it submits a 5-job dependency CHAIN:
#   J1  gen+paired+rFID  PHASE=all  radimagenet r=0.4   -> recon vols + LPIPS/PSNR/SSIM + rFID(RadImageNet)
#   J2  fid-only         PHASE=fid  radimagenet r=1.0   (afterok J1; ratio-isolation control)
#   J3  fid-only         PHASE=fid  inception   r=0.4   (afterok J2, reuses recon)
#   J4  fid-only         PHASE=fid  swav Woodland-exact (afterok J3)
#   J5  fid-only         PHASE=fid  dinov2      r=0.4   (afterok J4)
# EXTS_OVR overrides the fid-only tag list (default "rad10 inception swav dinov2").
# The chain guarantees recon exists before the fid-only jobs and serialises the
# shared feature-cache path (ignore_existing=True anyway recomputes per run).
#
# Grid (matches scripts/stage_datamix_ckpts.sh):
#   ukbT1      cells: ukb_T1                       (paired + rFID)
#   ukbT1flair cells: ukb_T1 ukb_FLAIR             (paired + rFID)
#   ukbT1adni  cells: ukb_T1 adni_T1               (paired + rFID)
#   pooled     cells: ukb_T1 ukb_FLAIR adni_T1     (rFID compared; paired computed as byproduct)
#   steps: 100000 120000 140000 160000 200000 240000
# => Part A 30 rows (5 model-cells x 6) + Part B 18 rows (pooled 3 cells x 6) = 48 rows x 4 jobs = 192 jobs.
#
# Prereqs on GSDS:
#   - ckpts staged at $STAGE_ROOT/eval-<grp>-ck<step>/weights/vae/checkpoint-<step>/model.pt
#   - per-cell .npy / volume cache + pooled manifests present (same as prior per-cell evals)
#   - RadImageNet FEATURE_EXTRACTOR_PATH (env.local.sh); SWAV_WEIGHT; DINO_REPO + DINO_WEIGHT
#
# Usage:
#   SWAV_WEIGHT=... DINO_REPO=... DINO_WEIGHT=... bash scripts/eval_datamix_cells.sh
#   DRYRUN=1 ... bash scripts/eval_datamix_cells.sh          # print sbatch lines, submit nothing
#   MODELS="ukbT1" STEPS="240000" ... bash scripts/eval_datamix_cells.sh   # subset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/env.local.sh"

export DATASET=pooled
export STAGE=stage1
STAGE_ROOT="${POOLED_OUTPUT_ROOT}/stage1"
CM="${SCRIPT_DIR}/compute_metric.sh"

: "${NUM_IMAGES:=500}"        # per-cell recon sample count (report per-cell convention)
: "${FID_R:=0.4}"             # ratio for the 2.5D extractors (RadImageNet/Inception/DINOv2)
: "${CONTENT_FRAC:=0.15}"     # Woodland content threshold for SwAV
: "${DRYRUN:=0}"

# --- grid ----------------------------------------------------------------------
: "${MODELS:=ukbT1 ukbT1flair ukbT1adni pooled}"
: "${STEPS:=100000 120000 140000 160000 200000 240000}"
declare -A CELLS=(
  [ukbT1]="ukb_T1"
  [ukbT1flair]="ukb_T1 ukb_FLAIR"
  [ukbT1adni]="ukb_T1 adni_T1"
  [pooled]="ukb_T1 ukb_FLAIR adni_T1"
)

# --- extractor presets (env passed to compute_metric.sh) -----------------------
gen_env()  { echo "PHASE=all FID_MODEL_NAME=radimagenet_resnet50 FID_CENTER_SLICES_RATIO=${FID_R}"; }
ext_env()  { # $1 = extractor tag
  case "$1" in
    rad10)     echo "PHASE=fid FID_MODEL_NAME=radimagenet_resnet50 FID_CENTER_SLICES_RATIO=1.0";;
    inception) echo "PHASE=fid FID_MODEL_NAME=imagenet_inception FID_CENTER_SLICES_RATIO=${FID_R}";;
    swav)      echo "PHASE=fid FID_MODEL_NAME=imagenet_swav FID_WOODLAND=1 FID_CONTENT_FRAC=${CONTENT_FRAC} SWAV_WEIGHT=${SWAV_WEIGHT:?set SWAV_WEIGHT}";;
    dinov2)    echo "PHASE=fid FID_MODEL_NAME=dinov2 FID_CENTER_SLICES_RATIO=${FID_R} DINO_REPO=${DINO_REPO:?set DINO_REPO} DINO_WEIGHT=${DINO_WEIGHT:?set DINO_WEIGHT} DINO_ARCH=${DINO_ARCH:-dinov2_vitl14}";;
  esac
}
# rad10 = RadImageNet r=1.0 control: isolates whether the convergence-band
# insensitivity is driven by ratio (r=1.0 flat vs J1 r=0.4) or by the extractor.
: "${EXTS_OVR:=}"
if [[ -n "${EXTS_OVR}" ]]; then EXTS=(${EXTS_OVR}); else EXTS=(rad10 inception swav dinov2); fi

submit() { # $1=depflag(""|--dependency=afterok:JID)  $2..=ENV assignments ; prints jobid
  local dep="$1"; shift
  if [[ "${DRYRUN}" == "1" ]]; then
    echo "    [dryrun] ${dep:+$dep }env $* sbatch ${CM}" >&2
    echo "$(( RANDOM + 100000 ))"   # fake jid so the chain prints
  else
    env "$@" sbatch --parsable ${dep:+$dep} "${CM}"
  fi
}

n_rows=0 n_jobs=0 n_skip=0
for grp in ${MODELS}; do
  for step in ${STEPS}; do
    exp="eval-${grp}-ck${step}"
    ck="${STAGE_ROOT}/${exp}/weights/vae/checkpoint-${step}/model.pt"
    if [[ ! -f "${ck}" ]]; then echo "[skip] missing ${ck}"; n_skip=$((n_skip+1)); continue; fi
    for cell in ${CELLS[$grp]}; do
      n_rows=$((n_rows+1))
      echo "[row] ${grp}  cell=${cell}  ck${step}"
      common="EXP_NAME=${exp} STAGE=stage1 VAE_CKPT_NAME=checkpoint-${step} EVAL_MODE=real_vs_recon DETERMINISTIC=1 NUM_IMAGES=${NUM_IMAGES} CELL=${cell}"
      # J1: gen + paired + RadImageNet rFID
      jid=$(submit "" ${common} $(gen_env)); n_jobs=$((n_jobs+1))
      echo "    J1 radimagenet r=${FID_R}  jid=${jid}"
      # J2..J4: fid-only extractors, chained
      for e in "${EXTS[@]}"; do
        jid=$(submit "--dependency=afterok:${jid}" ${common} $(ext_env "${e}")); n_jobs=$((n_jobs+1))
        echo "    J+ ${e}  jid=${jid}"
      done
    done
  done
done

echo
echo "rows=${n_rows}  jobs=${n_jobs}  skipped_ckpts=${n_skip}  (DRYRUN=${DRYRUN})"
echo "Watch: squeue -u \$USER ; results under ${STAGE_ROOT}/eval-*/  (rFID/paired in logs/ + outputs/)"
echo "Harvest into results_pooled_recon_s1_cells format after completion."
