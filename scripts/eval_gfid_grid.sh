#!/bin/bash
# Generation eval for the pooled diffusion triad (GSDS driver).
#
# Fills journal_plan/results_pooled_gfid.csv and results_pooled_guidance.csv.
#
# THE TWO-TIER DESIGN — why the modality slices carry the ranking:
#   A single-cell FID at n=500 has a ~21% within-arm CV (measured on the recon
#   trajectory), which is why the reconstruction study needed a paired 14-cell
#   test to separate arms at all. Raising n is the direct fix, but n_real is
#   bounded by the slice: FID is dominated by the SMALLER of the two sets, so
#   generating more cannot rescue a 465-volume cell. Hence:
#     modality (3 slices, n=2500) -> the ARM RANKING. Bootstrap CIs decide it.
#     cell     (13 slices, n=500) -> coverage/diagnostic breakdown + paired check.
#   13 not 14: brats_T1c is vae_only (out of the diffusion vocabulary, plan 4.2).
#
# REAL REFERENCE = the pre-shuffled TRAIN slices from build_gfid_slices.py.
#   Never point BASE_CSV at the raw pooled manifest: it is cohort-ordered, and
#   load_manifest takes the FIRST n rows, so all_T1's reference would come out
#   100% UKB while the conditions are drawn from the true mix. See that script.
#
# CONDITIONS are not invented here — compute_metric.py resamples real token sets
# from BASE_CSV (with replacement), so each slice generates that slice's own
# metadata distribution. Every volume gets a gen_*.cond.json sidecar recording
# the intended condition, so adherence can be scored later off these SAME
# volumes once the predictors exist. Nothing here needs regenerating for that.
#
# Usage (GSDS; override SBATCH resources at submit time, they default to AIBIO):
#   DRYRUN=1 MODE=probe    bash scripts/eval_gfid_grid.sh
#            MODE=guidance bash scripts/eval_gfid_grid.sh     # -> pick g*
#   GUIDANCE=2.0 MODE=main bash scripts/eval_gfid_grid.sh
#            MODE=null     bash scripts/eval_gfid_grid.sh     # arm-independent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CM="${SCRIPT_DIR}/compute_metric.sh"
SLICE_DIR="${SLICE_DIR:-${SCRIPT_DIR}/csv_files/gfid_slices}"

MODE="${MODE:-main}"
ARMS="${ARMS:-pooled-maisi-kl8e4-eff32-s1 pooled-sid-cor50-kl8e4-eff32-s1 pooled-vad-cov1var1-kl8e4-eff32-s1}"
VAE_CKPT="${VAE_CKPT:-checkpoint-320000}"   # MUST be the ckpt the latents came from
UNET_CKPT="${UNET_CKPT:-checkpoint-250000}"
GUIDANCE="${GUIDANCE:-1.0}"
GUIDANCE_SET="${GUIDANCE_SET:-1.0 1.5 2.0 3.0}"
FID_R="${FID_R:-0.4}"
BOOT="${BOOT:-100}"
CONTENT_FRAC="${CONTENT_FRAC:-0.15}"
DRYRUN="${DRYRUN:-0}"

MOD_SLICES="${MOD_SLICES:-all_T1 all_T2 all_FLAIR}"
CELL_SLICES="${CELL_SLICES:-ukb_T1 ukb_FLAIR adni_T1 adni_FLAIR brats_T1 brats_T2 brats_FLAIR hcp_T1 hcp_T2 ixi_T1 ixi_T2 oasis_T1 oasis_T2}"
N_MOD="${N_MOD:-2500}"
N_CELL="${N_CELL:-500}"

# rad at r=1.0 is deliberately absent from the panel: it scored 19.28 vs 19.29 on
# models whose SSIM was 0.315 vs 0.966. Inception r=0.4 is the ranking metric
# (rho=+0.70 vs the paired-metric consensus over 66 points); swav/dino/rad r=0.4
# ride along as a panel and must not drive the ranking.
EXTS=(inception swav dinov2)
gen_env() { echo "PHASE=all FID_MODEL_NAME=radimagenet_resnet50 FID_CENTER_SLICES_RATIO=${FID_R}"; }
ext_env() {
  case "$1" in
    inception) echo "PHASE=fid FID_MODEL_NAME=imagenet_inception FID_CENTER_SLICES_RATIO=${FID_R}";;
    swav)      echo "PHASE=fid FID_MODEL_NAME=imagenet_swav FID_WOODLAND=1 FID_CONTENT_FRAC=${CONTENT_FRAC} SWAV_WEIGHT=${SWAV_WEIGHT:?set SWAV_WEIGHT}";;
    dinov2)    echo "PHASE=fid FID_MODEL_NAME=dinov2 FID_CENTER_SLICES_RATIO=${FID_R} DINO_REPO=${DINO_REPO:?set DINO_REPO} DINO_WEIGHT=${DINO_WEIGHT:?set DINO_WEIGHT} DINO_ARCH=${DINO_ARCH:-dinov2_vitl14}";;
  esac
}

submit() {
  local dep="$1"; shift
  if [[ "${DRYRUN}" == "1" ]]; then
    echo "    [dryrun] ${dep:+$dep }env $* sbatch ${CM}" >&2
    echo "$(( RANDOM + 100000 ))"
  else
    env "$@" sbatch --parsable ${dep:+$dep} "${CM}"
  fi
}

n_rows=0 n_jobs=0

# One row = generate once (J1, which also gives rad) then re-read those volumes
# for each remaining extractor. Every job in a row MUST share POSTFIX, and rows
# must not -- PHASE=fid globs gen_*_<POSTFIX>.nii.gz.
row() {   # row <arm> <slice> <n> <guidance> <extra_env...>
  local arm="$1" slice="$2" n="$3" g="$4"; shift 4
  local csv="${SLICE_DIR}/${slice}.csv"
  [[ -f "${csv}" ]] || { echo "[skip] missing ${csv} (run scripts/build_gfid_slices.py)"; return; }
  local short; short="$(sed -e 's/^pooled-//' -e 's/-kl8e4-eff32-s1$//' <<<"${arm}")"
  n_rows=$((n_rows+1))
  echo "[row] ${short}  ${slice}  n=${n}  g=${g}"
  local common="EXP_NAME=${arm} STAGE=stage1 DATASET=pooled \
VAE_CKPT_NAME=${VAE_CKPT} UNET_CKPT_NAME=${UNET_CKPT} \
EVAL_MODE=real_vs_gen NUM_IMAGES=${n} GUIDANCE_SCALE=${g} \
BASE_CSV=${csv} CELL= OUT_TAG=${slice} FID_BOOTSTRAP=${BOOT} \
POSTFIX=gf_${short} $*"
  local jid; jid=$(submit "" ${common} $(gen_env)); n_jobs=$((n_jobs+1))
  echo "    J1 gen + rad r=${FID_R}   jid=${jid}"
  for e in "${EXTS[@]}"; do
    jid=$(submit "--dependency=afterok:${jid}" ${common} $(ext_env "${e}")); n_jobs=$((n_jobs+1))
    echo "    J+ ${e}  jid=${jid}"
  done
}

null_row() {  # null_row <slice> <n>  -- arm-independent floor; base/other disjoint
  local slice="$1" n="$2"
  local a="${SLICE_DIR}/${slice}_nullA.csv" b="${SLICE_DIR}/${slice}_nullB.csv"
  [[ -f "${a}" && -f "${b}" ]] || { echo "[skip] missing null pair for ${slice}"; return; }
  local avail; avail=$(( $(wc -l < "${a}") - 1 ))
  [[ "${avail}" -lt "${n}" ]] && { echo "  (null n ${n} -> ${avail}: slice too small for two full halves; floor is conservative)"; n="${avail}"; }
  n_rows=$((n_rows+1))
  echo "[null] ${slice}  n=${n}"
  # Parked under the maisi dir purely so it has a home; the value is arm-independent.
  local common="EXP_NAME=$(awk '{print $1}' <<<"${ARMS}") STAGE=stage1 DATASET=pooled \
VAE_CKPT_NAME=${VAE_CKPT} UNET_CKPT_NAME=${UNET_CKPT} \
EVAL_MODE=real_vs_real NUM_IMAGES=${n} \
BASE_CSV=${a} OTHER_CSV=${b} CELL= OUT_TAG=null_${slice} FID_BOOTSTRAP=${BOOT} \
POSTFIX=null"
  local jid; jid=$(submit "" ${common} $(gen_env)); n_jobs=$((n_jobs+1))
  echo "    J1 dump + rad  jid=${jid}"
  for e in "${EXTS[@]}"; do
    jid=$(submit "--dependency=afterok:${jid}" ${common} $(ext_env "${e}")); n_jobs=$((n_jobs+1))
    echo "    J+ ${e}  jid=${jid}"
  done
}

case "${MODE}" in
  probe)
    # Times the 3090s before anything large is committed. 50 volumes, one arm,
    # rad only -- read "GPU 0 [real_vs_gen]" tqdm rate out of the log.
    row "$(awk '{print $1}' <<<"${ARMS}")" ukb_T1 50 1.0
    ;;
  guidance)
    # g is a WITHIN-arm comparison against a fixed reference, so n=500 is enough
    # here even though the cross-arm ranking needs 2500. all_T1 = the richest slice.
    for arm in ${ARMS}; do for g in ${GUIDANCE_SET}; do row "${arm}" all_T1 500 "${g}"; done; done
    ;;
  main)
    for arm in ${ARMS}; do
      for s in ${MOD_SLICES};  do row "${arm}" "${s}" "${N_MOD}"  "${GUIDANCE}"; done
      for s in ${CELL_SLICES}; do row "${arm}" "${s}" "${N_CELL}" "${GUIDANCE}"; done
    done
    ;;
  null)
    for s in ${MOD_SLICES};  do null_row "${s}" "${N_MOD}";  done
    for s in ${CELL_SLICES}; do null_row "${s}" "${N_CELL}"; done
    ;;
  *) echo "unknown MODE=${MODE} (probe|guidance|main|null)" >&2; exit 1;;
esac

echo
echo "MODE=${MODE}  rows=${n_rows}  jobs=${n_jobs}  (DRYRUN=${DRYRUN})"
