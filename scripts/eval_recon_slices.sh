#!/bin/bash
# Pooled-SLICE reconstruction FID, before and after decoder fine-tuning. GSDS.
#
#   bash scripts/eval_recon_slices.sh                 # 18 rows: 3 arms x 3 slices x 2 decoders
#   DECODERS=dft bash scripts/eval_recon_slices.sh    # only the DFT half
#   EXTS="inception swav dinov2" bash scripts/eval_recon_slices.sh
#   DRYRUN=1 bash scripts/eval_recon_slices.sh
#
# WHY THIS EXISTS — two gaps it closes at once.
#
# (a) THE AGGREGATION MISMATCH. Every reconstruction number we have is a MEAN OF
#     PER-CELL FIDs (results_*_percell.csv), while every generation number is a
#     single FID over one pooled mixed sample. Those are different quantities, so
#     the report's Table 1 and Table 2 cannot legitimately be read across. This
#     measures reconstruction the way generation is measured — one FID over the
#     same pre-shuffled train slice, same n — so the two tables finally differ in
#     exactly one thing: generation vs reconstruction.
#
# (b) DFT HAS NEVER BEEN VALIDATED. vae_decft_stage1_eff16.json sets
#     validation_steps=1000 but nothing reached the DFT logs, so step 40,000 has
#     never been compared against the un-tuned stage1 decoder. We are holding
#     three decoders of unknown value. A decoder trained for robustness to latent
#     noise can drift toward the conditional mean and blur, which would show up
#     here as WORSE recon FID. Nothing so far rules that out, so DFT must not be
#     adopted before this grid runs.
#
# THE CHECKPOINT ASYMMETRY THIS ALSO FIXES. The per-cell recon table compares
# maisi@240k, sid@280k and vad@320k — each arm at its own selected point — while
# the diffusion latents all came from 320k. Only vad's two conditions coincided.
# Here every arm is measured at 320k (orig) and at its DFT of 320k, so the
# comparison is checkpoint-aligned throughout.
#
# WHAT MUST NOT DRIFT: FID_CENTER_SLICES_RATIO=0.4 and n=2500 match the existing
# generation table exactly. ratio 1.0 is quality-blind (it scored 19.28 vs 19.29
# on models whose SSIM was 0.315 vs 0.966) and is never used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CM="${SCRIPT_DIR}/compute_metric.sh"
SLICE_DIR="${SLICE_DIR:-${SCRIPT_DIR}/csv_files/gfid_slices}"

ARMS="${ARMS:-pooled-maisi-kl8e4-eff32-s1 pooled-sid-cor50-kl8e4-eff32-s1 pooled-vad-cov1var1-kl8e4-eff32-s1}"
SLICES="${SLICES:-all_T1 all_T2 all_FLAIR}"
DECODERS="${DECODERS:-orig dft}"      # orig = stage1 checkpoint-320000, dft = decft checkpoint-40000
ORIG_CKPT="${ORIG_CKPT:-checkpoint-320000}"
DFT_CKPT="${DFT_CKPT:-checkpoint-40000}"
DFT_SUFFIX="${DFT_SUFFIX:--decft-n1.0}"
N="${N:-2500}"
FID_R="${FID_R:-0.4}"
BOOT="${BOOT:-100}"
CONTENT_FRAC="${CONTENT_FRAC:-0.15}"
DRYRUN="${DRYRUN:-0}"

# Inception r=0.4 is the ranking metric (rho=+0.70 vs the paired-metric consensus
# over 66 points). swav/dino/rad ride along as a diagnostic panel when asked for
# and must not drive the ranking. Default to inception alone: the panel triples
# the job count and this grid's job is the DFT decision, not the extractor study.
EXTS=(${EXTS:-inception})

ext_env() {
  case "$1" in
    inception) echo "FID_MODEL_NAME=imagenet_inception FID_CENTER_SLICES_RATIO=${FID_R}";;
    rad)       echo "FID_MODEL_NAME=radimagenet_resnet50 FID_CENTER_SLICES_RATIO=${FID_R}";;
    swav)      echo "FID_MODEL_NAME=imagenet_swav FID_WOODLAND=1 FID_CONTENT_FRAC=${CONTENT_FRAC} SWAV_WEIGHT=${SWAV_WEIGHT:?set SWAV_WEIGHT}";;
    dinov2)    echo "FID_MODEL_NAME=dinov2 FID_CENTER_SLICES_RATIO=${FID_R} DINO_REPO=${DINO_REPO:?set DINO_REPO} DINO_WEIGHT=${DINO_WEIGHT:?set DINO_WEIGHT} DINO_ARCH=${DINO_ARCH:-dinov2_vitl14}";;
    *) echo "unknown extractor $1" >&2; exit 1;;
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

# One row = reconstruct once (J1, PHASE=all, first extractor) then re-read those
# same recon_*.nii.gz for each remaining extractor. Rows are separated by OUT_TAG,
# NOT by POSTFIX: compute_metric.py globs recon_*.nii.gz for real_vs_recon
# (compute_metric.py:1071) and ignores POSTFIX entirely, so two rows sharing an
# OUT_TAG would silently score each other's volumes.
row() {  # row <arm> <decoder> <slice>
  local arm="$1" dec="$2" slice="$3"
  local csv="${SLICE_DIR}/${slice}.csv"
  [[ -f "${csv}" ]] || { echo "[skip] missing ${csv} (run scripts/build_gfid_slices.py)"; return; }

  local exp ckpt tag
  case "${dec}" in
    orig) exp="${arm}";               ckpt="${ORIG_CKPT}"; tag="orig${ORIG_CKPT##checkpoint-}";;
    dft)  exp="${arm}${DFT_SUFFIX}";  ckpt="${DFT_CKPT}";  tag="dft${DFT_CKPT##checkpoint-}";;
    *) echo "unknown decoder ${dec}" >&2; exit 1;;
  esac

  n_rows=$((n_rows+1))
  echo "[row] $(sed -e 's/^pooled-//' -e 's/-kl8e4-eff32-s1$//' <<<"${arm}")  ${dec}  ${slice}  n=${N}"

  # UNET_EXP_NAME stays on the stage1 arm even for the DFT decoder: DFT froze the
  # encoder, so analysis/latent_stats.csv (scale_factor, global_mean) is unchanged
  # and lives only there. compute_metric.sh reads it from UNET_EXP_DIR by design.
  local common="EXP_NAME=${exp} UNET_EXP_NAME=${arm} \
VAE_CKPT_NAME=${ckpt} STAGE=stage1 DATASET=pooled \
EVAL_MODE=real_vs_recon NUM_IMAGES=${N} \
BASE_CSV=${csv} CELL= OUT_TAG=${tag}_${slice} FID_BOOTSTRAP=${BOOT}"

  local jid
  jid=$(submit "" ${common} PHASE=all $(ext_env "${EXTS[0]}")); n_jobs=$((n_jobs+1))
  echo "    J1 recon + ${EXTS[0]}   jid=${jid}"
  local e
  for e in "${EXTS[@]:1}"; do
    jid=$(submit "--dependency=afterok:${jid}" ${common} PHASE=fid $(ext_env "${e}")); n_jobs=$((n_jobs+1))
    echo "    J+ ${e}  jid=${jid}"
  done
}

for arm in ${ARMS}; do
  for dec in ${DECODERS}; do
    for s in ${SLICES}; do
      row "${arm}" "${dec}" "${s}"
    done
  done
done

echo
echo "rows=${n_rows}  jobs=${n_jobs}  n=${N}  ratio=${FID_R}  extractors=${EXTS[*]}  (DRYRUN=${DRYRUN})"
echo "fills: journal_plan/results_pooled_recon_slices.csv"
echo "read : paired per (arm, slice) — orig vs dft. DFT is adopted only if it wins;"
echo "       a blurred conditional-mean decoder would show up here as worse."
