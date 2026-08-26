#!/bin/bash
# Convergence-trajectory eval for the pooled kl8e4/eff32 triad (GSDS driver).
#
# Answers "where does each arm converge, and do the arms rank differently at
# different points in training?" — the 320k-only snapshot we had could not tell a
# plateau from a lucky checkpoint (maisi LPIPS dipped to 0.0275 at 240k and
# bounced straight back to 0.0299 at 260k).
#
# Grid: 3 arms x 11 checkpoints x 1 cell x 5 jobs = 165 jobs.
#   Trajectory SHAPE was identical in all 14 cells for maisi, so one cell is
#   enough here; run the full per-cell sweep only at the checkpoint this picks.
#
# Each row is a 5-job afterok chain that generates the recon volumes ONCE (J1)
# and then re-reads them for each extractor (J2..J5). PHASE=fid globs
# gen_*_<POSTFIX>.nii.gz, so every job in a row MUST share POSTFIX and rows must
# not — hence POSTFIX=tj_<arm>_<step>.
#
# Usage:
#   DRYRUN=1 bash scripts/eval_triad_trajectory.sh          # print, submit nothing
#   SWAV_WEIGHT=... DINO_REPO=... DINO_WEIGHT=... \
#     bash scripts/eval_triad_trajectory.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CM="${SCRIPT_DIR}/compute_metric.sh"
STAGE_ROOT="${STAGE_ROOT:-${POOLED_OUTPUT_ROOT:?set POOLED_OUTPUT_ROOT or STAGE_ROOT}/stage1}"

# --- grid --------------------------------------------------------------------
ARMS="${ARMS:-pooled-maisi-kl8e4-eff32-s1 pooled-sid-cor50-kl8e4-eff32-s1 pooled-vad-cov1var1-kl8e4-eff32-s1}"
# 60k/80k close the hole below the old grid: maisi's PSNR peaked at 100k, which
# was simply the first point ever measured, so the rise was never observed.
STEPS="${STEPS:-60000 80000 100000 120000 140000 160000 200000 240000 260000 280000 320000}"
CELL="${CELL:-ukb_T1}"
SPLIT="${SPLIT:-test}"          # matches results_pooled_recon_s1_cells.csv
NUM_IMAGES="${NUM_IMAGES:-500}"
FID_R="${FID_R:-0.4}"
CONTENT_FRAC="${CONTENT_FRAC:-0.15}"

# --- background-excluded SSIM ------------------------------------------------
# Whole-volume SSIM is unusable on these arms: sid-cor50 hcp_T1 scored 0.5242 vs
# maisi 0.9698 while LPIPS (0.0349 vs 0.0350) and PSNR (30.10 vs 30.35) were
# effectively identical. A synthetic sweep reproduces exactly that decoupling —
# a background residue of ~1-3% of the intensity range drives whole-volume SSIM
# from 0.999 to 0.27 while the foreground-masked value never leaves 0.99.
SSIM_FG="${SSIM_FG:-1}"
SSIM_FG_ERODE="${SSIM_FG_ERODE:-3}"   # SSIM's 7^3 window straddles the brain edge
SSIM_FG_THRESH="${SSIM_FG_THRESH:-1e-6}"

DRYRUN="${DRYRUN:-0}"

gen_env() { echo "PHASE=all FID_MODEL_NAME=radimagenet_resnet50 FID_CENTER_SLICES_RATIO=${FID_R}"; }
ext_env() {
  case "$1" in
    inception) echo "PHASE=fid FID_MODEL_NAME=imagenet_inception FID_CENTER_SLICES_RATIO=${FID_R}";;
    swav)      echo "PHASE=fid FID_MODEL_NAME=imagenet_swav FID_WOODLAND=1 FID_CONTENT_FRAC=${CONTENT_FRAC} SWAV_WEIGHT=${SWAV_WEIGHT:?set SWAV_WEIGHT}";;
    dinov2)    echo "PHASE=fid FID_MODEL_NAME=dinov2 FID_CENTER_SLICES_RATIO=${FID_R} DINO_REPO=${DINO_REPO:?set DINO_REPO} DINO_WEIGHT=${DINO_WEIGHT:?set DINO_WEIGHT} DINO_ARCH=${DINO_ARCH:-dinov2_vitl14}";;
    rad10)     echo "PHASE=fid FID_MODEL_NAME=radimagenet_resnet50 FID_CENTER_SLICES_RATIO=1.0";;
  esac
}
# rad10 is deliberately absent: r=1.0 measured 19.28 vs 19.29 for models whose
# SSIM was 0.315 vs 0.966, so it carries no quality signal. Add "rad10" here only
# to reproduce the historical column.
EXTS=(inception swav dinov2)

submit() {
  local dep="$1"; shift
  if [[ "${DRYRUN}" == "1" ]]; then
    echo "    [dryrun] ${dep:+$dep }env $* sbatch ${CM}" >&2
    echo "$(( RANDOM + 100000 ))"
  else
    env "$@" sbatch --parsable ${dep:+$dep} "${CM}"
  fi
}

n_rows=0 n_jobs=0 n_skip=0
for arm in ${ARMS}; do
  short="$(sed -e 's/^pooled-//' -e 's/-kl8e4-eff32-s1$//' <<<"${arm}")"
  for step in ${STEPS}; do
    ck="${STAGE_ROOT}/${arm}/weights/vae/checkpoint-${step}/model.pt"
    if [[ ! -f "${ck}" ]]; then echo "[skip] missing ${ck}"; n_skip=$((n_skip+1)); continue; fi
    n_rows=$((n_rows+1))
    echo "[row] ${short}  ck${step}"
    common="EXP_NAME=${arm} STAGE=stage1 DATASET=pooled VAE_CKPT_NAME=checkpoint-${step} \
EVAL_MODE=real_vs_recon DETERMINISTIC=1 NUM_IMAGES=${NUM_IMAGES} CELL=${CELL} SPLIT=${SPLIT} \
POSTFIX=tj_${short}_${step} \
SSIM_FG=${SSIM_FG} SSIM_FG_ERODE=${SSIM_FG_ERODE} SSIM_FG_THRESH=${SSIM_FG_THRESH}"
    jid=$(submit "" ${common} $(gen_env)); n_jobs=$((n_jobs+1))
    echo "    J1 rad r=${FID_R} + paired(+ssim_fg)  jid=${jid}"
    for e in "${EXTS[@]}"; do
      jid=$(submit "--dependency=afterok:${jid}" ${common} $(ext_env "${e}")); n_jobs=$((n_jobs+1))
      echo "    J+ ${e}  jid=${jid}"
    done
  done
done

echo
echo "rows=${n_rows}  jobs=${n_jobs}  skipped=${n_skip}  (DRYRUN=${DRYRUN})"
cat <<'NOTE'

Disk: each row keeps 500 recon volumes alive for its whole chain (J2..J5 re-read
them). At ~5 MB/volume that is ~2.5 GB per row and ~80 GB if all 33 rows are
retained. After a row's last job completes, drop its volumes:
  rm -rf <STAGE_ROOT>/<arm>/cells/ukb_T1_test/outputs/volumes/*tj_<short>_<step>*

Harvest (ssim_fg is a print(), the rest are logger lines carrying an
"INFO:compute_metric:" prefix — do not anchor patterns with ^):
  awk '$1=="EXP_NAME"{en=$3} $1=="POSTFIX"{pf=$3} /FID model /{ex=$NF}
       /enable_center_slices/{r=$NF} /woodland_exact/{wl=$3}
       $1=="LPIPS"{lp=$3} $1=="PSNR"{ps=$3} $1=="SSIM"{ss=$3} $1=="SSIM_FG:"{sf=$2}
       /FID Avg:/{fid=$NF}
       END{if(fid!="")printf "%-34s %-22s ext=%-20s r=%-5s wl=%-5s FID=%-9s LPIPS=%-8s PSNR=%-8s SSIM=%-8s SSIM_FG=%s\n",
           en,pf,ex,r,wl,fid,lp,ps,ss,sf}' <log>
  # NOTE: awk reserves `exp` (exponential), hence `en`/`ex`.
NOTE
