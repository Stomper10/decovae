#!/bin/bash
# 3D MedDiff Phase 2 (decoder FT) + Phase 3 (latent extraction) in ONE allocation.
#   sbatch scripts/train_3DMD_pack.sh
#
# WHY PACK. The two phases are sequential -- Phase 3 encodes with the model Phase 2
# produces -- so this is not parallelism, it is one queue wait instead of two. That
# is the dominant cost: gpu-4farm has been taking about a day to admit a job this
# week, against 10h for Phase 2 and a measured 1h19m for Phase 3 (52,169 train +
# 6,493 val on 4 GPUs, job 221275). Both fit inside one 24h walltime with margin.
#
# EXP_NAME=pooled_s2, AND THAT IS LOAD-BEARING. train_3d_meddiff.sh defaults
# EXP_NAME to ukb_c4 and auto-resumes from the highest-step checkpoint under
# ${EXP_ROOT}/my_model. Submitting stage2 without setting it (job 265880) would have
# found ukb_c4's step=300000 UKB checkpoint and injected it as
# resume_from_checkpoint -- which takes priority over init_weights_from -- so the
# pooled Phase 2 would have silently continued a different dataset's model. The
# launcher's own comment says to use a distinct EXP_NAME per stage; this honours it.
#
# PHASE 3 READS PHASE 2'S ENDPOINT, not its val-best. Our DFT used each arm's
# checkpoint-40000 (the endpoint of a 40k run), and Phase 2 is the analog of DFT, so
# the endpoint is the symmetric choice. The val-best slots (latest_checkpoint*) are
# rolling and would also drift if Phase 2 ever reruns.
#
# THE HEADER IS COPIED FROM train_3d_meddiff.sh, NOT from the ADH/SEG packs. Those
# are 4-GPU gpu-4farm jobs and every directive that differs here came from taking
# them as the template: partition, gres, ntasks-per-node and cpus-per-task were all
# wrong at first. ntasks-per-node=8 is the one that would have hung rather than
# failed loudly -- srun would spawn a single task while the yaml asks PL for 8
# devices. The two phases want different values (Phase 3's own launcher uses
# ntasks=1 with no srun) but 8 serves both: the batch script body runs once
# regardless of ntasks, so Phase 3's plain background `python &` is unaffected,
# while Phase 2's `srun` picks up all 8. cpus-per-task=14 then gives BOTH phases the
# same per-process CPU budget their own launchers use (Phase 3: 56/4 = 14).
#
# gpu-8farm AND 8 GPUs, BOTH MATCHING PHASE 1 (train_3d_meddiff.sh). The yaml says
# `gpus: 8`, and allocating 4 against that is the mismatch train_3d_meddiff.sh warns
# about ("make sure the yaml's gpus matches --gres"). The partition matters just as
# much: both farms have 8-GPU nodes, but by convention 8farm's queue is entirely
# gres/gpu:8 requests while 4farm's is gres/gpu:4. An 8-GPU job on 4farm is the worst
# of both -- it waits for a WHOLE node to drain while 4-GPU jobs keep filling the
# half-nodes it cannot use. This header was copied from the ADH/SEG packs, which are
# 4-GPU 4farm jobs; the 3DMD lineage is 8farm.
# It also keeps GPU-hours comparable across phases, which the plan requires of the
# cross-architecture baseline, and it keeps Phase 2 inside the walltime: Phase 1
# measured 205 steps/min on 8 GPUs, and Phase 2 is slower per step (batch_size 1,
# whole volume), so 40k steps is ~10h at 8 GPUs but ~20h at 4 -- which plus Phase 3
# leaves no margin under 24h. Phase 3 shards across all 8 as well, halving its
# measured 1h19m.
#
# LATENTS GO TO pooled_s2/latents, NOT pooled/latents. The latter holds Phase 1's
# 174 GB of stage1 latents; overwriting them would destroy the only artefact that
# could reproduce the earlier state.
#SBATCH --job-name=3dmd_pack
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=14
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/3d_meddiff/pooled_s2/logs/3dmd_pack_%j.log
#SBATCH --open-mode=append
# NOT `set -u`: env.local.sh runs conda activate -> /etc/bashrc -> unbound
# $BASHRCSOURCED kills the shell before the first echo (job 263079, 2026-09-01).
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc
conda activate 3d_meddiff

EXP_NAME="${EXP_NAME:-pooled_s2}"
CONFIG="${CONFIG:-configs/3d_meddiff/PatchVolume_4x_pooled_s2.yaml}"
DATA_JSON="${DATA_JSON:-configs/3d_meddiff/data_pooled.json}"
SPLITS="${SPLITS:-train val}"
NUM_SHARDS="${NUM_SHARDS:-8}"

EXP_ROOT="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}"
LOGS_DIR="${EXP_ROOT}/logs"
OUT_DIR="${OUT_DIR:-${EXP_ROOT}/latents}"
mkdir -p "${LOGS_DIR}" "${OUT_DIR}"

TARGET_STEPS=$(python3 -c "
from omegaconf import OmegaConf; print(int(OmegaConf.load('${CONFIG}').model.max_steps))")

echo "=== 3dmd_pack job ${SLURM_JOB_ID} on $(hostname) @ $(date) ==="
echo "  exp_name     : ${EXP_NAME}   (NOT the ukb_c4 default -- see header)"
echo "  config       : ${CONFIG}"
echo "  target_steps : ${TARGET_STEPS}"
echo "  out_dir      : ${OUT_DIR}"

START_TS=$(date +%s)

# --requeue does NOT cover walltime expiry (TIMEOUT); that lesson cost a run on
# 2026-06-21. The trap gives us 180s to let the srun child die gracefully -- PL has
# been checkpointing every 3,000 steps, so the next allocation resumes from there --
# and then falls through to the resubmit block at the bottom.
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): walltime approaching, letting the child exit ==="
  [[ -n "${CHILD}" ]] && kill -TERM "${CHILD}" 2>/dev/null
}
trap on_term TERM USR1

# ---------------------------------------------------------------------------
# PHASE 2 — decoder fine-tune
# ---------------------------------------------------------------------------
if [[ -f "${EXP_ROOT}/.phase2_done" ]]; then
  echo "[phase2] sentinel present — skipping"
else
  # Resume from OUR OWN highest-step ckpt if one exists, scoped to EXP_ROOT so a
  # different experiment's checkpoints can never be picked up. Absent that, the
  # yaml's init_weights_from (stage1 step=100,000) is used for a fresh start.
  LATEST=$(find "${EXP_ROOT}/my_model" -maxdepth 5 -type f -name '*.ckpt' 2>/dev/null \
           | awk -F 'step=' 'NF>1 { n=$2; sub(/[^0-9].*/,"",n); print n"\t"$0 }' \
           | sort -k1,1n | tail -n 1 | cut -f2-)
  RUN_CONFIG="${CONFIG}"
  if [[ -n "${LATEST}" ]]; then
    RUN_CONFIG="${LOGS_DIR}/config_resume_${SLURM_JOB_ID}.yaml"
    python3 -c "
from omegaconf import OmegaConf
c = OmegaConf.load('${CONFIG}'); c.model.resume_from_checkpoint = '${LATEST}'
OmegaConf.save(c, '${RUN_CONFIG}')"
    echo "[phase2] resume: ${LATEST}"
  else
    echo "[phase2] fresh start from init_weights_from (stage1 step=100,000)"
  fi

  export WANDB_PROJECT_3DMD="${WANDB_PROJECT_3DMD:-decovae-3dmeddiff}"
  export WANDB_RUN_NAME="${EXP_NAME}" WANDB_RUN_ID="3dmd-${EXP_NAME}" WANDB_RESUME=allow
  srun python external/3d_meddiff/train/train_PatchVolume_stage2.py --config "${RUN_CONFIG}" &
  CHILD=$!
  wait "${CHILD}"; rc=$?
  echo "[phase2] exit=${rc} @ $(date)"
  # The sentinel is keyed on REACHING TARGET_STEPS, not on exit 0: a walltime kill
  # can leave a clean-looking exit while training is only part-way, and Phase 3
  # would then encode with an under-trained decoder and look fine.
  REACHED=$(find "${EXP_ROOT}/my_model" -maxdepth 5 -type f -name '*.ckpt' 2>/dev/null \
            | awk -F 'step=' 'NF>1 { n=$2; sub(/[^0-9].*/,"",n); print n }' \
            | sort -n | tail -n 1)
  REACHED="${REACHED:-0}"
  echo "[phase2] highest step on disk: ${REACHED} / ${TARGET_STEPS}"
  if [[ "${REACHED}" -ge "${TARGET_STEPS}" ]]; then
    touch "${EXP_ROOT}/.phase2_done"; echo "[phase2] COMPLETE"
  else
    echo "[phase2] INCOMPLETE — phase 3 will not run this allocation"
  fi
fi

# ---------------------------------------------------------------------------
# PHASE 3 — latent extraction, from Phase 2's ENDPOINT checkpoint
# ---------------------------------------------------------------------------
if [[ ! -f "${EXP_ROOT}/.phase2_done" ]]; then
  echo "[phase3] skipped — phase 2 not complete"
elif [[ -f "${EXP_ROOT}/.phase3_done" ]]; then
  echo "[phase3] sentinel present — skipping"
else
  AE_CKPT=$(find "${EXP_ROOT}/my_model" -maxdepth 5 -type f -name '*.ckpt' 2>/dev/null \
            | awk -F 'step=' -v t="${TARGET_STEPS}" 'NF>1 { n=$2; sub(/[^0-9].*/,"",n);
                if (n+0 == t+0) print $0 }' | head -n 1)
  if [[ -z "${AE_CKPT}" ]]; then
    echo "[phase3] FATAL: no checkpoint at step=${TARGET_STEPS}. Not falling back to"
    echo "[phase3] latest_checkpoint.ckpt -- that is a rolling val-best slot, and"
    echo "[phase3] silently encoding with the wrong model is the failure this guards."
  else
    echo "[phase3] ae_ckpt: ${AE_CKPT}"
    for split in ${SPLITS}; do
      echo "[phase3] split=${split} @ $(date)"
      for ((s = 0; s < NUM_SHARDS; s++)); do
        CUDA_VISIBLE_DEVICES="${s}" python scripts/extract_3d_meddiff_latents.py \
            --data-json "${DATA_JSON}" --split "${split}" \
            --ae-ckpt "${AE_CKPT}" --out-dir "${OUT_DIR}" \
            --shard-index "${s}" --num-shards "${NUM_SHARDS}" &
      done
      wait
    done
    n_lat=$(find "${OUT_DIR}" -name '*.npy' 2>/dev/null | wc -l)
    # Expected count comes from the data json, not a hard-coded guess. data_pooled.json
    # is train 52,169 + val 112 = 52,281 -- the val split here is 3DMD's own small
    # in-training validation set, NOT our pooled valid of 6,493. A threshold written
    # against 6,493 would never be met, the sentinel would never fire, and because the
    # job runs for hours the crash-loop guard would not catch it either: infinite
    # resubmit. Allow a 1% shortfall for volumes that fail to load, as some did in
    # Phase 1.
    N_EXP=$(python3 -c "
import json; d=json.load(open('${DATA_JSON}'))
print(sum(len(d[s]) for s in '${SPLITS}'.split()))")
    N_MIN=$(( N_EXP * 99 / 100 ))
    echo "[phase3] latents on disk: ${n_lat} / expected ${N_EXP} (min ${N_MIN})"
    if [[ "${n_lat}" -ge "${N_MIN}" ]]; then
      touch "${EXP_ROOT}/.phase3_done"; echo "[phase3] COMPLETE"
    else
      echo "[phase3] INCOMPLETE — expected ${N_EXP}, got ${n_lat}"
    fi
  fi
fi

# ---------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START_TS ))
p2=$([[ -f "${EXP_ROOT}/.phase2_done" ]] && echo done || echo INCOMPLETE)
p3=$([[ -f "${EXP_ROOT}/.phase3_done" ]] && echo done || echo INCOMPLETE)
echo "=== phase2=${p2}  phase3=${p3}  elapsed=${ELAPSED}s ==="

if [[ "${p2}" == "done" && "${p3}" == "done" ]]; then
  echo "=== 3dmd_pack DONE — next is Phase 4 (BiFlowNet) ==="
elif [[ "${ELAPSED}" -lt 1800 ]]; then
  # CRASH-LOOP GUARD. The resubmit exists for walltime preemption; if nothing
  # progressed and the job was short, resubmitting re-runs the same crash. The UNet
  # pack burned 5 allocations in 20 min that way on 2026-08-28.
  echo "=== CRASH LOOP: no phase completed and the job lasted only ${ELAPSED}s."
  echo "=== NOT resubmitting. Read ${LOGS_DIR}/3dmd_pack_${SLURM_JOB_ID}.log ==="
else
  echo "=== self-resubmit: sbatch scripts/train_3DMD_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch scripts/train_3DMD_pack.sh
fi
exit 0
