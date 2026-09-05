#!/bin/bash
# Data-scaling ablation for the BraTS tumour segmentor — 3-into-1 co-resident.
# ONE sbatch trains real-only at n = 100 / 250 / 500, co-resident on the node's
# GPUs, to place the already-measured n=1000 point on a curve.
#   sbatch scripts/train_SEG_pack.sh
#
# WHAT THIS IS FOR. The deliverable is "does DeCo-VAE synthetic data improve
# segmentation", and that question is only answerable where real data is still the
# binding constraint. The n=1000 run reached WT 0.9256, which is already the level
# published BraTS work reports using the FULL 1,251-case corpus and all four
# modalities — so 1000 may well sit on the flat part of the curve, where adding
# synthetic volumes cannot raise the number no matter how good they are. Running the
# augmentation arm there would produce a null result that says nothing about the
# generator. This locates the steep part first, and it converts a future gain into
# an exchange rate ("a synthetic volume is worth X real ones") instead of a delta.
#
# STEPS ARE HELD FIXED, NOT EPOCHS — the whole point of the control. steps/epoch is
# n/(batch_size x gpus) = n/8, so a shared 200 epochs would give n=1000 25,000
# optimizer steps and n=100 only 2,400. The small-n arms would then underperform
# partly from undertraining, and the curve would exaggerate its own slope. Every arm
# here gets ~15,000 steps. That target is read off the n=1000 history: dice_mean was
# 0.7845 by epoch 100 (12,500 steps) against 0.7955 at its epoch-177 best, so 15,000
# is past the knee. The n=1000 point is NOT retrained — its history.jsonl already
# records every epoch, so the step-matched value is read from it at analysis time.
#
# NESTED SUBSETS. --real_limit is df.head(n) (seg_dataset.py::_build_records), so
# 100 c 250 c 500 c 1000 by construction. That is deliberate: disjoint draws would
# add subset-composition noise on top of the data-quantity effect this measures.
#
# VALIDATION IS DECOUPLED (--val_every). Step-matching means ~1,250 epochs at n=100,
# and validating each one -- 126 volumes x 18 sliding windows, redundantly on every
# rank -- would cost more than the training. Each arm gets ~20 validations instead.
#
# 56 CPUs, NOT the 112 the adherence pack asks for. This is a QUEUE decision, not a
# throughput one: at 112 the only eligible hosts were node23/24, because node25 and
# node27 had 64 and 80 idle cores, and job 264416 sat behind six higher-priority jobs
# for that reason. 56 doubles the candidate hosts. Throughput does not suffer -- the
# n=1000 baseline (job 264335) ran on 56 and was not CPU bound once the page cache
# warmed, and 56/(3 groups x 4 gpus) still leaves 4 loader workers per rank.
#SBATCH --job-name=seg_pack
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/brats/downstream/segpack/segpack_%j.log
#SBATCH --open-mode=append
# NOT `set -u`: env.local.sh runs conda activate -> /etc/bashrc -> unbound
# $BASHRCSOURCED kills the shell before the first echo (job 263079, 2026-09-01).
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=brats
source scripts/resolve_dataset.sh    # DATA_DIR, OUTPUT_ROOT, TRAIN_CSV, VALID_CSV

# The FLAIR per-modality CSVs, matching the n=1000 baseline (job 264335) exactly.
# resolve_dataset points at brats_train.csv, which is the T1n default.
TRAIN_CSV="${SCRIPT_DIR}/csv_files/brats_FLAIR_train.csv"
VALID_CSV="${SCRIPT_DIR}/csv_files/brats_FLAIR_valid.csv"
DATASET_CFG="configs/brats/dataset.json"
SEG_CFG="configs/brats/tumor_seg.json"
OUT="${OUTPUT_ROOT}/downstream/tumor_seg"
RUNDIR="${OUTPUT_ROOT}/downstream/segpack"; mkdir -p "${RUNDIR}" "${OUT}"

TARGET_STEPS="${TARGET_STEPS:-15000}"
N_VALIDATIONS="${N_VALIDATIONS:-20}"
# n : (epochs and val_every are DERIVED below from TARGET_STEPS, never hand-set)
LIMITS=(100 250 500)

read BS NW LR WD <<EOF
$(python3 -c "import json;c=json.load(open('${SEG_CFG}'));print(c['batch_size'],c['num_workers'],c['lr'],c['weight_decay'])")
EOF
NG="${SLURM_GPUS_ON_NODE:-4}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------
fail=0
declare -a TODO=()
for n in "${LIMITS[@]}"; do
  name="real_only_FLAIR_n${n}"
  exp="${OUT}/${name}"; mkdir -p "${exp}/logs" "${exp}/weights"
  # Skip on a COMPLETION sentinel, never on best.pt: best.pt is written the first
  # time dice improves, so keying the skip on it would let a walltime resubmit mark
  # a partially-trained arm "done". train_tumor_seg.py has no resume, so an
  # interrupted arm must restart from scratch and the sentinel is what forces that.
  if [[ -f "${exp}/weights/.segpack_done" ]]; then
    echo "[preflight] ${name}: already complete — skipping"; continue
  fi
  spe=$(( n / (BS * NG) ))                       # steps per epoch
  [[ "${spe}" -lt 1 ]] && { echo "  !! n=${n} < bs*gpus=$((BS*NG))"; fail=1; continue; }
  ep=$(( TARGET_STEPS / spe ))
  ve=$(( ep / N_VALIDATIONS )); [[ "${ve}" -lt 1 ]] && ve=1
  n_avail=$(( $(wc -l < "${TRAIN_CSV}") - 1 ))
  [[ "${n}" -gt "${n_avail}" ]] && { echo "  !! n=${n} > available ${n_avail}"; fail=1; }
  printf "[preflight] %-22s n=%4d  %3d steps/ep  epochs=%5d  (=%6d steps)  val_every=%3d\n" \
         "${name}" "${n}" "${spe}" "${ep}" "$((spe*ep))" "${ve}"
  TODO+=("${n}:${ep}:${ve}")
done
if [[ "${fail}" -ne 0 ]]; then echo "=== PREFLIGHT FAILED — not launching. ==="; exit 1; fi
if [[ "${#TODO[@]}" -eq 0 ]]; then echo "=== all arms already trained; nothing to do. ==="; exit 0; fi
echo "[preflight] train pool=$(( $(wc -l < "${TRAIN_CSV}") - 1 ))  valid=$(( $(wc -l < "${VALID_CSV}") - 1 ))"
echo "[preflight] OK — ${#TODO[@]} arm(s) to train"

NGRP=${#TODO[@]}
NWK=$(( SLURM_CPUS_PER_TASK / (NGRP * NG) )); [[ "${NWK}" -lt 1 ]] && NWK=1
echo "[preflight] groups=${NGRP} gpus=${NG} num_workers=${NWK} per rank"

START_TS=$(date +%s)
export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS=1
STAGGER_SEC=45
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then sleep 3; echo "[MPS] daemon up"
else echo "[MPS] FAILED to start — co-residency will likely deadlock"; fi
echo "=== segpack job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP} ==="

( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

PIDS=(); PORT=29810
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1

for spec in "${TODO[@]}"; do
  IFS=':' read -r n ep ve <<< "${spec}"
  name="real_only_FLAIR_n${n}"
  exp="${OUT}/${name}"
  # The sentinel is touched only when torchrun EXITS 0, so a walltime kill leaves it
  # absent and the resubmit retrains this arm from the start.
  ( TORCHINDUCTOR_CACHE_DIR="${exp}/torchinductor" TRITON_CACHE_DIR="${exp}/triton" \
    torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port="${PORT}" \
      -m downstream.train_tumor_seg \
        --dataset_config_path "${DATASET_CFG}" --seg_config_path "${SEG_CFG}" \
        --train_csv "${TRAIN_CSV}" --valid_csv "${VALID_CSV}" \
        --data_dir "${DATA_DIR}" --output_dir "${OUT}" --run_name "${name}" \
        --real_limit "${n}" --epochs "${ep}" --val_every "${ve}" \
        --batch_size "${BS}" --num_workers "${NWK}" --lr "${LR}" --weight_decay "${WD}" \
      >> "${exp}/logs/${name}_segpack_${SLURM_JOB_ID}.log" 2>&1 \
    && touch "${exp}/weights/.segpack_done" ) &
  PIDS+=($!)
  echo "launched ${name} (pid $!, n=${n}, epochs=${ep}, val_every=${ve}) port ${PORT}; staggering ${STAGGER_SEC}s"
  PORT=$((PORT+10))
  sleep "${STAGGER_SEC}"
done

for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p" 2>/dev/null || true; done
done
kill "${SMI}" 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null && echo "[MPS] daemon stopped"

need_more=0; done_n=0
for spec in "${TODO[@]}"; do
  IFS=':' read -r n _ _ <<< "${spec}"
  name="real_only_FLAIR_n${n}"
  if [[ -f "${OUT}/${name}/weights/.segpack_done" ]]; then
    echo "  ${name}: complete"; done_n=$((done_n+1))
  else
    echo "  ${name}: INCOMPLETE (no resume — will retrain from scratch)"; need_more=1
  fi
done

# CRASH-LOOP GUARD — the resubmit exists for walltime preemption. If nothing
# finished and the job was short, resubmitting just re-runs the same crash; the UNet
# pack burned 5 allocations in 20 min that way on 2026-08-28.
ELAPSED=$(( $(date +%s) - START_TS ))
if [ "${need_more}" -eq 1 ] && [ "${done_n}" -eq 0 ] && [ "${ELAPSED}" -lt 1800 ]; then
  echo "=== CRASH LOOP: no arm finished and the job lasted only ${ELAPSED}s."
  echo "=== NOT resubmitting. Read ${OUT}/real_only_FLAIR_n*/logs/*_segpack_${SLURM_JOB_ID}.log ==="
elif [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (an arm is unfinished): sbatch scripts/train_SEG_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_SEG_pack.sh"
else
  echo "=== all arms trained; no resubmit ==="
fi
echo "=== segpack DONE ==="
exit 0
