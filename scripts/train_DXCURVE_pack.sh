#!/bin/bash
# AD/MCI/CN data-scaling curve — five real-only models in ONE allocation, SEQUENTIALLY.
#   sbatch scripts/train_DXCURVE_pack.sh
#
# WHY THIS EXISTS. The augmentation experiment is ~16 models per task (real-only, four
# synthetic ratios x three baseline arms, plus TSTR), so it can only be run at ONE
# training-set size and that size has to be chosen before spending the compute. The
# tumour segmentor showed what happens otherwise: its WT dice was already 0.8957 at
# n=100 against 0.9256 at n=1000, so an augmentation arm run there would have returned
# a null for lack of headroom rather than for anything about the generator.
#
# WHY SEQUENTIAL, NOT CO-RESIDENT. Both clusters are cap=1, so the only parallelism is
# inside one allocation -- but SFCN at 192^3 measured ~26 GB per group and a fourth
# group OOM'd (adherence pack, 2026-09-04). Five points would not fit. They do fit in
# TIME: the dx pool is 5,603 volumes against brain-age's 47,820, so each point is short
# and five in series stay inside the walltime.
#
# STEP-MATCHED, NOT EPOCH-MATCHED. steps/epoch is n/(bs*gpus), so a shared epoch count
# gives the small-n arms a fraction of the optimizer steps and the curve then measures
# undertraining as if it were data scarcity. Epochs are derived from TARGET_STEPS here,
# exactly as scripts/train_SEG_pack.sh does.
#
# SUBSETS ARE STRATIFIED AND NESTED, built by scripts/make_dx_curve_subsets.py. Not
# --real_limit: that is df.head(n), and the dx pool is in manifest order (grouped by
# cohort), so head(n) is 100% ADNI up to n=2,500 -- the curve would confound cohort
# with n. See that script's header.
#
# SELECTION ON VALID, REPORTING ON TEST. Plan section 6 requires the downstream numbers
# to come from a held-out real test, because generation is conditioned on the same
# attribute the predictor reads back; scoring on the split that chose the checkpoint
# would close that loop. --test_csv scores the selected best.pt once at the end.
# CLUSTER: these SBATCH lines are AIBIO's (gpu-4farm, h100). If this runs on GSDS the
# partition and gres must be swapped for that cluster's -- the body is portable, the
# header is not. Both clusters are cap=1, so this competes with whatever is queued
# there: AIBIO currently holds the 3DMD Phase 2+3 pack and then Phase 4, GSDS holds the
# nobrats FID re-measurement and TSTR.
#SBATCH --job-name=dx_curve
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/downstream/dxcurve/dxcurve_%j.log
#SBATCH --open-mode=append
# NOT `set -u`: env.local.sh runs conda activate -> /etc/bashrc -> unbound
# $BASHRCSOURCED kills the shell before the first echo (job 263079, 2026-09-01).
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh

DATASET_CFG="configs/pooled/dataset.json"
ADH="${OUTPUT_ROOT}/downstream/adherence"
SUB_DIR="${ADH}/dx_curve"
OUT="${OUTPUT_ROOT}/downstream/dxcurve"
mkdir -p "${OUT}"

SIZES="${SIZES:-250 500 1000 2500 full}"
# 12,000, set from the judge's own curve rather than picked: dx_clf trained 60 epochs
# on the full 5,603 (350 steps/epoch = 21,000 steps) and peaked at epoch 31, i.e.
# ~10,850 steps. 12,000 clears that knee with margin, and the smaller-n points overfit
# sooner so it covers them too.
TARGET_STEPS="${TARGET_STEPS:-12000}"
# Validate every VAL_STEPS, not every epoch: n=250 is 15 steps/epoch = 800 epochs, and
# scoring the 689 validation volumes after each would cost more than the training.
VAL_STEPS="${VAL_STEPS:-1000}"
LABEL_MAP='{"healthy":0,"MCI":1,"AD":2}'
BS="${BS:-4}"; LR="${LR:-1e-3}"; WD="${WD:-1e-4}"; DROPOUT="${DROPOUT:-0.5}"
NG="${SLURM_GPUS_ON_NODE:-4}"
NW=$(( SLURM_CPUS_PER_TASK / NG )); [[ "${NW}" -lt 1 ]] && NW=1

VA="${ADH}/dx_clf/adh_dx_valid.csv"
TE="${SUB_DIR}/dx_test.csv"

echo "=== dx_curve job ${SLURM_JOB_ID} on $(hostname) @ $(date) ==="
echo "  sizes=${SIZES}  target_steps=${TARGET_STEPS}  gpus=${NG} workers=${NW}"
echo "  valid=$(( $(wc -l < "${VA}") - 1 ))  test=$(( $(wc -l < "${TE}") - 1 ))"

fail=0
for n in ${SIZES}; do
  tr="${SUB_DIR}/dx_train_n${n}.csv"
  [[ -f "${tr}" ]] || { echo "[preflight] MISSING ${tr}"; fail=1; }
done
[[ -f "${VA}" ]] || { echo "[preflight] MISSING ${VA}"; fail=1; }
[[ -f "${TE}" ]] || { echo "[preflight] MISSING ${TE}"; fail=1; }
[[ "${fail}" -ne 0 ]] && { echo "=== PREFLIGHT FAILED ==="; exit 1; }

START_TS=$(date +%s)
export NCCL_SHM_DISABLE=1 OMP_NUM_THREADS=1
CHILD=""
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): walltime approaching ==="
  [[ -n "${CHILD}" ]] && kill -TERM "${CHILD}" 2>/dev/null
}
trap on_term TERM USR1

for n in ${SIZES}; do
  name="dx_n${n}"
  exp="${OUT}/${name}"
  # Skip on a COMPLETION sentinel, never on best.pt: best.pt appears after the first
  # epoch that improves, so keying the skip on it would let a walltime resubmit mark a
  # part-trained point "done". train_attr_predictor.py resumes from weights/last.pt
  # (added 2026-09-11), so an interrupted point continues from its last full epoch;
  # the sentinel still decides what counts as finished.
  if [[ -f "${exp}/weights/.dxcurve_done" ]]; then
    echo "[${name}] sentinel present — skipping"; continue
  fi
  mkdir -p "${exp}/logs" "${exp}/weights"
  tr="${SUB_DIR}/dx_train_n${n}.csv"
  n_tr=$(( $(wc -l < "${tr}") - 1 ))
  spe=$(( n_tr / (BS * NG) )); [[ "${spe}" -lt 1 ]] && spe=1
  ep=$(( TARGET_STEPS / spe )); [[ "${ep}" -lt 1 ]] && ep=1
  ve=$(( (VAL_STEPS + spe - 1) / spe )); [[ "${ve}" -gt "${ep}" ]] && ve=${ep}
  echo "[${name}] n=${n_tr}  ${spe} steps/ep  epochs=${ep}  (=$(( spe * ep )) steps)  val_every=${ve}  @ $(date)"

  ( TORCHINDUCTOR_CACHE_DIR="${exp}/torchinductor" TRITON_CACHE_DIR="${exp}/triton" \
    torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29940 \
      -m downstream.train_attr_predictor \
        --dataset_config_path "${DATASET_CFG}" \
        --train_csv "${tr}" --valid_csv "${VA}" --test_csv "${TE}" \
        --data_dir "${DATA_DIR}" --output_dir "${OUT}" --run_name "${name}" \
        --target dx --label_map "${LABEL_MAP}" --class_weighted \
        --batch_size "${BS}" --num_workers "${NW}" --epochs "${ep}" --val_every "${ve}" \
        --lr "${LR}" --weight_decay "${WD}" --dropout "${DROPOUT}" \
      >> "${exp}/logs/${name}_${SLURM_JOB_ID}.log" 2>&1 ) &
  CHILD=$!
  wait "${CHILD}"; rc=$?
  # The sentinel needs test_metrics.json, which is written only after the run reaches
  # the end and scores the selected checkpoint. exit 0 alone is not enough: a walltime
  # kill can look clean while leaving the point part-trained.
  if [[ "${rc}" -eq 0 && -f "${exp}/test_metrics.json" ]]; then
    touch "${exp}/weights/.dxcurve_done"
    echo "[${name}] COMPLETE  $(python3 -c "
import json;d=json.load(open('${exp}/test_metrics.json'))
print('test balanced_acc=%.4f acc=%.4f (from epoch %d)'%(d['balanced_acc'],d['acc'],d['from_epoch']))")"
  else
    echo "[${name}] INCOMPLETE (rc=${rc})"
  fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
done_n=0; todo=0
for n in ${SIZES}; do
  if [[ -f "${OUT}/dx_n${n}/weights/.dxcurve_done" ]]; then done_n=$((done_n+1))
  else todo=$((todo+1)); echo "  dx_n${n}: INCOMPLETE"; fi
done
echo "=== done=${done_n} todo=${todo} elapsed=${ELAPSED}s ==="

if [[ "${todo}" -eq 0 ]]; then
  echo "=== dx_curve DONE ==="
  python3 - <<'EOF'
import json, os
OUT = os.environ.get("OUT", "/data/wonyoungjang/decodata/pooled/downstream/dxcurve")
print(f"{'n':>6} {'test_bacc':>10} {'test_acc':>9}")
for n in ("250", "500", "1000", "2500", "full"):
    p = f"{OUT}/dx_n{n}/test_metrics.json"
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"{n:>6} {d['balanced_acc']:>10.4f} {d['acc']:>9.4f}")
EOF
elif [[ "${ELAPSED}" -lt 1800 && "${done_n}" -eq 0 ]]; then
  # CRASH-LOOP GUARD — the resubmit is for walltime preemption; if nothing finished
  # and the job was short, resubmitting re-runs the same crash.
  echo "=== CRASH LOOP: nothing completed in ${ELAPSED}s. NOT resubmitting. ==="
else
  echo "=== self-resubmit: sbatch scripts/train_DXCURVE_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch scripts/train_DXCURVE_pack.sh
fi
exit 0
