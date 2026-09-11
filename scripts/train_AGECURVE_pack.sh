#!/bin/bash
# Brain-age data-scaling curve — six real-only regressors in ONE allocation, SEQUENTIALLY.
#   sbatch scripts/train_AGECURVE_pack.sh
#
# WHY. Same reason as scripts/train_AGECURVE_pack.sh: the augmentation grid is ~13
# models per task, so it runs at ONE real training-set size, and that size must have
# headroom. Brain age is the task most likely to have none at full size -- the pool is
# 46,038 volumes -- which is exactly what this curve checks.
#
# SIZES 250..10,000, NOT FULL. Step-matched points all cost the same, and a full-pool
# point would need the judge's own budget: age_reg (47,820 vols, 4 GPUs x bs 4 = 2,989
# steps/epoch) flattened at epoch 38-41, ~120k steps, 4x everything else here. The
# already-trained age_reg is the full-size reference instead (val MAE 2.97; it includes
# hcp and was selected on validation, so it is a ceiling marker, not a curve point).
#
# TARGET_STEPS 30,000 covers the largest point: n=10,000 is 625 steps/epoch, i.e. 48
# epochs. If a point's best epoch is its last, it was still improving when the budget
# ran out; the COMPLETE line says so, and that point needs more steps.
#
# VALIDATION EVERY VAL_STEPS, NOT EVERY EPOCH. n=250 is 15 steps/epoch = 2,000 epochs;
# scoring 5,727 validation volumes after each would cost far more than training.
#
# Subsets: scripts/make_age_curve_subsets.py (cohort x modality x age quintile,
# nested). Validation / test: the 4-cohort CSVs built for TSTR v2. Selection on
# validation, reporting on test.
#SBATCH --job-name=agecurve
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/downstream/agecurve/agecurve_%j.log
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
SUB_DIR="${ADH}/age_curve"
OUT="${OUTPUT_ROOT}/downstream/agecurve"
mkdir -p "${OUT}"

SIZES="${SIZES:-250 500 1000 2500 5000 10000}"
TARGET_STEPS="${TARGET_STEPS:-30000}"
VAL_STEPS="${VAL_STEPS:-2000}"
BS="${BS:-4}"; LR="${LR:-1e-3}"; WD="${WD:-1e-4}"; DROPOUT="${DROPOUT:-0.5}"
NG="${SLURM_GPUS_ON_NODE:-4}"
NW=$(( SLURM_CPUS_PER_TASK / NG )); [[ "${NW}" -lt 1 ]] && NW=1

VA="${OUTPUT_ROOT}/downstream/tstr/csv/age_valid_4co.csv"
TE="${OUTPUT_ROOT}/downstream/tstr/csv/age_test_4co.csv"

echo "=== age_curve job ${SLURM_JOB_ID} on $(hostname) @ $(date) ==="
echo "  sizes=${SIZES}  target_steps=${TARGET_STEPS}  gpus=${NG} workers=${NW}"
echo "  valid=$(( $(wc -l < "${VA}") - 1 ))  test=$(( $(wc -l < "${TE}") - 1 ))"

fail=0
for n in ${SIZES}; do
  tr="${SUB_DIR}/age_train_n${n}.csv"
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
  name="age_n${n}"
  exp="${OUT}/${name}"
  # Skip on a COMPLETION sentinel, never on best.pt: best.pt appears after the first
  # epoch that improves, so keying the skip on it would let a walltime resubmit mark a
  # part-trained point "done". The trainer resumes from weights/last.pt, so an
  # interrupted point continues from its last validation; the sentinel decides what
  # counts as finished.
  if [[ -f "${exp}/weights/.agecurve_done" ]]; then
    echo "[${name}] sentinel present — skipping"; continue
  fi
  mkdir -p "${exp}/logs" "${exp}/weights"
  tr="${SUB_DIR}/age_train_n${n}.csv"
  n_tr=$(( $(wc -l < "${tr}") - 1 ))
  spe=$(( n_tr / (BS * NG) )); [[ "${spe}" -lt 1 ]] && spe=1
  ep=$(( TARGET_STEPS / spe )); [[ "${ep}" -lt 1 ]] && ep=1
  ve=$(( (VAL_STEPS + spe - 1) / spe )); [[ "${ve}" -gt "${ep}" ]] && ve=${ep}
  echo "[${name}] n=${n_tr}  ${spe} steps/ep  epochs=${ep}  (=$(( spe * ep )) steps)  val_every=${ve}  @ $(date)"

  ( TORCHINDUCTOR_CACHE_DIR="${exp}/torchinductor" TRITON_CACHE_DIR="${exp}/triton" \
    torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29941 \
      -m downstream.train_brain_age \
        --dataset_config_path "${DATASET_CFG}" \
        --train_csv "${tr}" --valid_csv "${VA}" --test_csv "${TE}" \
        --data_dir "${DATA_DIR}" --output_dir "${OUT}" --run_name "${name}" \
        --batch_size "${BS}" --num_workers "${NW}" --epochs "${ep}" --val_every "${ve}" \
        --lr "${LR}" --weight_decay "${WD}" --dropout "${DROPOUT}" \
      >> "${exp}/logs/${name}_${SLURM_JOB_ID}.log" 2>&1 ) &
  CHILD=$!
  wait "${CHILD}"; rc=$?
  # The sentinel needs test_metrics.json, which is written only after the run reaches
  # the end and scores the selected checkpoint. exit 0 alone is not enough: a walltime
  # kill can look clean while leaving the point part-trained.
  if [[ "${rc}" -eq 0 && -f "${exp}/test_metrics.json" ]]; then
    touch "${exp}/weights/.agecurve_done"
    echo "[${name}] COMPLETE  $(python3 -c "
import json;d=json.load(open('${exp}/test_metrics.json'))
print('test mae=%.3f r2=%.3f (best epoch %d of %d)%s'%(d['mae'],d['r2'],d['from_epoch'],${ep},'  <- best is the LAST epoch: still improving, needs more steps' if d['from_epoch']==${ep}-1 else ''))")"
  else
    echo "[${name}] INCOMPLETE (rc=${rc})"
  fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
done_n=0; todo=0
for n in ${SIZES}; do
  if [[ -f "${OUT}/age_n${n}/weights/.agecurve_done" ]]; then done_n=$((done_n+1))
  else todo=$((todo+1)); echo "  age_n${n}: INCOMPLETE"; fi
done
echo "=== done=${done_n} todo=${todo} elapsed=${ELAPSED}s ==="

if [[ "${todo}" -eq 0 ]]; then
  echo "=== age_curve DONE ==="
  python3 - <<'EOF'
import json, os
OUT = os.environ.get("OUT", "/data/wonyoungjang/decodata/pooled/downstream/agecurve")
print(f"{'n':>6} {'test_mae':>9} {'test_r2':>8}")
for n in ("250", "500", "1000", "2500", "5000", "10000"):
    p = f"{OUT}/age_n{n}/test_metrics.json"
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"{n:>6} {d['mae']:>9.3f} {d['r2']:>8.3f}")
EOF
elif [[ "${ELAPSED}" -lt 1800 && "${done_n}" -eq 0 ]]; then
  # CRASH-LOOP GUARD — the resubmit is for walltime preemption; if nothing finished
  # and the job was short, resubmitting re-runs the same crash.
  echo "=== CRASH LOOP: nothing completed in ${ELAPSED}s. NOT resubmitting. ==="
else
  echo "=== self-resubmit: sbatch scripts/train_AGECURVE_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch scripts/train_AGECURVE_pack.sh
fi
exit 0
