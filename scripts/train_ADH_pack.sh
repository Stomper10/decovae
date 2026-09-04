#!/bin/bash
# PRODUCTION 4-into-1 co-resident adherence-predictor launcher.
# ONE sbatch trains all four judges — modality / sex / dx classifiers and the age
# regressor — co-resident on the node's GPUs. Per-group skip when best.pt already
# exists, so this is idempotent and safe to resubmit.
#   sbatch scripts/train_ADH_pack.sh
#
# WHY A PACK: the cluster cap is ONE running job across gpu-8farm and gpu-4farm
# combined, so four separate submissions would mean four separate queue waits — and
# those have run 16h and 47h this week. Packing turns four waits into one.
#
# WHY gpu-4farm AND NOT 8farm, against the usual instinct:
#   SFCNClassifier is 2.95M parameters, ~1/60 of the diffusion UNet. Four of them do
#   not need eight H100s. What this workload actually needs is disk and CPU: 51,169
#   volumes x 60 epochs is ~3M reads of a 28 MB array, so it is I/O bound and the GPU
#   idles waiting. 4farm has FIVE nodes (23-27) against 8farm's two (21-22), so the
#   queue is far shorter, and a 4-GPU job can still request the full 112 CPUs.
#
# EPOCHS ARE PER TARGET, not the shared 60 the single-target launcher defaults to.
# That default came from the brain-age regressor. Modality is a 3-way call between
# T1/T2/FLAIR, which differ in gross contrast — it converges in a fraction of that,
# and 51k x 60 epochs would likely not even fit the 12h wall.
#
# CLASS WEIGHTING IS NOT OPTIONAL for modality and dx. Modality train is
# T1 26,146 / FLAIR 21,987 / T2 3,036 — an 8.6x imbalance. Unweighted, the classifier
# can score well while abandoning T2 entirely, which is precisely the class whose
# adherence we most need to measure.
#SBATCH --job-name=adh_pack
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=112
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -o /data/wonyoungjang/decodata/pooled/downstream/adhpack/adhpack_%j.log
#SBATCH --open-mode=append
# NOT `set -u`: env.local.sh runs conda activate -> /etc/bashrc -> unbound
# $BASHRCSOURCED kills the shell before the first echo (job 263079, 2026-09-01).
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
export DATASET=pooled
source scripts/resolve_dataset.sh   # DATA_DIR, OUTPUT_ROOT, TRAIN_CSV, VALID_CSV

DATASET_CFG="configs/pooled/dataset.json"
OUT="${OUTPUT_ROOT}/downstream/adherence"
RUNDIR="${OUTPUT_ROOT}/downstream/adhpack"; mkdir -p "${RUNDIR}" "${OUT}"

# name : task : epochs : label_map   (label_map empty => regression)
GROUPS=(
  "modality_clf:cls:15:{\"T1\":0,\"T2\":1,\"FLAIR\":2}"
  "sex_clf:cls:30:{\"M\":0,\"F\":1}"
  "dx_clf:cls:60:{\"healthy\":0,\"MCI\":1,\"AD\":2}"
  "age_reg:reg:60:"
)
# Weighted losses for the imbalanced targets only; sex is near 50/50.
declare -A CW=( [modality_clf]=1 [sex_clf]=0 [dx_clf]=1 )
declare -A TGT=( [modality_clf]=modality [sex_clf]=sex [dx_clf]=dx [age_reg]=age )

BS="${BS:-4}"; LR="${LR:-1e-3}"; WD="${WD:-1e-4}"; DROPOUT="${DROPOUT:-0.5}"
NG="${SLURM_GPUS_ON_NODE:-4}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))

# ---------------------------------------------------------------------------
# BUILD THE PER-TARGET CSVs + PREFLIGHT. build_adherence_csv filters per target
# (dx -> adni,oasis and the 3 clinical classes; modality -> T1/T2/FLAIR, dropping
# T1c which is vae_only and outside the diffusion vocabulary), so each judge trains
# on exactly the population it will later be asked to score.
# ---------------------------------------------------------------------------
fail=0
declare -a TODO=()
for spec in "${GROUPS[@]}"; do
  IFS=':' read -r name task epochs lmap <<< "${spec}"
  tgt="${TGT[$name]}"
  exp="${OUT}/${name}"; mkdir -p "${exp}/logs" "${exp}/weights"
  tr_csv="${exp}/adh_${tgt}_train.csv"; va_csv="${exp}/adh_${tgt}_valid.csv"
  if [[ -f "${exp}/weights/best.pt" ]]; then
    echo "[preflight] ${name}: best.pt exists — skipping"
    continue
  fi
  python3 scripts/build_adherence_csv.py --manifest "${TRAIN_CSV}" --target "${tgt}" \
      --out_csv "${tr_csv}" >/dev/null 2>&1
  python3 scripts/build_adherence_csv.py --manifest "${VALID_CSV}" --target "${tgt}" \
      --out_csv "${va_csv}" >/dev/null 2>&1
  n_tr=$(( $(wc -l < "${tr_csv}" 2>/dev/null || echo 1) - 1 ))
  n_va=$(( $(wc -l < "${va_csv}" 2>/dev/null || echo 1) - 1 ))
  printf "[preflight] %-13s target=%-8s task=%s epochs=%2s  train=%6d valid=%5d\n" \
         "${name}" "${tgt}" "${task}" "${epochs}" "${n_tr}" "${n_va}"
  [[ "${n_tr}" -lt 100 || "${n_va}" -lt 20 ]] && { echo "  !! too few rows"; fail=1; }
  TODO+=("${spec}")
done
if [[ "${fail}" -ne 0 ]]; then echo "=== PREFLIGHT FAILED — not launching. ==="; exit 1; fi
if [[ "${#TODO[@]}" -eq 0 ]]; then echo "=== all judges already trained; nothing to do. ==="; exit 0; fi
echo "[preflight] OK — ${#TODO[@]} judge(s) to train"

# Workers are split across every rank of every group: cpus / (groups x gpus).
NGRP=${#TODO[@]}
NW=$(( SLURM_CPUS_PER_TASK / (NGRP * NG) )); [[ "${NW}" -lt 1 ]] && NW=1
echo "[preflight] groups=${NGRP} gpus=${NG} num_workers=${NW} per rank"

START_TS=$(date +%s)
export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS=1
STAGGER_SEC=45
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d 2>/dev/null; then sleep 3; echo "[MPS] daemon up"
else echo "[MPS] FAILED to start — co-residency will likely deadlock"; fi
echo "=== adhpack job ${SLURM_JOB_ID} on $(hostname); GPUs=${CUDA_VISIBLE_DEVICES}; groups=${NGRP} ==="

( while true; do
    echo "[smi $(date +%F_%T)] $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tr '\n' '|')"
    sleep 300
  done ) &
SMI=$!

PIDS=(); PORT=29910
TERMED=false
on_term() {
  $TERMED && return; TERMED=true
  echo "=== [signal] caught at $(date +%F_%T): forwarding TERM to ${#PIDS[@]} groups ==="
  kill "${SMI}" 2>/dev/null
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM USR1

for spec in "${TODO[@]}"; do
  IFS=':' read -r name task epochs lmap <<< "${spec}"
  tgt="${TGT[$name]}"
  exp="${OUT}/${name}"
  tr_csv="${exp}/adh_${tgt}_train.csv"; va_csv="${exp}/adh_${tgt}_valid.csv"
  COMMON=( --dataset_config_path "${DATASET_CFG}" --train_csv "${tr_csv}" \
    --valid_csv "${va_csv}" --data_dir "${DATA_DIR}" --output_dir "${OUT}" \
    --run_name "${name}" --batch_size "${BS}" --num_workers "${NW}" \
    --epochs "${epochs}" --lr "${LR}" --weight_decay "${WD}" --dropout "${DROPOUT}" )
  if [[ "${task}" == "cls" ]]; then
    MOD=(-m downstream.train_attr_predictor --target "${tgt}" --label_map "${lmap}")
    [[ "${CW[$name]}" == "1" ]] && MOD+=(--class_weighted)
  else
    MOD=(-m downstream.train_brain_age)
  fi
  TORCHINDUCTOR_CACHE_DIR="${exp}/torchinductor" TRITON_CACHE_DIR="${exp}/triton" \
  torchrun --nproc_per_node=${NG} --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${PORT}" \
    "${MOD[@]}" "${COMMON[@]}" \
    >> "${exp}/logs/${name}_adhpack_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
  echo "launched ${name} (pid $!, task=${task}, epochs=${epochs}) port ${PORT}; staggering ${STAGGER_SEC}s"
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
  IFS=':' read -r name _ _ _ <<< "${spec}"
  if [[ -f "${OUT}/${name}/weights/best.pt" ]]; then
    echo "  ${name}: best.pt OK"; done_n=$((done_n+1))
  else
    echo "  ${name}: NO best.pt"; need_more=1
  fi
done

# CRASH-LOOP GUARD — the resubmit exists for walltime preemption. If nothing
# produced a checkpoint and the job was short, resubmitting re-runs the same crash;
# the UNet pack burned 5 allocations in 20 min that way on 2026-08-28.
ELAPSED=$(( $(date +%s) - START_TS ))
if [ "${need_more}" -eq 1 ] && [ "${done_n}" -eq 0 ] && [ "${ELAPSED}" -lt 1800 ]; then
  echo "=== CRASH LOOP: no judge finished and the job lasted only ${ELAPSED}s."
  echo "=== NOT resubmitting. Read ${OUT}/<name>/logs/*_adhpack_${SLURM_JOB_ID}.log. ==="
elif [ "${need_more}" -eq 1 ]; then
  echo "=== self-resubmit (a judge is unfinished): sbatch scripts/train_ADH_pack.sh ==="
  cd "${SCRIPT_DIR}" && sbatch "scripts/train_ADH_pack.sh"
else
  echo "=== all judges trained; no resubmit ==="
fi
echo "=== adhpack DONE ==="
exit 0
