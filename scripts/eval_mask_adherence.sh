#!/bin/bash
# Mask adherence / real-data ceiling (scripts/eval_mask_adherence.py). CPU by default:
# segmentor inference only, so it never competes for the GPU job cap.
#
#   SRC=real sbatch scripts/eval_mask_adherence.sh
#   SRC=gen GEN_DIR=<cell>/outputs/volumes LABEL=cn_vad_A_orig sbatch scripts/eval_mask_adherence.sh
#
#SBATCH --job-name=mask_adh
#SBATCH --partition=cpu-standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
set -o pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SCRIPT_DIR}"
source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc
conda activate "${CONDA_ENV:-deco_v15}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

: "${SRC:?set SRC (real|gen)}"
: "${LABEL:=${SRC}_pooled_FLAIR}"
: "${OUT_DIR:=${SCRIPT_DIR}/journal_plan/mask_adherence}"
mkdir -p "${OUT_DIR}"
ARGS=(--src "${SRC}" --out "${OUT_DIR}/${LABEL}.csv")
[[ "${SRC}" == gen ]] && ARGS+=(--gen_dir "${GEN_DIR:?set GEN_DIR}")
[[ -n "${MAX_CASES:-}" ]] && ARGS+=(--max_cases "${MAX_CASES}")
echo "[mask_adh] ${LABEL}  src=${SRC}  $(date)"
python -m scripts.eval_mask_adherence "${ARGS[@]}"
