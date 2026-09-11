#!/bin/bash
# §3a — SLURM wrapper for eval_seg_domain_shift.py
# One GPU, N cases each scored in native + pooled space.
#
# Usage:
#   sbatch --export=ALL,N_CASES=25 scripts/eval_seg_domain_shift.sh
#
#SBATCH --job-name=seg_shift
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=P2
#SBATCH --exclude=b00,b06,b07,b08,b10,b12,b13,b14,b15,b16,b17,b18,b24,b26,b28,b31
#SBATCH --time=0-01:00:00
#SBATCH --mem=64GB
#SBATCH -o /leelabsg/data/wonyoungjang/decovae/journal_plan/seg_domain_shift_%j.log
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
source "${SCRIPT_DIR}/env.local.sh"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

: "${N_CASES:=25}"
: "${BRATS_ROOT:=/leelabsg/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/train}"
: "${SEG_CKPT:=/leelabsg/data/wonyoungjang/decodata/brats/downstream/tumor_seg/real_only_FLAIR/weights/best.pt}"
: "${OUT:=${SCRIPT_DIR}/journal_plan/results_seg_domain_shift.csv}"

python3 "${SCRIPT_DIR}/scripts/eval_seg_domain_shift.py" \
    --brats_root "${BRATS_ROOT}" \
    --seg_ckpt "${SEG_CKPT}" \
    --n_cases "${N_CASES}" \
    --out "${OUT}"
