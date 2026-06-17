#!/bin/bash
# Conditioning-adherence eval: run a real-trained predictor over the diffusion
# model's conditioned generations and compare to the intended condition
# (gen_*.cond.json sidecars from compute_metric --eval_mode real_vs_gen).
#
# Run once per attribute, pointing at one generation set (cell × guidance):
#   age (regressor):
#     TARGET=age TASK=reg PRED_CKPT=.../brain_age/real_only/weights/best.pt \
#       GEN_DIR=.../vad-cov50-kl5e3/cells/all_T1_g20/outputs/volumes POSTFIX=30step_all_T1_g20 \
#       sbatch adherence_eval.sh
#   dx (classifier):
#     TARGET=dx TASK=cls PRED_CKPT=.../adherence/dx_clf/weights/best.pt ... sbatch adherence_eval.sh
#
# Lightweight single-process eval; still requests 4 GPUs to satisfy the cluster's
# QOSMinGRES rule (single-GPU jobs are rejected). Uses cuda:0 only.
#
#SBATCH --job-name=adherence
#SBATCH --account=gpu
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi

: "${DATASET:=pooled}"
source "${SCRIPT_DIR}/scripts/resolve_dataset.sh"

: "${TARGET:?set TARGET (age|sex|dx|cdrsb)}"
: "${TASK:?set TASK (reg|cls)}"
: "${PRED_CKPT:?set PRED_CKPT (predictor best.pt)}"
: "${GEN_DIR:?set GEN_DIR (outputs/volumes of a generation set)}"
: "${POSTFIX:?set POSTFIX (e.g. 30step_all_T1_g20)}"
: "${DATASET_CFG:=configs/${DATASET}/dataset.json}"
# results land next to the generations by default
: "${OUT_DIR:=$(dirname "${GEN_DIR}")/adherence}"
: "${LABEL_MAP:=}"   # optional; defaults to the map stored in the ckpt

mkdir -p "${OUT_DIR}"
EXP_LOG="${OUT_DIR}/adherence_${TARGET}_${POSTFIX}_${SLURM_JOB_ID:-$$}.log"
exec >> "${EXP_LOG}" 2>&1

LM_FLAG=""; [[ -n "${LABEL_MAP}" ]] && LM_FLAG="--label_map ${LABEL_MAP}"

echo "Adherence eval"
echo "  target/task : ${TARGET} / ${TASK}"
echo "  predictor   : ${PRED_CKPT}"
echo "  gen_dir     : ${GEN_DIR}"
echo "  postfix     : ${POSTFIX}"
echo "  out_dir     : ${OUT_DIR}"

srun python -m scripts.adherence_eval \
    --gen_dir "${GEN_DIR}" \
    --postfix "${POSTFIX}" \
    --target "${TARGET}" \
    --task "${TASK}" \
    --predictor_ckpt "${PRED_CKPT}" \
    --dataset_config_path "${DATASET_CFG}" \
    --output_csv "${OUT_DIR}/adherence_${TARGET}_${POSTFIX}.csv" \
    --output_json "${OUT_DIR}/adherence_${TARGET}_${POSTFIX}.json" \
    ${LM_FLAG}
exit 0
