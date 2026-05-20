#!/bin/bash
# Launcher for the 3D-MedDiffusion PatchVolume baseline.
#
# Uses the upstream training entry point unchanged. Only thing that varies
# between datasets is the --config yaml from configs/3d_meddiff/. Override at
# submission time:
#   CONFIG=configs/3d_meddiff/PatchVolume_4x_ixi.yaml sbatch train_3d_meddiff.sh
#
# Multi-GPU is handled by pytorch-lightning DDP inside the train script
# (cfg.model.gpus). Make sure the yaml's `gpus` matches `--gres` here.
#
#SBATCH --job-name=3d_meddiff
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
# pytorch-lightning needs --ntasks-per-node == cfg.model.gpus when launched
# under srun. For non-8-GPU runs override both at sbatch time, e.g.:
#   sbatch --gres=gpu:h100:4 --ntasks-per-node=4 train_3d_meddiff.sh
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=14
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi

# train_VAE.sh activates the DecoVAE env; 3D MedDiff needs its own.
source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc
conda activate 3d_meddiff

: "${CONFIG:=configs/3d_meddiff/PatchVolume_4x_ukb.yaml}"
: "${EXP_NAME:=ukb_c4}"

EXP_ROOT="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}"
LOGS_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOGS_DIR}"
EXP_LOG="${LOGS_DIR}/3d_meddiff_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

echo "3D MedDiff PatchVolume baseline"
echo "  config  : ${CONFIG}"
echo "  exp_name: ${EXP_NAME}"
echo "  job_id  : ${SLURM_JOB_ID}"

# Auto-resume from latest checkpoint if the 100k-step run was requeued.
# PL writes to ${default_root_dir}/my_model/version_N/checkpoints/ where N
# increments each run. Pick the .ckpt with the highest mtime across all
# versions and inject it into a temp yaml so train_PatchVolume.py picks it up.
LATEST_CKPT=$(ls -1t "${EXP_ROOT}/my_model/version_"*/checkpoints/*.ckpt 2>/dev/null | head -n 1)
if [[ -n "${LATEST_CKPT}" ]]; then
    RUN_CONFIG="${LOGS_DIR}/config_resume_${SLURM_JOB_ID}.yaml"
    python -c "
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG}')
cfg.model.resume_from_checkpoint = '${LATEST_CKPT}'
OmegaConf.save(cfg, '${RUN_CONFIG}')
print('Injected resume_from_checkpoint:', '${LATEST_CKPT}')
"
    echo "  resume  : ${LATEST_CKPT}"
    echo "  config (rewritten): ${RUN_CONFIG}"
else
    RUN_CONFIG="${CONFIG}"
    echo "  resume  : (none — fresh start)"
fi

# Auto-requeue on SIGUSR1 (sent 300s before time limit). PL writes a
# checkpoint to default_root_dir before SLURM kills the process.
max_restarts=1000
scontext=$(scontrol show job "${SLURM_JOB_ID}")
restarts=$(echo "${scontext}" | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

function resubmit() {
    if [[ ${restarts} -lt ${max_restarts} ]]; then
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    else
        echo "Restart limit reached"
        exit 1
    fi
}
trap 'resubmit' SIGUSR1

srun python external/3d_meddiff/train/train_PatchVolume.py --config "${RUN_CONFIG}" &
wait
exit 0
