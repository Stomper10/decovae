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

# W&B logging — picked up by patches/3d_meddiff/wandb_logger.patch.
# Unset WANDB_PROJECT_3DMD to fall back to TensorBoard-only behavior.
: "${WANDB_PROJECT_3DMD:=decovae-3dmeddiff}"
: "${WANDB_RUN_NAME:=${EXP_NAME}}"
# Stable run id keeps every SLURM requeue attached to the same W&B run.
: "${WANDB_RUN_ID:=3dmd-${EXP_NAME}}"
: "${WANDB_RESUME:=allow}"
export WANDB_PROJECT_3DMD WANDB_RUN_NAME WANDB_RUN_ID WANDB_RESUME

EXP_ROOT="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}"
LOGS_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOGS_DIR}"
EXP_LOG="${LOGS_DIR}/3d_meddiff_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

echo "3D MedDiff PatchVolume baseline"
echo "  config  : ${CONFIG}"
echo "  exp_name: ${EXP_NAME}"
echo "  job_id  : ${SLURM_JOB_ID}"

# Auto-resume from the highest-step checkpoint. ModelCheckpoint writes two
# kinds of .ckpt under ${default_root_dir}/my_model/version_N/checkpoints/:
#   1. top-level latest_checkpoint{,-v1,-v2}.ckpt — the val-best top-3 (NOT
#      necessarily the freshest step; e.g. v-best may sit at step 88k while
#      training already hit 100k).
#   2. epoch=E-step=S-...-train/recon_loss=L.ckpt — every-N-steps snapshots,
#      one ckpt nested per directory. These are the true progress markers.
# We extract the `step=S` token from any matching path and pick the highest S.
# Fallback: mtime sort if no step-tagged ckpt found (e.g. very early fresh run).
LATEST_CKPT=$(find "${EXP_ROOT}/my_model" -maxdepth 5 -type f -name '*.ckpt' 2>/dev/null \
              | awk -F 'step=' 'NF>1 { n=$2; sub(/[^0-9].*/,"",n); print n"\t"$0 }' \
              | sort -k1,1n | tail -n 1 | cut -f2-)
if [[ -z "${LATEST_CKPT}" ]]; then
    LATEST_CKPT=$(ls -1t "${EXP_ROOT}/my_model/version_"*/checkpoints/*.ckpt 2>/dev/null | head -n 1)
fi
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
