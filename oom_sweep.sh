#!/bin/bash
# Single-h100 OOM sweep for the pooled VAE (scripts/oom_sweep.py).
#SBATCH --job-name=oom_sweep
#SBATCH --account=gpu
# gpu-4farm QOS enforces a minimum GRES (>=4 GPUs); single-GPU is held with
# QOSMinGRES. So request the 4-GPU minimum but the sweep only uses cuda:0.
#SBATCH --partition=gpu-4farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH -o oom_sweep_%j.log
#SBATCH --open-mode=append

set -eo pipefail  # NOT -u: conda activate / /etc/bashrc reference unbound vars
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -f "${SCRIPT_DIR}/env.local.sh" ]]; then source "${SCRIPT_DIR}/env.local.sh"; fi
cd "${SCRIPT_DIR}"
echo "[oom_sweep] node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python3 scripts/oom_sweep.py --model_config_path configs/pooled/model_fm.json
