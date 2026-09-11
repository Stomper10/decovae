#!/bin/bash
# 3D MedDiff Phase 4 — BiFlowNet latent diffusion. Separate launcher, not a STAGE
# branch of train_3d_meddiff.sh, because nothing is shared: a different entry point
# (train_BiFlowNet_SingleRes.py), argparse instead of an omegaconf yaml, a different
# dataset class, and latents rather than volumes as input.
#
#   sbatch train_3d_meddiff_phase4.sh
#   RESUME_CKPT=/abs/path.pt sbatch train_3d_meddiff_phase4.sh
#
# LAUNCH = srun (1 task) -> torchrun (8 procs), NOT srun python x8. Phases 1-2 run
# under Lightning, which reads SLURM_PROCID itself, so ntasks-per-node=8 works there.
# BiFlowNet is plain torch DDP: dist.init_process_group("nccl") with env:// needs
# RANK/WORLD_SIZE, which only torchrun sets. Copying Phase 1's ntasks-per-node=8 is
# what killed job 266382 -- all 8 ranks died at init with "environment variable RANK
# expected, but not set". Upstream's README launches it with torchrun too; the header
# and srun line below follow train_VAE.sh, the repo's torchrun-under-srun template.
#
# CONDITIONING. Phase 4 IS conditional, on a single categorical drawn from the data
# json's KEYS (Singleres_dataset hands `int(key)` to the model as cls_idx). We use
# modality: 0=T1, 1=T2, 2=FLAIR, built by scripts/make_3dmd_phase4_classes.py. That is
# what makes the baseline comparable to our per-modality gFID slices at all -- an
# unconditional 3DMD could only be scored against the pooled mixture. It is NOT
# like-for-like: our arms take a 13-D vector (modality + sex + dx + age + cdrsb)
# against this single categorical. That is 3DMD's own architecture and the plan says
# to run their recipe verbatim; state the difference in the paper.
#
# --resolution IS 48, NOT the upstream default of 32. Our latents are [8, 48, 48, 48]
# (verified on the Phase 1 cache). The dataset divides this by 64 and passes it to the
# model, so a wrong value does not crash -- it silently trains at the wrong scale.
#SBATCH --job-name=3dmd_p4
#SBATCH --account=gpu
#SBATCH --partition=gpu-8farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH -o /dev/null
#SBATCH --open-mode=append
set -o pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)}"
cd "${SCRIPT_DIR}"
[[ -f env.local.sh ]] && source env.local.sh
source ~/miniconda3/bin/activate 2>/dev/null || source ~/.bashrc
conda activate 3d_meddiff

: "${EXP_NAME:=pooled_s2}"
: "${DATA_JSON:=configs/3d_meddiff/SingleRes_pooled.json}"
: "${NUM_CLASSES:=3}"          # T1 / T2 / FLAIR — must match the json's key count
: "${RESOLUTION:=48 48 48}"    # our latent grid; upstream default 32 is wrong here
: "${VOLUME_CHANNELS:=8}"
: "${BATCH_SIZE:=16}"
: "${STEP_BUDGET:=250000}"     # = our diffusion UNet's 250k steps; EPOCHS is derived from it below
: "${EPOCHS:=}"                 # set explicitly only to override the budget
: "${CKPT_EVERY:=5000}"         # each checkpoint also samples + decodes a volume on rank 0
: "${KEEP_EVERY:=25000}"        # milestones kept for the checkpoint sweep (needs biflownet_keep_milestones.patch)
: "${RESUME_CKPT:=}"

EXP_ROOT="/data/wonyoungjang/decodata/3d_meddiff/${EXP_NAME}"
RESULTS_DIR="${EXP_ROOT}/biflownet"
LOGS_DIR="${EXP_ROOT}/logs"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"
EXP_LOG="${LOGS_DIR}/3dmd_p4_${SLURM_JOB_ID}.log"
exec >> "${EXP_LOG}" 2>&1

# The AE checkpoint is needed to decode samples during training. It must be PHASE 2's
# endpoint -- the same model Phase 3 encoded with -- or the decoder will not match the
# latents. Phase 2 froze the encoder, so this is the step=100,000 encoder plus the
# fine-tuned decoder.
TARGET_STEPS=$(python3 -c "
from omegaconf import OmegaConf
print(int(OmegaConf.load('configs/3d_meddiff/PatchVolume_4x_pooled_s2.yaml').model.max_steps))")
: "${AE_CKPT:=$(find "${EXP_ROOT}/my_model" -maxdepth 5 -type f -name '*.ckpt' 2>/dev/null \
    | awk -F 'step=' -v t="${TARGET_STEPS}" 'NF>1 { n=$2; sub(/[^0-9].*/,"",n);
        if (n+0 == t+0) print $0 }' | head -n 1)}"

echo "3D MedDiff Phase 4 — BiFlowNet"
echo "  exp_name    : ${EXP_NAME}"
echo "  data_json   : ${DATA_JSON}"
echo "  num_classes : ${NUM_CLASSES}"
echo "  resolution  : ${RESOLUTION}"
echo "  ae_ckpt     : ${AE_CKPT:-<NOT FOUND>}"
echo "  results_dir : ${RESULTS_DIR}"
echo "  job_id      : ${SLURM_JOB_ID}"

if [[ -z "${AE_CKPT}" || ! -f "${AE_CKPT}" ]]; then
  echo "[FATAL] no Phase 2 checkpoint at step=${TARGET_STEPS} under ${EXP_ROOT}/my_model."
  echo "[FATAL] Not falling back to latest_checkpoint.ckpt — that is a rolling val-best"
  echo "[FATAL] slot, and decoding with a model that did not produce these latents is"
  echo "[FATAL] exactly the silent failure this guards. Run Phase 2+3 first."
  exit 1
fi
if [[ ! -f "${DATA_JSON}" ]]; then
  echo "[FATAL] ${DATA_JSON} missing. Build it after Phase 3:"
  echo "[FATAL]   python3 scripts/make_3dmd_phase4_classes.py --latent_dir ${EXP_ROOT}/latents"
  exit 1
fi
python3 -c "
import json, sys
d = json.load(open('${DATA_JSON}'))
if len(d) != ${NUM_CLASSES}:
    sys.exit(f'[FATAL] json has {len(d)} classes, NUM_CLASSES=${NUM_CLASSES}')
print(f'  json classes: {sorted(d)}')" || exit 1

# STEP BUDGET. Upstream trains 1000 epochs with no step limit, and its DataLoader has no
# DistributedSampler: every rank walks the FULL latent set, so one epoch is
# ceil(N / batch_size) steps, not N / (batch_size * gpus). On 51,169 latents that is
# 3,199 steps -- 1000 epochs would be ~3.2M steps and would hold the one AIBIO GPU slot
# for weeks. The budget matches our UNet; checkpoints every CKPT_EVERY with milestones
# every KEEP_EVERY kept, so the endpoint is not assumed to be the best (it was not for
# the VAE or for 3DMD Phase 2).
if [[ -z "${EPOCHS}" ]]; then
  EPOCHS=$(python3 -c "
import json, math, os
d = json.load(open('${DATA_JSON}'))
n = sum(len(os.listdir(v + '_latents')) for v in d.values())   # Singleres_dataset appends _latents
spe = math.ceil(n / ${BATCH_SIZE})
print(math.ceil(${STEP_BUDGET} / spe))") || { echo "[FATAL] could not derive EPOCHS"; exit 1; }
fi
echo "  step budget : ${STEP_BUDGET} -> epochs ${EPOCHS}  ckpt_every ${CKPT_EVERY}  keep_every ${KEEP_EVERY}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
export OMP_NUM_THREADS=1

max_restarts=1000
restarts=$(scontrol show job ${SLURM_JOB_ID} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
function resubmit() {
    if [[ ${restarts} -lt ${max_restarts} ]]; then scontrol requeue ${SLURM_JOB_ID}; exit 0; fi
    exit 1
}
trap 'resubmit' SIGUSR1

# Auto-resume from our own newest checkpoint unless one was named explicitly. The
# upstream script reuses the experiment directory when --ckpt is given, so this keeps
# a requeued job in the same run rather than starting 001-BiFlowNet, 002-, ...
if [[ -z "${RESUME_CKPT}" ]]; then
  RESUME_CKPT=$(ls -1t "${RESULTS_DIR}"/*/checkpoints/*.pt 2>/dev/null | head -n 1)
fi
CKPT_ARG=""
if [[ -n "${RESUME_CKPT}" && -f "${RESUME_CKPT}" ]]; then
  CKPT_ARG="--ckpt ${RESUME_CKPT}"; echo "  resume      : ${RESUME_CKPT}"
else
  echo "  resume      : (none — fresh start)"
fi

NPROC_PER_NODE=8
srun --cpu-bind=none,v --accel-bind=g torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  external/3d_meddiff/train/train_BiFlowNet_SingleRes.py \
    --data-path "${DATA_JSON}" \
    --results-dir "${RESULTS_DIR}" \
    --AE-ckpt "${AE_CKPT}" \
    --num-classes "${NUM_CLASSES}" \
    --volume-channels "${VOLUME_CHANNELS}" \
    --resolution ${RESOLUTION} \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --ckpt-every "${CKPT_EVERY}" \
    --keep-every "${KEEP_EVERY}" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-112} / NPROC_PER_NODE ))" \
    ${CKPT_ARG} &
wait
exit 0
