#!/bin/bash
# One-shot bootstrap for the 3D-MedDiffusion baseline.
#
# Clones the upstream repo into external/3d_meddiff/ and applies the local
# patches under patches/3d_meddiff/ so DecoVAE-side data manifests work.
# The external dir is .gitignored — re-running this script is the canonical
# way to fetch (and re-patch) it.
#
# Usage:
#   bash scripts/setup_3d_meddiff.sh

set -euo pipefail

REPO_URL="https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion.git"
TARGET_DIR="external/3d_meddiff"
PATCH_DIR="patches/3d_meddiff"

mkdir -p external

if [[ -d "${TARGET_DIR}" ]]; then
    echo "[setup] ${TARGET_DIR} already exists. Pull to update? [y/N]"
    read -r ans
    if [[ "${ans}" =~ ^[Yy]$ ]]; then
        rm -rf "${TARGET_DIR}"
    else
        echo "[setup] Leaving existing checkout untouched."
        exit 0
    fi
fi

echo "[setup] Cloning ${REPO_URL} -> ${TARGET_DIR}"
git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
rm -rf "${TARGET_DIR}/.git"

if compgen -G "${PATCH_DIR}/*.patch" > /dev/null; then
    echo "[setup] Applying patches under ${PATCH_DIR}/"
    REPO_ROOT="$(pwd)"
    for p in "${PATCH_DIR}"/*.patch; do
        echo "[setup]   patch -p1 < ${p}"
        ( cd "${TARGET_DIR}" && patch -p1 < "${REPO_ROOT}/${p}" )
    done
else
    echo "[setup] No patches found under ${PATCH_DIR}/ (skipping)."
fi

cat <<'EOF'

[setup] Clone + patch complete.

3D-MedDiffusion needs its own conda env (Python 3.11 / torch 2.2 / cu121),
incompatible with the DecoVAE env (Python 3.12 / torch 2.6+ / cu124).

Create it once:

    conda create -n 3d_meddiff python=3.11.11 -y
    conda activate 3d_meddiff
    pip install -r external/3d_meddiff/requirements.txt
    pip install wandb           # required by patches/3d_meddiff/wandb_logger.patch

Build a data.json from a DecoVAE manifest (UKB / IXI / BraTS).  Our patched
VQGANDataset_4x accepts an explicit train/val split:

    python scripts/build_3d_meddiff_data_json.py \
        --dataset ukb_20252 \
        --train-csv ${TRAIN_CSV} --valid-csv ${VALID_CSV} \
        --data-dir ${DATA_DIR} \
        --output configs/3d_meddiff/data_ukb.json

Launch training:

    sbatch train_3d_meddiff.sh         # default UKB c=4
    CONFIG=configs/3d_meddiff/PatchVolume_4x_ixi.yaml sbatch train_3d_meddiff.sh
EOF
