#!/bin/bash
# Template for per-user environment setup.
#
# 1. Copy this file to env.local.sh
# 2. Edit the values below to match your local environment
# 3. env.local.sh is gitignored and will not be committed
#
# env.local.sh is sourced at the top of every train_*.sh / compute_metric.sh /
# extract_emb.sh launcher. All variables defined here become defaults for the
# `${VAR:=default}` expansion further down in those scripts, so you do not
# need to edit the launchers themselves to switch environments.
#
# Per-user paths defined here are forwarded to Python entry points as CLI
# arguments by the launchers (e.g. DATA_DIR -> --data_dir). The base JSON
# configs in configs/<dataset>/ ship with `null` placeholders for these keys.

# --- Shell initialization & conda env ---------------------------------------
# source ~/.bashrc
# source ~/miniconda3/bin/activate
# conda activate decovae

# --- TMPDIR / torch.compile cache -------------------------------------------
# Override if /tmp is small or shared. Comment out to use system defaults.
# TMPDIR_LOCAL=/path/to/large/tmp
# TORCH_CACHE_LOCAL=/path/to/torch_inductor_cache
# mkdir -p "${TMPDIR_LOCAL}" "${TORCH_CACHE_LOCAL}"
# export TMPDIR="${TMPDIR_LOCAL}"
# export TEMP="${TMPDIR_LOCAL}"
# export TMP="${TMPDIR_LOCAL}"
# export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_LOCAL}"

# --- Dataset paths -----------------------------------------------------------
# Each dataset gets a <PREFIX>_* block; launchers pick the right one via the
# DATASET env var (defaults to ukb_20252). See scripts/resolve_dataset.sh.
# Override STAGE per-invocation:    STAGE=stage2 sbatch train_VAE.sh
# Override DATASET per-invocation:  DATASET=ixi  sbatch train_VAE.sh
#
# export CODE_DIR="$(pwd)"                              # repo root (extract_emb only)
#
# UKB (field 20252, T1 brain MRI)
# export UKB_DATA_DIR=/path/to/UKB/imaging
# export UKB_TRAIN_CSV=/path/to/csv_files/train.csv
# export UKB_VALID_CSV=/path/to/csv_files/valid.csv
# export UKB_OUTPUT_ROOT=/path/to/outputs/ukb_20252
#
# IXI (cross-site, T1)
# export IXI_DATA_DIR=/path/to/IXI
# export IXI_TRAIN_CSV=/path/to/csv_files/ixi_T1_train.csv
# export IXI_VALID_CSV=/path/to/csv_files/ixi_T1_valid.csv
# export IXI_OUTPUT_ROOT=/path/to/outputs/ixi
#
# BraTS-GLI 2023
# export BRATS_DATA_DIR=/path/to/BraTS2023/.../TrainingData
# export BRATS_TRAIN_CSV=/path/to/csv_files/brats_train.csv
# export BRATS_VALID_CSV=/path/to/csv_files/brats_valid.csv
# export BRATS_OUTPUT_ROOT=/path/to/outputs/brats

# POOLED foundation corpus (multi-cohort). DATA_DIR is the offline .npy cache
# root (scripts/preprocess_cache.py); CSV is the pooled manifest. The pooled
# adapter loads already-preprocessed cache volumes (configs/pooled: cached_input).
# export POOLED_DATA_DIR=/path/to/decovae_cache
# export POOLED_TRAIN_CSV=/path/to/csv_files/pooled_manifest_train.csv
# export POOLED_VALID_CSV=/path/to/csv_files/pooled_manifest_valid.csv
# export POOLED_OUTPUT_ROOT=/path/to/outputs/pooled

# --- W&B (optional) ----------------------------------------------------------
# Set your Weights & Biases entity (team or user).
# Leave unset to disable W&B logging or run in offline mode.
# export WANDB_ENTITY=your-wandb-entity            # (-> --wandb_entity)

# --- Metric / FID feature extractor -----------------------------------------
# RadImageNet checkpoint used by compute_metric.sh for 2.5D FID.
# Download once and point to the .pth file. (-> --feature_extractor_path)
# export FEATURE_EXTRACTOR_PATH=/path/to/RadImageNet-ResNet50_notop.pth
