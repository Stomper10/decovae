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

# --- Default paths for experiments ------------------------------------------
# Set these to your local data / output locations.
# OUTPUT_ROOT is the per-dataset root; each launcher appends "${STAGE}" to it
# (e.g. ${OUTPUT_ROOT}/stage1). Override STAGE per-invocation:
#   STAGE=stage2 sbatch train_VAE.sh
# export OUTPUT_ROOT=./outputs/ukb_20252          # experiments live under <OUTPUT_ROOT>/<STAGE>/<EXP_NAME>/
# export DATA_DIR=/path/to/UKB/imaging            # raw image volumes
# export TRAIN_CSV=/path/to/train.csv             # subject metadata + label
# export VALID_CSV=/path/to/valid.csv
# export CODE_DIR="$(pwd)"                        # repository root (extract_emb only)

# --- W&B (optional) ----------------------------------------------------------
# Set your Weights & Biases entity (team or user).
# Leave unset to disable W&B logging or run in offline mode.
# export WANDB_ENTITY=your-wandb-entity
