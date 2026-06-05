#!/bin/bash
# Resolve dataset-specific paths into the variables the launchers consume.
#
# Sourced by train_VAE.sh / train_UNET.sh / train_DFT.sh / compute_metric.sh /
# extract_emb.sh *after* env.local.sh has been sourced and after DATASET has
# been set (defaults to ukb_20252).
#
# env.local.sh defines per-dataset blocks:
#   UKB_DATA_DIR / UKB_TRAIN_CSV / UKB_VALID_CSV / UKB_OUTPUT_ROOT
#   IXI_DATA_DIR / IXI_TRAIN_CSV / IXI_VALID_CSV / IXI_OUTPUT_ROOT
#   BRATS_DATA_DIR / ...
#
# This script forwards the matching block into the four canonical names the
# launchers expand with ${VAR:=...}. A caller that has *already* set
# DATA_DIR (etc.) keeps the override.

case "${DATASET}" in
    ukb_20252) _DS_PREFIX=UKB ;;
    ixi)       _DS_PREFIX=IXI ;;
    brats)     _DS_PREFIX=BRATS ;;
    pooled)    _DS_PREFIX=POOLED ;;
    *)
        echo "[resolve_dataset] Unknown DATASET='${DATASET}'." >&2
        echo "  Expected one of: ukb_20252 | ixi | brats | pooled" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

_v="${_DS_PREFIX}_DATA_DIR"    ; : "${DATA_DIR:=${!_v}}"
_v="${_DS_PREFIX}_TRAIN_CSV"   ; : "${TRAIN_CSV:=${!_v}}"
_v="${_DS_PREFIX}_VALID_CSV"   ; : "${VALID_CSV:=${!_v}}"
_v="${_DS_PREFIX}_OUTPUT_ROOT" ; : "${OUTPUT_ROOT:=${!_v}}"
unset _v _DS_PREFIX

export DATA_DIR TRAIN_CSV VALID_CSV OUTPUT_ROOT
