"""UK Biobank field 20252 (T1 brain MRI) dataset adapter.

This adapter wires the UKB Field 20252 imaging release into the
DecoVAE training / extraction / evaluation scripts. It assumes a label
CSV with the following columns:

    eid          : UK Biobank subject id (also encoded in image_path)
    rel_path     : T1 MRI volume path, relative to ``data_dir``
    age, sex     : demographics (sex coded 0/1)
    p25001_i2    : cortical grey matter volume    (cGM)
    p25003_i2    : cerebrospinal fluid volume     (CSF)
    p25005_i2    : total grey matter volume (used to derive dGM = p25005 - p25001)
    p25007_i2    : white matter volume            (WM)

Image paths must be deep enough that ``path.split("/")[4]`` is the
eid directory (e.g. ``/.../<eid>_20252_2_0/T1/T1_brain_to_MNI.nii.gz``).
See ``extract_subject_id`` for the exact slicing logic.

Use ``configs/ukb_20252/dataset.json`` (with a per-user
``dataset.local.json`` for ``data_dir`` / CSV paths) to point at your
local UKB extract.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import DatasetAdapter


# UKB Field codes for brain volumetry (DataField IDs joined with instance 2 suffix).
UKB_FIELD_CGM = "p25001_i2"  # cortical grey matter volume
UKB_FIELD_CSF = "p25003_i2"  # cerebrospinal fluid volume
UKB_FIELD_GM_TOTAL = "p25005_i2"  # total grey matter volume (cortical + deep)
UKB_FIELD_WM = "p25007_i2"  # white matter volume

# 5-bucket stratified sampling cut points used by ``meta_value_distribution``
# (legacy compute_fid logic). Values are quantiles of the training distribution.
_META_QUANTILES = {"min": 0.0, "p05": 0.1667, "p25": 0.3333,
                   "p75": 0.6667, "p95": 0.8333, "max": 1.0}
_META_WEIGHTS = (0.05, 0.20, 0.50, 0.20, 0.05)

# Index into ``image_path.split("/")`` that yields the subject directory.
# Assumes paths like ``/<root>/<dataset_dir>/<subject_dir>/T1/...``.
_SUBJECT_DIR_INDEX = 4
_SUBJECT_ID_LEN = 7  # eid prefix length within the subject directory name


class UKB20252Adapter(DatasetAdapter):
    name = "ukb_20252"
    modality = "mri"

    def extract_subject_id(self, image_path: str) -> str:
        # Convention from train_UNET.py and extract_emb.py: the 5th
        # path component (index 4) is the eid directory, and the first 7 chars
        # are the eid itself. Preserved verbatim for regression equality.
        return image_path.split("/")[_SUBJECT_DIR_INDEX][:_SUBJECT_ID_LEN]

    def load_manifest(self, csv_path: str, data_dir: str,
                      n: int | None = None) -> list[dict]:
        df = pd.read_csv(csv_path)
        if n is not None:
            df = df[:n]
        return [
            {"image": os.path.join(data_dir, rel), "class": self.modality}
            for rel in df["rel_path"]
        ]

    def normalize_label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["dgm"] = df[UKB_FIELD_GM_TOTAL] - df[UKB_FIELD_CGM]
        df["norm_age"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
        df["norm_csf"] = (df[UKB_FIELD_CSF] - df[UKB_FIELD_CSF].min()) / (df[UKB_FIELD_CSF].max() - df[UKB_FIELD_CSF].min())
        df["norm_cgm"] = (df[UKB_FIELD_CGM] - df[UKB_FIELD_CGM].min()) / (df[UKB_FIELD_CGM].max() - df[UKB_FIELD_CGM].min())
        df["norm_dgm"] = (df["dgm"] - df["dgm"].min()) / (df["dgm"].max() - df["dgm"].min())
        df["norm_wm"] = (df[UKB_FIELD_WM] - df[UKB_FIELD_WM].min()) / (df[UKB_FIELD_WM].max() - df[UKB_FIELD_WM].min())
        return df

    def derive_conditions(self, row: pd.Series) -> list[float]:
        return [
            float(row["norm_age"]),
            float(row["sex"]),
            float(row["norm_csf"]),
            float(row["norm_cgm"]),
            float(row["norm_dgm"]),
            float(row["norm_wm"]),
        ]

    def meta_value_distribution(self, n: int, seed: int) -> np.ndarray:
        # Five-bucket stratified sample replicating the legacy compute_fid logic.
        # Uses the global numpy RNG (np.random.seed) to match pre-refactor outputs
        # bit-for-bit; switching to default_rng would change generated samples.
        np.random.seed(seed)
        q = _META_QUANTILES
        edges = [(q["min"], q["p05"]), (q["p05"], q["p25"]),
                 (q["p25"], q["p75"]), (q["p75"], q["p95"]),
                 (q["p95"], q["max"])]
        parts = [np.random.uniform(lo, hi, int(n * w))
                 for (lo, hi), w in zip(edges, _META_WEIGHTS)]
        out = np.concatenate(parts)
        np.random.shuffle(out)
        return out
