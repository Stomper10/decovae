"""IXI brain MRI dataset adapter.

The IXI release ships flat per-modality directories:
``/<root>/IXI-T1/IXI002-Guys-0828-T1.nii.gz`` etc., with sites Guys / HH / IOP.
Demographics live in ``IXI.xls`` and are joined in by ID at CSV build time.

CSV columns expected:
    eid       : IXI subject id, e.g. "IXI002"
    rel_path  : NIfTI path relative to ``data_dir``
    site      : "Guys" / "HH" / "IOP" (string; encoded numerically in conditions)
    age, sex  : demographics from IXI.xls (sex coded 0/1; NaN allowed)
    modality  : "T1" / "T2" / "PD" (optional; used to filter manifests)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import DatasetAdapter


_SITE_INDEX = {"Guys": 0, "HH": 1, "IOP": 2}


class IXIAdapter(DatasetAdapter):
    name = "ixi"
    modality = "mri"

    def extract_subject_id(self, image_path: str) -> str:
        # Filenames are ``IXI<id>-<site>-<study>-<modality>.nii.gz``; the
        # first ``-``-separated token is the canonical subject id.
        return os.path.basename(image_path).split("-")[0]

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
        if "age" in df:
            amin, amax = df["age"].min(), df["age"].max()
            df["norm_age"] = (df["age"] - amin) / (amax - amin) if amax > amin else 0.0
        else:
            df["norm_age"] = np.nan
        if "site" in df:
            # Map known sites to fixed indices; unknown → -1 to surface bugs early.
            df["site_idx"] = df["site"].map(_SITE_INDEX).fillna(-1).astype(int)
        return df

    def derive_conditions(self, row: pd.Series) -> list[float]:
        return [
            float(row.get("norm_age", 0.0) or 0.0),
            float(row.get("sex", 0.0) or 0.0),
            float(row.get("site_idx", 0)),
        ]
