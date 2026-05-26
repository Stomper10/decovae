"""BraTS-GLI 2023 (Adult Glioma) dataset adapter.

The Synapse release ships per-case directories containing four MRI modalities
and a segmentation mask:

    BraTS-GLI-XXXXX-YYY/
        BraTS-GLI-XXXXX-YYY-t1n.nii.gz   # T1 native
        BraTS-GLI-XXXXX-YYY-t1c.nii.gz   # T1 contrast-enhanced (T1Gd)
        BraTS-GLI-XXXXX-YYY-t2w.nii.gz   # T2-weighted
        BraTS-GLI-XXXXX-YYY-t2f.nii.gz   # T2 FLAIR
        BraTS-GLI-XXXXX-YYY-seg.nii.gz   # NCR / ED / ET segmentation

Volumes are pre-processed: skull-stripped, co-registered, 240×240×155, 1 mm³ iso.

CSV columns expected:
    eid           : case id, e.g. "BraTS-GLI-00048-000"
    rel_path      : path to the *training* modality NIfTI, relative to data_dir
                    (default t1n; the CSV builder can swap modality)
    rel_path_seg  : optional path to the segmentation mask (used by mask-cond
                    downstream tasks; ignored by the unconditional VAE path)
"""
from __future__ import annotations

import os

import pandas as pd

from .base import DatasetAdapter


class BraTSAdapter(DatasetAdapter):
    name = "brats"
    modality = "mri"

    def extract_subject_id(self, image_path: str) -> str:
        # Source modality file:
        #   ``BraTS-GLI-00048-000-t1n.nii.gz`` → ``BraTS-GLI-00048-000`` (strip
        #   the modality suffix introduced by Synapse).
        # Embedding sidecar file (must round-trip the *full* id, not strip it):
        #   ``BraTS-GLI-00048-000_emb.nii.gz`` → ``BraTS-GLI-00048-000``
        # Without the embedding branch, ``rsplit("-", 1)`` would chop the
        # ``-000`` session token (e.g. ``BraTS-GLI-00048``), so the metadata
        # json matcher in extract_emb.py silently drops every case.
        stem = os.path.basename(image_path)
        for ext in (".nii.gz", ".nii", ".npy"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        for suffix in ("_emb", "_mu", "_sigma"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem.rsplit("-", 1)[0]

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
        # BraTS-GLI has no demographics; pass-through. A future extension can
        # compute normalized tumor volume from the seg mask for conditioning.
        return df.copy()

    def derive_conditions(self, row: pd.Series) -> list[float]:
        # Placeholder scalar so the UNet dataloader's `Lambdad(cond -> x[0])`
        # doesn't IndexError. The BraTS UNet runs unconditional (model_fm.json
        # sets include_meta_input=false), so this value is ignored at training
        # time. ControlNetMaisi carries the actual (mask) condition.
        return [0.0]
