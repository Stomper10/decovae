"""Abstract dataset adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DatasetAdapter(ABC):
    """Pluggable per-dataset glue.

    Concrete adapters declare class-level ``name``/``modality`` and implement
    the methods below. The training / extraction / metric scripts call into
    this surface for everything that varies between datasets.
    """

    name: str = ""
    modality: str = "mri"  # consumed by VAE_Transform.fixed_modality
    id_column: str = "eid"  # CSV column whose values map back to subject_id

    @abstractmethod
    def extract_subject_id(self, image_path: str) -> str:
        """Return the canonical subject id for a raw image path.

        Used to (a) name embedding files and (b) join image rows back to the
        label CSV. Must be deterministic and idempotent.
        """

    def embedding_filename(self, subject_id: str) -> str:
        """Filename for the latent embedding of a given subject."""
        return f"{subject_id}_emb.nii.gz"

    def mu_filename(self, subject_id: str) -> str:
        """Filename for the latent mean tensor of a given subject."""
        return self.embedding_filename(subject_id).replace("_emb.nii.gz", "_mu.npy")

    def sigma_filename(self, subject_id: str) -> str:
        """Filename for the latent log-sigma tensor of a given subject."""
        return self.embedding_filename(subject_id).replace("_emb.nii.gz", "_sigma.npy")

    @abstractmethod
    def load_manifest(self, csv_path: str, data_dir: str,
                      n: int | None = None) -> list[dict]:
        """Build a MONAI-style data list from a label CSV.

        Each entry is at minimum ``{"image": <abs path>, "class": modality}``.
        ``n``, when provided, truncates to the first n rows (e.g. validation cap).
        """

    @abstractmethod
    def normalize_label_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` augmented with whatever ``norm_*`` columns
        ``derive_conditions`` will read."""

    @abstractmethod
    def derive_conditions(self, row: pd.Series) -> list[float]:
        """Compute the conditional vector written into each embedding's JSON."""

    def meta_value_distribution(self, n: int, seed: int) -> np.ndarray | None:
        """Sampling distribution for ``compute_metric.py`` real_vs_gen meta values.

        Return ``None`` to fall back to uniform [0, 1].
        """
        return None
