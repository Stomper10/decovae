"""Brain-age dataset wrapper supporting real + synthetic mix.

The dataset returns ``(volume, age)`` pairs from an arbitrary mix of real
NIfTI volumes (loaded from disk via MONAI) and synthetic volumes (loaded
from disk as ``.npy`` / ``.nii.gz``).

The synthetic-mix support is a thin extension point: at construction time
pass ``synthetic_index_csv`` pointing at a CSV with ``rel_path`` + ``age``
columns built by ``scripts/generate_synthetic_for_age.py`` (TODO).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRangePercentilesd,
)


@dataclass
class BrainAgeSample:
    image_path: str
    age: float
    source: str  # "real" | "synthetic"


def _build_records(csv_path: str, data_dir: str, source: str,
                   limit: Optional[int] = None) -> list[BrainAgeSample]:
    df = pd.read_csv(csv_path)
    if "age" not in df.columns:
        raise ValueError(f"{csv_path} missing 'age' column required for brain-age regression.")
    if "rel_path" not in df.columns:
        raise ValueError(f"{csv_path} missing 'rel_path' column.")
    if limit is not None:
        df = df.head(limit)
    return [
        BrainAgeSample(image_path=f"{data_dir.rstrip('/')}/{row.rel_path}",
                       age=float(row.age),
                       source=source)
        for row in df.itertuples(index=False)
    ]


def build_transforms(resolution: tuple[int, int, int],
                     orientation_axcodes: str = "RAS") -> Compose:
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=orientation_axcodes),
        Resized(keys=["image"], spatial_size=resolution, size_mode="all"),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5,
                                        b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], dtype="float32"),
    ])


def make_dataset(real_csv: str, data_dir: str,
                 resolution: tuple[int, int, int],
                 synthetic_csv: Optional[str] = None,
                 synthetic_dir: Optional[str] = None,
                 real_limit: Optional[int] = None,
                 synth_limit: Optional[int] = None,
                 orientation_axcodes: str = "RAS",
                 cache_rate: float = 0.0) -> CacheDataset:
    """Build a MONAI CacheDataset mixing real and synthetic volumes.

    For real-only training pass ``synthetic_csv=None``. To run the
    synthetic-aug or synthetic-only modes, generate volumes with
    ``scripts/generate_synthetic_for_age.py`` first and point at the
    resulting CSV here.
    """
    records: list[BrainAgeSample] = _build_records(real_csv, data_dir, "real", real_limit)
    if synthetic_csv is not None:
        if synthetic_dir is None:
            raise ValueError("synthetic_dir required when synthetic_csv is provided.")
        records.extend(_build_records(synthetic_csv, synthetic_dir, "synthetic", synth_limit))

    data = [{"image": r.image_path, "age": r.age, "source": r.source}
            for r in records]
    return CacheDataset(data=data,
                        transform=build_transforms(resolution, orientation_axcodes),
                        cache_rate=cache_rate)


def stratify_ages(ages: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize ages into bins. Useful for stratified train/val split."""
    return np.digitize(ages, bins=np.quantile(ages, np.linspace(0, 1, n_bins + 1)[1:-1]))
