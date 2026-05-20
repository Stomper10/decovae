"""BraTS tumor-seg dataset wrapper — supports real + synthetic mask-conditional mix.

Layout matches ``brain_age_dataset.py``: a CSV of (rel_path, rel_path_seg)
pairs is loaded into a MONAI CacheDataset. Synthetic samples are mask-
conditional VAE+UNet outputs paired with the conditioning mask, generated
later via ``scripts/generate_synthetic_for_seg.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
)


@dataclass
class SegSample:
    image_path: str
    mask_path: str
    source: str  # "real" | "synthetic"


def _build_records(csv_path: str, data_dir: str, source: str,
                   limit: Optional[int] = None) -> list[SegSample]:
    df = pd.read_csv(csv_path)
    for col in ("rel_path", "rel_path_seg"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing '{col}' column.")
    if limit is not None:
        df = df.head(limit)
    return [
        SegSample(image_path=f"{data_dir.rstrip('/')}/{row.rel_path}",
                  mask_path=f"{data_dir.rstrip('/')}/{row.rel_path_seg}",
                  source=source)
        for row in df.itertuples(index=False)
    ]


def build_transforms(resolution: tuple[int, int, int]) -> Compose:
    # BraTS-GLI 2023 labels: 0=bg, 1=NCR, 2=ED, 3=ET. ConvertToMultiChannel
    # produces (TC, WT, ET) one-hot via the standard BraTS class merges.
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=resolution, size_mode="all", mode="trilinear"),
        Resized(keys=["mask"], spatial_size=resolution, size_mode="all", mode="nearest"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["mask"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "mask"], dtype="float32"),
    ])


def make_dataset(real_csv: str, data_dir: str,
                 resolution: tuple[int, int, int],
                 synthetic_csv: Optional[str] = None,
                 synthetic_dir: Optional[str] = None,
                 real_limit: Optional[int] = None,
                 synth_limit: Optional[int] = None,
                 cache_rate: float = 0.0) -> CacheDataset:
    records = _build_records(real_csv, data_dir, "real", real_limit)
    if synthetic_csv is not None:
        if synthetic_dir is None:
            raise ValueError("synthetic_dir required when synthetic_csv is provided.")
        records.extend(_build_records(synthetic_csv, synthetic_dir, "synthetic", synth_limit))
    data = [{"image": r.image_path, "mask": r.mask_path, "source": r.source}
            for r in records]
    return CacheDataset(data=data, transform=build_transforms(resolution),
                        cache_rate=cache_rate)
