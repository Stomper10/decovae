"""BraTS tumor-seg dataset wrapper — supports real + synthetic mask-conditional mix.

Layout matches ``brain_age_dataset.py``: a CSV of (rel_path, rel_path_seg)
pairs is loaded into a MONAI CacheDataset. Synthetic samples are mask-
conditional VAE+UNet outputs paired with the conditioning mask, generated
later via ``scripts/generate_synthetic_for_seg.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    Resized,
)


class ConvertBratsGLI2023Labelsd(MapTransform):
    """(TC, WT, ET) one-hot from BraTS-GLI 2023 labels {1=NCR, 2=ED, 3=ET}.

    This exists because MONAI's ConvertToMultiChannelBasedOnBratsClassesd encodes
    the PRE-2023 convention, in which ET is label 4 (monai 1.4.0,
    array.py::ConvertToMultiChannelBasedOnBratsClasses.__call__):

        result = [(img == 1) | (img == 4),
                  (img == 1) | (img == 4) | (img == 2),
                  img == 4]

    BraTS-GLI 2023 renumbered ET from 4 to 3, and no voxel in this corpus is ever
    4 (verified: the union of raw label values over the training segmentations is
    {0, 1, 2, 3}). Against this data the MONAI transform therefore makes ET
    IDENTICALLY EMPTY, and silently drops the enhancing tumour out of TC and WT as
    well -- label 3 is typically the largest sub-region (BraTS-GLI-00000-000:
    32,731 ET voxels against 11,738 NCR). Job 264329 reported ET = 0.0000 exactly,
    for 26 straight epochs, for this reason and not because FLAIR lacks the signal.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            # Fail loudly rather than silently mis-map if pre-2023 data is ever
            # mixed in: under that convention 4 is ET and 3 is unused, so the
            # correct mapping is the opposite one and a wrong guess is invisible.
            if bool((img == 4).any()):
                raise ValueError(
                    "label 4 present: this looks like the pre-2023 BraTS "
                    "convention, for which ET is 4, not 3.")
            regions = [(img == 1) | (img == 3),                 # TC = NCR + ET
                       (img == 1) | (img == 2) | (img == 3),    # WT = NCR + ED + ET
                       (img == 3)]                              # ET
            stack = torch.stack if isinstance(img, torch.Tensor) else np.stack
            d[key] = stack(regions, 0)
        return d


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


def build_transforms(resolution: tuple[int, int, int],
                     roi_size: Optional[tuple[int, int, int]] = None,
                     train: bool = False) -> Compose:
    # BraTS-GLI 2023 labels: 0=bg, 1=NCR, 2=ED, 3=ET -> (TC, WT, ET) one-hot.
    # See ConvertBratsGLI2023Labelsd for why MONAI's own transform cannot be used.
    tf = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Resized(keys=["image"], spatial_size=resolution, size_mode="all", mode="trilinear"),
        Resized(keys=["mask"], spatial_size=resolution, size_mode="all", mode="nearest"),
        ConvertBratsGLI2023Labelsd(keys=["mask"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    # Train on roi_size crops, exactly the window sliding_window_inference uses at
    # validation. Two reasons this is not optional: (a) SegResNet downsamples 3x,
    # so a spatial dim that is not a multiple of 8 makes the decoder skip-add
    # mismatch (155 -> 78 -> 39 -> 20, then up to 40 against 39) and the forward
    # pass dies; the crop guarantees divisibility regardless of `resolution`.
    # (b) a full 240x240x144 volume at init_filters=32 holds ~1 GB per level-0
    # activation per sample, which does not fit alongside its backward graph.
    if train and roi_size is not None:
        tf.append(RandSpatialCropd(keys=["image", "mask"], roi_size=roi_size,
                                   random_size=False))
        for ax in range(3):
            tf.append(RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=ax))
    tf.append(EnsureTyped(keys=["image", "mask"], dtype="float32"))
    return Compose(tf)


def make_dataset(real_csv: str, data_dir: str,
                 resolution: tuple[int, int, int],
                 roi_size: Optional[tuple[int, int, int]] = None,
                 train: bool = False,
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
    return CacheDataset(
        data=data,
        transform=build_transforms(resolution, roi_size=roi_size, train=train),
        cache_rate=cache_rate)
