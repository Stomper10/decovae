"""Generic attribute dataset for adherence predictors.

Returns ``(volume, label)`` pairs for an arbitrary metadata column, so a single
SFCN-classifier training script can fit the categorical conditioning attributes
(sex, dx) and — via ``task="reg"`` — the continuous ones (age, cdrsb).

Categorical targets are mapped to integer class ids through an explicit
``label_map`` (e.g. ``{"CN": 0, "MCI": 1, "AD": 2}``); rows whose value is
missing or outside the map are dropped (a predictor is only trained on volumes
that actually carry the label). Continuous targets are returned as floats,
optionally min-max normalised.

The image transform is shared with :mod:`downstream.brain_age_dataset` so the
predictors see the exact same preprocessing as the brain-age regressor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from monai.data import CacheDataset

from downstream.brain_age_dataset import build_transforms, paths_are_npy


@dataclass
class AttrSample:
    image_path: str
    label: float          # int class id (cls) or float value (reg)
    source: str           # "real" | "synthetic"


def _build_records(csv_path: str, data_dir: str, target_col: str, task: str,
                   source: str, label_map: Optional[dict] = None,
                   value_min: Optional[float] = None, value_max: Optional[float] = None,
                   limit: Optional[int] = None) -> list[AttrSample]:
    df = pd.read_csv(csv_path)
    if "rel_path" not in df.columns:
        raise ValueError(f"{csv_path} missing 'rel_path' column.")
    if target_col not in df.columns:
        raise ValueError(f"{csv_path} missing target column {target_col!r}.")

    records: list[AttrSample] = []
    n_drop = 0
    for row in df.itertuples(index=False):
        rel = getattr(row, "rel_path")
        raw = getattr(row, target_col)
        if task == "cls":
            if label_map is None:
                raise ValueError("label_map required for task='cls'.")
            key = None if pd.isna(raw) else str(raw)
            if key not in label_map:        # missing / unseen category → drop
                n_drop += 1
                continue
            label: float = float(label_map[key])
        else:  # reg
            if pd.isna(raw):
                n_drop += 1
                continue
            v = float(raw)
            if value_min is not None and value_max is not None:
                v = (v - value_min) / (value_max - value_min + 1e-12)
            label = v
        records.append(AttrSample(image_path=f"{data_dir.rstrip('/')}/{rel}",
                                  label=label, source=source))
        if limit is not None and len(records) >= limit:
            break
    if n_drop:
        print(f"[attr_dataset] {csv_path}: dropped {n_drop} rows "
              f"with missing/unmapped {target_col!r}.")
    return records


def make_attr_dataset(real_csv: str, data_dir: str,
                      resolution: tuple[int, int, int],
                      target_col: str, task: str = "cls",
                      label_map: Optional[dict] = None,
                      value_min: Optional[float] = None,
                      value_max: Optional[float] = None,
                      synthetic_csv: Optional[str] = None,
                      synthetic_dir: Optional[str] = None,
                      real_limit: Optional[int] = None,
                      synth_limit: Optional[int] = None,
                      orientation_axcodes: str = "RAS",
                      cache_rate: float = 0.0) -> CacheDataset:
    """MONAI CacheDataset of (image, label) for one metadata attribute.

    Mirrors :func:`downstream.brain_age_dataset.make_dataset` but for a generic
    target column; synthetic mixing is supported for symmetry (unused by the
    adherence predictors, which train on real data only).
    """
    records = _build_records(real_csv, data_dir, target_col, task, "real",
                             label_map, value_min, value_max, real_limit)
    if synthetic_csv is not None:
        if synthetic_dir is None:
            raise ValueError("synthetic_dir required when synthetic_csv is provided.")
        records.extend(_build_records(synthetic_csv, synthetic_dir, target_col, task,
                                      "synthetic", label_map, value_min, value_max,
                                      synth_limit))

    data = [{"image": r.image_path, "label": r.label, "source": r.source}
            for r in records]
    return CacheDataset(data=data,
                        transform=build_transforms(resolution, orientation_axcodes,
                                                   npy=paths_are_npy(records)),
                        cache_rate=cache_rate)
