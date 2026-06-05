# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, Dict, List, Optional

import torch
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

SUPPORT_MODALITIES = ["ct", "mri"]


def define_fixed_intensity_transform(
    intensity_norm: Dict[str, Any],
    image_keys: List[str] = ["image"],
) -> List:
    """Build the intensity-normalization transform from a config dict.

    Args:
        intensity_norm: One of
            - ``{"type": "percentile", "lower": 0.0, "upper": 99.5,
                 "b_min": 0.0, "b_max": 1.0, "clip": False}``
            - ``{"type": "range", "a_min": -1000, "a_max": 1000,
                 "b_min": 0.0, "b_max": 1.0, "clip": True}``
            - ``{"type": "none"}`` → no transform.
        image_keys: Dictionary keys to apply the transform to.
    """
    norm_type = intensity_norm.get("type", "none").lower()
    if norm_type == "percentile":
        return [ScaleIntensityRangePercentilesd(
            keys=image_keys,
            lower=intensity_norm["lower"],
            upper=intensity_norm["upper"],
            b_min=intensity_norm["b_min"],
            b_max=intensity_norm["b_max"],
            clip=intensity_norm.get("clip", False),
        )]
    if norm_type == "range":
        return [ScaleIntensityRanged(
            keys=image_keys,
            a_min=intensity_norm["a_min"],
            a_max=intensity_norm["a_max"],
            b_min=intensity_norm["b_min"],
            b_max=intensity_norm["b_max"],
            clip=intensity_norm.get("clip", True),
        )]
    if norm_type == "none":
        return []
    warnings.warn(f"Unknown intensity_norm.type='{norm_type}'; skipping intensity normalization.")
    return []


def define_random_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """Modality-specific random intensity augmentations.

    Augmentation policy is tied to imaging physics rather than pre-normalization
    statistics, so it stays modality-keyed even after we externalize the fixed
    intensity transform.
    """
    modality = modality.lower()
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Random intensity augmentation only supports {SUPPORT_MODALITIES}. "
            f"Got {modality}. Skipping."
        )
        return []

    if modality == "ct":
        return []  # CT HU intensity is stable across datasets.
    if modality == "mri":
        return [
            RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)),
            RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.5, 1.0)),
            RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)),
            RandHistogramShiftd(keys=image_keys, prob=0.05, num_control_points=10),
        ]
    return []


def define_vae_transform(
    is_train: bool,
    modality: str,
    random_aug: bool,
    resolution: List[int],
    intensity_norm: Dict[str, Any],
    orientation_axcodes: str = "RAS",
    k: int = 4,
    patch_size: List[int] = [128, 128, 128],
    val_patch_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
    cached: bool = False,
) -> Compose:
    """Compose the VAE preprocessing pipeline.

    All dataset-specific shape/orientation/intensity choices flow in via
    explicit args (``resolution``, ``orientation_axcodes``, ``intensity_norm``)
    so the function itself stays dataset-agnostic.

    ``cached=True``: inputs are already-preprocessed cache volumes (``.npy``,
    single-channel, final resolution, intensity already in [0, 1]). The whole
    deterministic preprocessing block (load-orient-resize-channel-intensity-
    spacing) is skipped — only load + add-channel runs — so the pooled cache
    feeds the VAE without redundant (and lossy) re-preprocessing. Random
    augmentation and patch cropping still apply when ``is_train``.
    """
    modality = modality.lower()
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Modality {modality} is outside {SUPPORT_MODALITIES}; "
            "MRI-only ops (channel selection, MRI random intensity aug) will be skipped."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    if cached:
        # Cache volumes are bare 3D arrays (no channel dim), already at final
        # resolution / orientation / [0,1] intensity. Just load and add channel.
        common_transform = [
            SelectItemsd(keys=keys, allow_missing_keys=True),
            LoadImaged(keys=keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=keys, allow_missing_keys=True, channel_dim="no_channel"),
        ]
    else:
        common_transform = [
            SelectItemsd(keys=keys, allow_missing_keys=True),
            LoadImaged(keys=keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
            Orientationd(keys=keys, axcodes=orientation_axcodes, allow_missing_keys=True),
            Resized(keys=keys, spatial_size=tuple(resolution), size_mode="all", allow_missing_keys=True),
        ]

        if modality == "mri":
            common_transform.append(Lambdad(keys=image_keys, func=lambda x: x[select_channel : select_channel + 1, ...]))

        common_transform.extend(define_fixed_intensity_transform(intensity_norm, image_keys=image_keys))

        if spacing_type == "fixed":
            common_transform.append(
                Spacingd(keys=image_keys + label_keys, allow_missing_keys=True, pixdim=spacing, mode=interp_mode)
            )

    random_transform = []
    if is_train and random_aug:
        random_transform.extend(define_random_intensity_transform(modality, image_keys=image_keys))
        random_transform.extend(
            [RandFlipd(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axis=axis) for axis in range(3)]
            + [
                RandRotate90d(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axes=axes)
                for axes in [(0, 1), (1, 2), (0, 2)]
            ]
            + [
                RandScaleIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, factors=(0.9, 1.1)),
                RandShiftIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, offsets=0.05),
            ]
        )

        if spacing_type == "rand_zoom":
            random_transform.extend(
                [
                    RandZoomd(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        min_zoom=0.5,
                        max_zoom=1.5,
                        keep_size=False,
                        mode=interp_mode,
                    ),
                    RandRotated(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        keep_size=True,
                        mode=interp_mode,
                    ),
                ]
            )

    if is_train:
        train_crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(
                keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True
            ),
        ]
    else:
        val_crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    final_transform = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True)]

    if is_train:
        return Compose(
            common_transform + random_transform + train_crop + final_transform
            if random_aug
            else common_transform + train_crop + final_transform
        )
    return Compose(common_transform + val_crop + final_transform)


class VAE_Transform:
    """Caches train/val transforms keyed by modality."""

    def __init__(
        self,
        is_train: bool,
        random_aug: bool,
        resolution: List[int],
        intensity_norm: Dict[str, Any],
        orientation_axcodes: str = "RAS",
        k: int = 4,
        patch_size: List[int] = [128, 128, 128],
        val_patch_size: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        spacing_type: str = "original",
        spacing: Optional[List[float]] = None,
        image_keys: List[str] = ["image"],
        label_keys: List[str] = [],
        additional_keys: List[str] = [],
        select_channel: int = 0,
        cached: bool = False,
    ):
        if spacing_type not in ["original", "fixed", "rand_zoom"]:
            raise ValueError(
                f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}."
            )

        self.is_train = is_train
        self.transform_dict = {}

        for modality in SUPPORT_MODALITIES:
            self.transform_dict[modality] = define_vae_transform(
                is_train=is_train,
                modality=modality,
                random_aug=random_aug,
                resolution=resolution,
                intensity_norm=intensity_norm,
                orientation_axcodes=orientation_axcodes,
                k=k,
                patch_size=patch_size,
                val_patch_size=val_patch_size,
                output_dtype=output_dtype,
                spacing_type=spacing_type,
                spacing=spacing,
                image_keys=image_keys,
                label_keys=label_keys,
                additional_keys=additional_keys,
                select_channel=select_channel,
                cached=cached,
            )

    def __call__(self, img: dict, fixed_modality: Optional[str] = None) -> dict:
        modality = (fixed_modality or img["class"]).lower()
        if modality not in SUPPORT_MODALITIES:
            warnings.warn(
                f"Modality {modality} not in {SUPPORT_MODALITIES}; "
                "falling back to 'mri' transform pipeline."
            )
            modality = "mri"
        return self.transform_dict[modality](img)
