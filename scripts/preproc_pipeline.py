#!/usr/bin/env python3
"""Canonical deterministic preprocessing for the pooled foundation corpus.

One function: source NIfTI -> 192^3 @ 1mm, [0,1], fp16-ready float32 array.
Modality-aware, single-interpolation rigid alignment. Used by both the
dry-run QC (scripts/dryrun_preproc.py) and the full-batch cache writer
(scripts/preprocess_cache.py).

Pipeline (per §3 of the plan):
  RAS reorient (ants) ->
  skull-strip antspynet brain_extraction(modality) [SKIP if already stripped] ->
  ANTs N4 (mask-aware) ->
  rigid 6-DOF -> MNI152-1mm brain, warp (single interpolation) [reg='rigid']
     OR reorient/1mm/center-crop (reg='crop') ->
  192^3 center crop/pad (pure indexing) ->
  per-volume MASKED percentile [0, upper] -> [0,1]  (T1c: upper=99.9 else 99.5)

ants/antspynet are imported lazily so the module is cheap to import and so
worker processes can force CPU (CUDA_VISIBLE_DEVICES="") before TF loads.
"""
import os
import numpy as np

SIZE = 192
ANTS_MOD = {"T1": "t1", "T1c": "t1", "T2": "t2", "FLAIR": "flair"}

_MNI = {"path": None, "img": None}   # per-process cache


def crop_pad(arr, size=SIZE):
    """Center crop/pad a 3D array to size^3 — pure indexing, no interpolation."""
    out = np.zeros((size, size, size), dtype=np.float32)
    src, dst = [], []
    for n in arr.shape:
        if n >= size:
            s = (n - size) // 2; src.append(slice(s, s + size)); dst.append(slice(0, size))
        else:
            s = (size - n) // 2; src.append(slice(0, n)); dst.append(slice(s, s + n))
    out[tuple(dst)] = arr[tuple(src)].astype(np.float32)
    return out


def norm_pct(arr, upper=99.5):
    """Per-volume masked percentile: stats from brain (nonzero) voxels only."""
    brain = arr[arr > 0]
    if brain.size == 0:
        return arr.astype(np.float32)
    lo, hi = np.percentile(brain, [0.0, upper])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def get_mni_brain(mni_brain_path):
    """Load (and cache per-process) the brain-stripped MNI152 1mm template."""
    import ants
    if _MNI["path"] != mni_brain_path:
        _MNI["img"] = ants.image_read(mni_brain_path)
        _MNI["path"] = mni_brain_path
    return _MNI["img"]


def build_mni_brain(out_path):
    """One-time: brain-extract the ANTs MNI152 1mm template and write to out_path."""
    import ants, antspynet
    if os.path.exists(out_path):
        return out_path
    mni = ants.image_read(ants.get_ants_data("mni"))
    prob = antspynet.brain_extraction(mni, modality="t1", verbose=False)
    brain = mni * ants.threshold_image(prob, 0.5, 1.0, 1, 0)
    ants.image_write(brain, out_path)
    return out_path


def preprocess_to_192(src, modality, cohort, mni_brain_path, reg="rigid"):
    """Run the full pipeline. Returns a float32 192^3 array in [0,1].

    cohort=='brats' -> skip antspynet strip (already stripped), light boundary
    erosion to standardize the mask edge, then SAME rigid->MNI as everyone
    (BraTS native space is SRI24, not MNI -> re-align for one shared frame).
    """
    import ants, antspynet
    img = ants.reorient_image2(ants.image_read(src), orientation="RAS")
    upper = 99.9 if modality == "T1c" else 99.5

    if cohort == "brats":
        mask = ants.threshold_image(img, 1e-6, 1e12, 1, 0)
        mask = ants.morphology(mask, operation="erode", radius=1)
        brain = img * mask
    else:
        prob = antspynet.brain_extraction(img, modality=ANTS_MOD[modality], verbose=False)
        mask = ants.threshold_image(prob, 0.5, 1.0, 1, 0)
        brain = img * mask
    n4 = ants.n4_bias_field_correction(brain, mask=mask)

    if reg == "rigid":
        mni = get_mni_brain(mni_brain_path)
        xf = ants.registration(fixed=mni, moving=n4, type_of_transform="Rigid")
        warped = ants.apply_transforms(fixed=mni, moving=n4, transformlist=xf["fwdtransforms"])
        arr = crop_pad(np.asarray(warped.numpy()))
    else:  # crop: reorient RAS (done) -> 1mm -> center crop
        res = ants.resample_image(n4, (1.0, 1.0, 1.0), use_voxels=False, interp_type=0)
        arr = crop_pad(np.asarray(res.numpy()))
    return norm_pct(arr, upper)
