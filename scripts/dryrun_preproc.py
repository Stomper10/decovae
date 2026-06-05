#!/usr/bin/env python3
"""Pre-cache dry-run QC: run the FULL preprocessing pipeline on a few samples
per cohort x modality cell and save before/after montages — to eyeball every
cell and decide rigid-to-MNI152 (A) vs center-crop (B) BEFORE the ~64k batch.

Pipeline per volume (modality-aware):
  load -> (skull-strip antspynet[modality], skip if already stripped)
       -> ANTs N4 (mask-aware)
       -> A: rigid register to MNI152-1mm brain, warp (single interpolation)
          B: reorient RAS -> 1mm -> 192^3 center crop (MONAI)
       -> per-volume MASKED percentile [0,99.5]->[0,1]  (T1c: upper 99.9)
BraTS: already stripped + SRI24-registered -> skip strip & rigid (crop only),
       light boundary erosion to standardize mask edge.
Outputs (preproc_validation/):
  montage_dryrun_prepost.png  — all cells: RAW | A(rigid) axial/cor/sag
  montage_dryrun_rigidAB.png  — raw cohorts: B vs A, 2 subjects (pose check)
  dryrun_stats.csv
"""
import os, csv, tempfile, warnings
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ants, antspynet
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ResizeWithPadOrCropd)

warnings.filterwarnings("ignore")
CSV = "/data/wonyoungjang/decovae/csv_files"
OUT = "/data/wonyoungjang/decovae/preproc_validation"
os.makedirs(OUT, exist_ok=True)
N = 2          # samples per cell
SIZE = 192
BRATS_ROOT = "/data/wonyoungjang/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

ANTS_MOD = {"T1": "t1", "T1c": "t1", "T2": "t2", "FLAIR": "flair"}
RAW_COHORTS = {"ukb", "ixi", "hcp", "adni", "oasis"}   # need strip + rigid

geomB = Compose([                                       # B: geometry only
    LoadImaged(keys="image", image_only=False),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ResizeWithPadOrCropd(keys="image", spatial_size=(SIZE, SIZE, SIZE)),
])


def crop_pad(arr, size=SIZE):
    """Center crop/pad a 3D array to size^3 (pure indexing, no interpolation)."""
    out = np.zeros((size, size, size), dtype=np.float32)
    src_sl, dst_sl = [], []
    for n in arr.shape:
        if n >= size:
            s = (n - size) // 2; src_sl.append(slice(s, s + size)); dst_sl.append(slice(0, size))
        else:
            s = (size - n) // 2; src_sl.append(slice(0, n)); dst_sl.append(slice(s, s + n))
    out[tuple(dst_sl)] = arr[tuple(src_sl)]
    return out


def norm_pct(arr, upper=99.5):
    """Per-volume masked percentile: stats from brain (nonzero) voxels only."""
    brain = arr[arr > 0]
    if brain.size == 0:
        return arr.astype(np.float32)
    lo, hi = np.percentile(brain, [0.0, upper])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def get_mni_brain():
    cache = f"{OUT}/mni152_1mm_brain.nii.gz"
    if os.path.exists(cache):
        return ants.image_read(cache)
    mni = ants.image_read(ants.get_ants_data("mni"))
    prob = antspynet.brain_extraction(mni, modality="t1", verbose=False)
    brain = mni * ants.threshold_image(prob, 0.5, 1.0, 1, 0)
    ants.image_write(brain, cache)
    return brain


def preprocess(src, modality, cohort, mni_brain):
    """Return (rawA_axial, arrB, arrA) where arrB/arrA are 192^3 in [0,1]."""
    img = ants.image_read(src)
    raw_mid = img.numpy()[:, :, img.shape[2] // 2]
    is_brats = (cohort == "brats")
    upper = 99.9 if modality == "T1c" else 99.5

    if is_brats:
        mask = ants.threshold_image(img, 1e-6, 1e12, 1, 0)
        mask = ants.morphology(mask, operation="erode", radius=1)   # boundary std
        brain = img * mask
    else:
        prob = antspynet.brain_extraction(img, modality=ANTS_MOD[modality], verbose=False)
        mask = ants.threshold_image(prob, 0.5, 1.0, 1, 0)
        brain = img * mask
    n4 = ants.n4_bias_field_correction(brain, mask=mask)

    # B: MONAI geometry (RAS -> 1mm -> 192^3 center crop)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as t:
        tmp = t.name
    ants.image_write(n4, tmp)
    ob = geomB({"image": tmp})["image"]
    arrB = norm_pct(np.asarray(ob[0]), upper)
    os.remove(tmp)

    # A: rigid -> MNI152 brain (single interpolation), then 192^3 crop (indexing)
    if is_brats:                       # already SRI24-registered -> crop only
        arrA = arrB.copy()
    else:
        reg = ants.registration(fixed=mni_brain, moving=n4, type_of_transform="Rigid")
        warped = ants.apply_transforms(fixed=mni_brain, moving=n4,
                                       transformlist=reg["fwdtransforms"])
        arrA = norm_pct(crop_pad(np.asarray(warped.numpy())), upper)
    return raw_mid, arrB, arrA


def load_cells():
    cells = []   # (cell_name, cohort, modality, [src_paths])
    pm = list(csv.DictReader(open(f"{CSV}/pooled_manifest_train.csv")))
    seen = {}
    for r in pm:
        key = (r["cohort"], r["modality"])
        seen.setdefault(key, []).append(r["src_path"])
    for (coh, mod), paths in seen.items():
        cells.append((f"{coh}_{mod}", coh, mod, paths[:N]))
    # add brats_T1c (not in pooled manifest)
    bt1c = list(csv.DictReader(open(f"{CSV}/brats_T1c_train.csv")))[:N]
    cells.append(("brats_T1c", "brats", "T1c",
                  [os.path.join(BRATS_ROOT, r["rel_path"]) for r in bt1c]))
    return sorted(cells)


def main():
    mni = get_mni_brain()
    print("MNI brain template ready:", mni.shape, flush=True)
    cells = load_cells()
    results = {}   # cell -> list of (raw, B, A)
    stats = []
    for cell, coh, mod, paths in cells:
        results[cell] = []
        for j, src in enumerate(paths):
            try:
                raw, B, A = preprocess(src, mod, coh, mni)
                results[cell].append((raw, B, A))
                stats.append([cell, j, os.path.basename(src),
                              f"{A.min():.3f}/{A.max():.3f}/{A.mean():.3f}",
                              f"{(A>1e-6).mean()*100:.1f}%"])
                print(f"  {cell}[{j}] OK  nz={float((A>1e-6).mean())*100:.1f}%", flush=True)
            except Exception as e:
                stats.append([cell, j, os.path.basename(src), f"ERR:{e}", ""])
                print(f"  {cell}[{j}] ERR {e}", flush=True)

    # montage 1: prepost — rows=cells, cols=[RAW, A-axial, A-coronal, A-sagittal]
    cell_names = [c for c in results if results[c]]
    fig, ax = plt.subplots(len(cell_names), 4, figsize=(4 * 2.2, len(cell_names) * 2.2))
    for ri, cell in enumerate(cell_names):
        raw, B, A = results[cell][0]
        mid = [A.shape[0] // 2, A.shape[1] // 2, A.shape[2] // 2]
        imgs = [raw, A[mid[0], :, :], A[:, mid[1], :], A[:, :, mid[2]]]
        for ci, im in enumerate(imgs):
            a = ax[ri, ci]; a.imshow(np.rot90(im), cmap="gray", vmin=(None if ci == 0 else 0), vmax=(None if ci == 0 else 1))
            a.axis("off")
            if ci == 0: a.set_ylabel(cell, fontsize=8, rotation=0, labelpad=32, va="center")
            if ri == 0: a.set_title(["RAW", "A axial", "A coronal", "A sagittal"][ci], fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/montage_dryrun_prepost.png", dpi=88, bbox_inches="tight"); plt.close()
    print(f"montage: {OUT}/montage_dryrun_prepost.png", flush=True)

    # montage 2: rigid A vs B (raw cohorts only), 2 subjects, mid-axial
    raw_cells = [c for c in cell_names if c.split("_")[0] in RAW_COHORTS and len(results[c]) >= 2]
    fig, ax = plt.subplots(len(raw_cells), 4, figsize=(4 * 2.2, len(raw_cells) * 2.2))
    if len(raw_cells) == 1: ax = ax[None, :]
    for ri, cell in enumerate(raw_cells):
        (_, B0, A0), (_, B1, A1) = results[cell][0], results[cell][1]
        cols = [B0, B1, A0, A1]
        for ci, vol in enumerate(cols):
            m = vol.shape[2] // 2
            a = ax[ri, ci]; a.imshow(np.rot90(vol[:, :, m]), cmap="gray", vmin=0, vmax=1); a.axis("off")
            if ci == 0: a.set_ylabel(cell, fontsize=8, rotation=0, labelpad=32, va="center")
            if ri == 0: a.set_title(["B(crop) #1", "B(crop) #2", "A(rigid) #1", "A(rigid) #2"][ci], fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/montage_dryrun_rigidAB.png", dpi=88, bbox_inches="tight"); plt.close()
    print(f"montage: {OUT}/montage_dryrun_rigidAB.png", flush=True)

    with open(f"{OUT}/dryrun_stats.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cell", "i", "file", "min/max/mean", "%nonzero"]); w.writerows(stats)
    print("=== DRYRUN_DONE ===", flush=True)


if __name__ == "__main__":
    main()
