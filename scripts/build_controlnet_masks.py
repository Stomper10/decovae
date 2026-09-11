#!/usr/bin/env python3
"""BraTS tumour masks in the POOLED preprocessing space, for mask-conditional ControlNet.

WHY THIS IS NEEDED. The pooled latents were encoded from the cached 192^3 volumes,
which went through rigid -> MNI152 + center crop/pad (scripts/preproc_pipeline.py).
The BraTS segmentations only exist in native space (240x240x155). The first pooled
ControlNet interpolated the native mask straight to the latent grid, which gets the
size right and the tumour position wrong -- nothing maps native onto the cache frame.

WHY REGISTER TO THE CACHE, NOT RE-RUN THE PIPELINE. The pipeline did not save its
transforms, and ANTs rigid registration samples randomly with no seed, so re-running
it gives a different transform. Checked on BraTS-GLI-00000-000 T1:

    re-run pipeline (native -> MNI template)   corr vs cached .npy 0.9194
    register native -> cached .npy              corr vs cached .npy 1.0000

and the two tumour masks differ (Dice 0.946, ~1 voxel centroid shift). Registering
to the cached volume recovers the transform that actually produced it, so the mask
lands in exactly the space the latent came from.

HOW. The cached array is un-cropped back onto the MNI grid and given the template's
origin/spacing/direction -- exactly the geometry of the warped image before
crop_pad, since the cache is `crop_pad(warped.numpy())`. The moving image is built
the way the pipeline built it (RAS reorient, threshold + 1-voxel erosion, N4); N4
matters, without it the re-warped image only reaches corr ~0.90. The segmentation
(same native grid as every BraTS modality) is warped with genericLabel and crop_pad'ed.

One mask PER (subject, modality): each modality was registered independently, and
cached T1/T2/FLAIR of one subject differ by up to ~1.6 mm. A per-modality mask is
exact for that modality's latent; a shared one would not be.

QC. `corr` is the re-warped native image against the cached volume inside the brain.
It should be ~1.0; anything under --min_corr is recorded as a failure and no mask is
written, so a bad registration cannot reach training.

    python scripts/build_controlnet_masks.py --out_root /data/.../controlnet_masks \\
        --workers 28 [--shard 0 --nshards 1] [--limit 4]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_T = os.environ.get("PP_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse
import csv
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import preproc_pipeline as P

CSV_DIR = "/data/wonyoungjang/decovae/csv_files"
SPLITS = ("train", "valid", "test")
DIFFUSION_MODALITIES = ("T1", "T2", "FLAIR")      # T1c is vae_only


def uncrop(arr192, shape):
    """Inverse placement of P.crop_pad onto the pre-crop grid (cropped margins -> 0)."""
    out = np.zeros(shape, np.float32)
    src, dst = [], []
    for n in shape:
        if n >= P.SIZE:
            s = (n - P.SIZE) // 2; dst.append(slice(s, s + P.SIZE)); src.append(slice(0, P.SIZE))
        else:
            s = (P.SIZE - n) // 2; dst.append(slice(0, n)); src.append(slice(s, s + n))
    out[tuple(dst)] = arr192[tuple(src)]
    return out


def _corr(a, b):
    m = (a > 0) | (b > 0)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def process_one(task):
    row, cache_root, out_root, mni_path, min_corr = task
    key = row["cache_key"]
    out_fp = os.path.join(out_root, f"{key}_seg.npy")
    base = {"cache_key": key, "modality": row["modality"]}
    if os.path.exists(out_fp):
        return {**base, "status": "skip"}
    t0 = time.time()
    try:
        import ants
        src = row["src_path"]
        eid = os.path.basename(os.path.dirname(src))
        seg_fp = os.path.join(os.path.dirname(src), f"{eid}-seg.nii.gz")

        mni = P.get_mni_brain(mni_path)
        cached = np.load(os.path.join(cache_root, f"{key}.npy")).astype(np.float32).squeeze()
        fixed = ants.from_numpy(uncrop(cached, mni.shape), origin=mni.origin,
                                spacing=mni.spacing, direction=mni.direction)

        # moving: exactly as preprocess_to_192 builds it for cohort == "brats"
        img = ants.reorient_image2(ants.image_read(src), orientation="RAS")
        mask = ants.morphology(ants.threshold_image(img, 1e-6, 1e12, 1, 0), operation="erode", radius=1)
        n4 = ants.n4_bias_field_correction(img * mask, mask=mask)

        xf = ants.registration(fixed=fixed, moving=n4, type_of_transform="Rigid")
        warped = P.norm_pct(P.crop_pad(ants.apply_transforms(
            fixed=fixed, moving=n4, transformlist=xf["fwdtransforms"]).numpy()),
            99.9 if row["modality"] == "T1c" else 99.5)
        corr = _corr(warped, cached)

        seg = ants.reorient_image2(ants.image_read(seg_fp), orientation="RAS")
        seg_w = P.crop_pad(ants.apply_transforms(
            fixed=fixed, moving=seg, transformlist=xf["fwdtransforms"],
            interpolator="genericLabel").numpy()).astype(np.uint8)
        nat = seg.numpy()
        stats = {**base, "corr": round(corr, 5), "secs": round(time.time() - t0, 1)}
        for lab in (1, 2, 3):
            stats[f"L{lab}_native"] = int((nat == lab).sum())
            stats[f"L{lab}_pooled"] = int((seg_w == lab).sum())
        if corr < min_corr:
            return {**stats, "status": "fail", "msg": f"corr {corr:.4f} < {min_corr}"}
        tmp = out_fp + ".tmp.npy"
        np.save(tmp, seg_w)
        os.replace(tmp, out_fp)                       # atomic: a killed worker leaves no half-mask
        return {**stats, "status": "ok"}
    except Exception as e:                            # noqa: BLE001 -- logged, never aborts the batch
        return {**base, "status": "fail", "msg": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="/data/wonyoungjang/decovae_cache")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_corr", type=float, default=0.99)
    a = ap.parse_args()

    rows = []
    for sp in SPLITS:
        d = pd.read_csv(os.path.join(CSV_DIR, f"pooled_manifest_{sp}.csv"))
        d = d[(d.cohort == "brats") & (d.vae_only == 0) & d.modality.isin(DIFFUSION_MODALITIES)]
        rows += d.assign(split=sp)[["cache_key", "src_path", "modality", "split"]].to_dict("records")
    rows = sorted(rows, key=lambda r: r["cache_key"])[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    mni_path = os.path.join(a.cache_root, "mni152_1mm_brain.nii.gz")
    os.makedirs(os.path.join(a.out_root, "brats"), exist_ok=True)
    print(f"[masks] shard {a.shard}/{a.nshards}: {len(rows)} volumes, workers={a.workers} "
          f"x {_T} threads, min_corr={a.min_corr}", flush=True)

    tasks = [(r, a.cache_root, a.out_root, mni_path, a.min_corr) for r in rows]
    results, n = [], {"ok": 0, "skip": 0, "fail": 0}
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, tasks, chunksize=2)):
            results.append(res); n[res["status"]] += 1
            if res["status"] == "fail":
                print(f"  FAIL {res['cache_key']}: {res.get('msg')}", flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                print(f"  {i+1}/{len(rows)}  {n}", flush=True)

    qc_fp = os.path.join(a.out_root, f"_qc_shard{a.shard}.csv")
    keys = sorted({k for r in results for k in r})
    with open(qc_fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(results)
    done = [r for r in results if r["status"] == "ok"]
    if done:
        c = np.array([r["corr"] for r in done])
        print(f"corr over ok: min {c.min():.4f}  p1 {np.percentile(c, 1):.4f}  median {np.median(c):.4f}")
    print(f"DONE shard {a.shard}: {n} -> {qc_fp}", flush=True)


if __name__ == "__main__":
    main()
