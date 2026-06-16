"""Extract 3D-MedDiffusion PatchVolume latents (UKB c=4 nii OR pooled .npy cache).

This is the DecoVAE-side wrapper around the upstream Phase 3 step
(`external/3d_meddiff/train/generate_training_latent.py`). The upstream script
assumes the upstream `{class: dir}` JSON schema, no intensity normalization,
and 64-divisible volumes — none of which hold for our UKB MNI data. We
reproduce the *training-time* preprocessing exactly and add a MAISI-aligned
resize so the latent grid matches `extract_emb.py`:

POOLED (.npy) PATH: when a data-json entry ends in `.npy` it is the preprocessed
pooled cache (fp16 192^3, already percentile-normalized to [0,1] — the SAME
input the pooled VAE/3DMD training consumed). We load it straight to a tensor and
SKIP both the RescaleIntensity AND the round-128 resize: 192 is already a
multiple of patch_size=64, so patch_encode tiles cleanly into a [8, 48, 48, 48]
latent (same 48^3 spatial grid as the pooled MAISI latent; only channels differ).
Resizing 192->256 would change the latent geometry, so it must NOT run for .npy.
Since stage2 freezes encoder+pre_vq_conv+codebook, these latents are identical
whether extracted from the stage1 or the final stage2 ckpt (extract from stage1).

UKB (.nii) PATH (unchanged):

  1. explicit-split JSON  {"train": [paths], "val": [paths]}  (our schema)
  2. intensity:  tio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5,99.5))
                 — identical to the training dataset (patches/3d_meddiff/vqgan_4x.patch)
  3. resize:     each dim rounded to nearest multiple of 128 (trilinear), so
                 182x218x182 -> 128x256x128, matching extract_emb.py:round_number.
                 128/256/128 are all multiples of patch_size=64, so patch_encode
                 tiles cleanly into a [8, 32, 64, 32] latent (same spatial grid
                 as the 4-channel MAISI latent — only the channel count differs).
  4. encode:     AE.patch_encode(x, quantize=False)  -> continuous pre-quant
                 embedding `h`, then rescale by codebook min/max to [-1, 1].
                 This is exactly what generate_training_latent.py feeds to the
                 BiFlowNet (Phase 4), so the cached .npy doubles as the Phase 4
                 training input and the geometry/stat analysis source.

Output: one float32 .npy per subject, shape [C=8, D, H, W], under --out-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchio as tio

# Import the upstream AE from the vendored tree.
_EXT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "3d_meddiff"))
sys.path.insert(0, _EXT)
from AutoEncoder.model.PatchVolume import patchvolumeAE  # noqa: E402


def round_number(n: int, base: int = 128) -> int:
    """Round n to the nearest positive multiple of base (extract_emb.py parity)."""
    return int(max(round(float(n) / float(base)), 1.0) * base)


def subject_id(path: str) -> str:
    # pooled .npy cache: basename without ext IS the cache_key (e.g. ukb_1907867_T1)
    if path.endswith(".npy"):
        return os.path.splitext(os.path.basename(path))[0]
    # /data/.../20252_unzip/<SUBJECT_DIR>/T1/T1_brain_to_MNI.nii.gz  -> <SUBJECT_DIR>
    parts = path.rstrip("/").split("/")
    return parts[-3] if len(parts) >= 3 else os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-json", required=True, help="explicit-split data json")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--ae-ckpt", required=True, help="PatchVolume Lightning .ckpt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--resize-base", type=int, default=128)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap number of volumes (smoke test)")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this process handles files[shard_index::num_shards]")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    with open(args.data_json) as f:
        files = json.load(f)[args.split]
    if args.max_samples is not None:
        files = files[: args.max_samples]
    # Deterministic round-robin shard so multiple GPUs split the work evenly.
    files = files[args.shard_index :: args.num_shards]
    out_split = os.path.join(args.out_dir, args.split)
    os.makedirs(out_split, exist_ok=True)
    print(f"[extract] split={args.split} shard={args.shard_index}/{args.num_shards} "
          f"files={len(files)} -> {out_split}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = patchvolumeAE.load_from_checkpoint(args.ae_ckpt).to(device).eval()
    cb_min = ae.codebook.embeddings.min()
    cb_max = ae.codebook.embeddings.max()
    print(f"[extract] AE loaded; codebook range [{cb_min.item():.4f}, {cb_max.item():.4f}]", flush=True)

    rescale = tio.RescaleIntensity(out_min_max=(0.0, 1.0), percentiles=(0.5, 99.5))

    done = 0
    for i, fp in enumerate(files):
        sid = subject_id(fp)
        outp = os.path.join(out_split, f"{sid}.npy")
        if os.path.isfile(outp):
            done += 1
            continue
        try:
            if fp.endswith(".npy"):
                # pooled cache: already [0,1] at 192^3 (mult. of 64). Mirror the
                # training loader's .npy branch (vqgan_4x.py): load straight to a
                # tensor, SKIP RescaleIntensity + the round-128 resize.
                arr = np.squeeze(np.load(fp)).astype("float32")
                data = torch.from_numpy(arr)[None]      # [1, D, H, W] in [0,1]
            else:
                img = tio.ScalarImage(fp)
                img = rescale(img)
                # round each spatial dim to a multiple of resize-base, then resize.
                shp = img.spatial_shape  # (W, H, D) in torchio order
                new_shape = tuple(round_number(int(s), args.resize_base) for s in shp)
                img = tio.Resize(new_shape, image_interpolation="linear")(img)
                data = img.data.to(torch.float32)       # [1, *new_shape]

            x = data * 2.0 - 1.0                     # match training [-1, 1]
            x = x.transpose(1, 3).transpose(2, 3)    # Singleres axis convention
            x = x.unsqueeze(0).to(device)            # [1, 1, D, H, W]

            with torch.no_grad():
                z = ae.patch_encode(x, quantize=False, patch_size=args.patch_size)
                z = (z - cb_min) / (cb_max - cb_min) * 2.0 - 1.0

            np.save(outp, z.squeeze(0).cpu().numpy().astype(np.float32))
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[extract] ERROR {sid} ({fp}): {e}", flush=True)
            continue

        if (i + 1) % 50 == 0 or i == 0:
            print(f"[extract] {i + 1}/{len(files)} (saved {done}) last={sid} "
                  f"shape={z.shape[1:]}", flush=True)

    print(f"[extract] DONE split={args.split}: saved/skipped {done}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
