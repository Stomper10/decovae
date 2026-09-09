#!/usr/bin/env python3
"""Group Phase 3 latents into per-modality class directories for BiFlowNet.

WHY THIS IS NEEDED. 3D MedDiff's Phase 4 is CONDITIONAL, and the condition comes from
the data json's KEYS: Singleres_dataset reads `{"0": base, "1": base, ...}` and hands
`int(key)` to the model as `cls_idx`. Phase 3 writes a flat train/ and val/, so the
class structure has to be built on top of it.

Conditioning on MODALITY (0=T1, 1=T2, 2=FLAIR) is what makes the baseline comparable
at all: our own gFID is reported per modality slice (all_T1 / all_T2 / all_FLAIR), and
an unconditional 3DMD could only be scored against the pooled mixture. It is still not
a like-for-like conditioning -- our arms take a 13-D vector (modality + sex + dx + age
+ cdrsb) against 3DMD's single categorical -- but that is 3DMD's own architecture and
the plan says to run their recipe verbatim. State it in the paper rather than fix it.

T1c IS EXCLUDED, matching the diffusion modality vocabulary. brats_T1c is vae_only=1:
it exists only in BraTS and is 100% dx=tumor, so it sits outside {T1,T2,FLAIR} for
every generative comparison we make.

SYMLINKS, NOT COPIES. Phase 1's latent set was 174 GB; duplicating it per class would
cost that again for nothing. Each class dir holds links into the Phase 3 output.

Usage:
  python3 scripts/make_3dmd_phase4_classes.py \
      --latent_dir /data/wonyoungjang/decodata/3d_meddiff/pooled_s2/latents
"""
from __future__ import annotations

import argparse
import json
import os

MODALITIES = ["T1", "T2", "FLAIR"]        # index == conditioning class id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", required=True,
                    help="Phase 3 output holding train/ and val/.")
    ap.add_argument("--out_root", default=None,
                    help="Where the class dirs go (default: <latent_dir>/by_class).")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--json_out", default="configs/3d_meddiff/SingleRes_pooled.json")
    args = ap.parse_args()

    src = os.path.join(args.latent_dir, args.split)
    if not os.path.isdir(src):
        raise SystemExit(f"no such split dir: {src}")
    out_root = args.out_root or os.path.join(args.latent_dir, "by_class")

    # Singleres_dataset appends '_latents' to every json value, so the json names the
    # BASE and the files must live in <base>_latents.
    mapping, counts = {}, {}
    for idx, mod in enumerate(MODALITIES):
        base = os.path.join(out_root, args.split, f"cls{idx}_{mod}")
        d = base + "_latents"
        os.makedirs(d, exist_ok=True)
        n = 0
        for f in os.scandir(src):
            if not f.name.endswith(".npy"):
                continue
            # cache_key is "<cohort>_<subject>_<MODALITY>"; T1c must not match T1.
            if f.name[:-4].rsplit("_", 1)[-1] != mod:
                continue
            link = os.path.join(d, f.name)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(f.path), link)
            n += 1
        mapping[str(idx)] = base
        counts[mod] = n

    total = sum(counts.values())
    print(f"split={args.split}  total {total}")
    for i, m in enumerate(MODALITIES):
        print(f"  class {i} = {m:<6} {counts[m]:>6}")
    if total == 0:
        raise SystemExit("no latents matched — check --latent_dir")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"\nwrote {args.json_out}")
    print(json.dumps(mapping, indent=4))


if __name__ == "__main__":
    main()
