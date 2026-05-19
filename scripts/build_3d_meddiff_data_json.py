"""Bridge DecoVAE dataset manifests into the 3D-MedDiffusion data.json format.

3D-MedDiffusion's ``VQGANDataset_4x`` upstream expects a JSON of the form
``{"<name>": "<dir>"}`` and globs ``*.nii.gz`` under each dir, taking the last
40 files as val.  Our patched copy (see ``patches/3d_meddiff/``) additionally
accepts explicit file lists with an explicit train/val split:

    {"train": ["/abs/path/case1.nii.gz", ...],
     "val":   ["/abs/path/caseN.nii.gz", ...]}

This script emits that schema so the 3D MedDiff baseline trains on exactly
the same split as ``train_VAE.py`` (fair comparison).

Usage:
    python scripts/build_3d_meddiff_data_json.py \\
        --dataset ukb_20252 \\
        --train-csv csv_files/ukb_20252_train.csv \\
        --valid-csv csv_files/ukb_20252_valid.csv \\
        --data-dir /data/wonyoungjang/20252_unzip \\
        --output configs/3d_meddiff/data_ukb.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import get_adapter


def _paths(adapter, csv_path, data_dir):
    manifest = adapter.load_manifest(csv_path, data_dir)
    return [entry["image"] for entry in manifest]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="Adapter name (ukb_20252 | ixi | brats).")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--data-dir", required=True,
                        help="Image data root; rel_path is joined against this.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    adapter = get_adapter(args.dataset)
    train_paths = _paths(adapter, args.train_csv, args.data_dir)
    val_paths = _paths(adapter, args.valid_csv, args.data_dir)

    payload = {"train": train_paths, "val": val_paths}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(train_paths)} train + {len(val_paths)} val paths to {args.output}")


if __name__ == "__main__":
    main()
