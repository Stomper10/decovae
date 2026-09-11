#!/usr/bin/env python3
"""Build TSV manifest for §3b tumor mode-averaging eval.

Rows: 1 real + 9 gen cells (3 arms × 3 configs: B-orig, A-orig, A-dft).

Columns (TSV): label  vol_src  gen_dir_or_-  max_cases
"""
from pathlib import Path
import os

STAGE_ROOT = Path(os.environ.get(
    "POOLED_OUTPUT_ROOT", "/leelabsg/data/wonyoungjang/decodata/pooled")) / "stage1"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "journal_plan" / ".tumor_mode_manifest.tsv"

ARMS = {
    "maisi": "pooled-maisi-kl8e4-eff32-s1",
    "sid":   "pooled-sid-cor50-kl8e4-eff32-s1",
    "vad":   "pooled-vad-cov1var1-kl8e4-eff32-s1",
}
CONFIGS = [
    # (label_suffix, decoder_suffix, cell_name)
    ("B_orig", "",              "g30_brats_FLAIR"),
    ("A_orig", "",              "g30_02_Acfg_orig_brats_FLAIR"),
    ("A_dft",  "-decft-n1.0",   "g30_02_Acfg_dft_brats_FLAIR"),
]

# match real N to a typical gen cell count so distribution comparisons are
# apples-to-apples; gen cells hold 500 each in this pack, so cap real at 500 too.
REAL_MAX = 500
GEN_MAX = 0  # 0 = all files in the cell dir (each cell has 500)

lines: list[str] = []
lines.append(f"real\treal\t-\t{REAL_MAX}")
for arm_short, arm_full in ARMS.items():
    for suffix, dec, cell in CONFIGS:
        base = arm_full + dec
        gen_dir = STAGE_ROOT / base / "cells" / cell / "outputs" / "volumes"
        if not gen_dir.exists():
            print(f"[skip] missing {gen_dir}")
            continue
        label = f"{arm_short}_{suffix}"
        lines.append(f"{label}\tgen\t{gen_dir}\t{GEN_MAX}")

OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT}  rows={len(lines)}")
