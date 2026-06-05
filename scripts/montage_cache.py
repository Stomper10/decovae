#!/usr/bin/env python3
"""Post-cache QC montage: read the FINISHED preprocessing cache (.npy, 192^3 fp16
in [0,1]) and tile a few samples per cohort x modality cell, with labels, so the
final cached corpus can be eyeballed cell-by-cell.

No preprocessing here — the cache is already 192^3 [0,1]; we just load + slice
mid-planes. Picks deterministic-but-varied samples per cell (evenly spaced over
the sorted manifest rows so it's not all the same site/subject).

Output: preproc_validation/montage_cache_<plane>.png  (one per plane)
        preproc_validation/montage_cache_grid.png      (all cells, 3 planes, 1 sample)
"""
import os, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV   = "/data/wonyoungjang/decovae/csv_files"
CACHE = "/data/wonyoungjang/decovae_cache"
OUT   = "/data/wonyoungjang/decovae/preproc_validation"
os.makedirs(OUT, exist_ok=True)
N_PER_CELL = 2          # samples per cell shown in the per-plane montages
SIZE = 192


def load_cells():
    """cell -> ordered list of cache_keys (across all splits)."""
    cells = collections.OrderedDict()
    for sp in ("train", "valid", "test"):
        with open(f"{CSV}/pooled_manifest_{sp}.csv") as f:
            for r in csv.DictReader(f):
                cells.setdefault((r["cohort"], r["modality"]), []).append(r["cache_key"])
    return collections.OrderedDict(sorted(cells.items()))


def pick(keys, n):
    """n evenly spaced keys (varied site/subject), only those whose .npy exists."""
    out, i = [], 0
    if not keys:
        return out
    idxs = [int(round(t)) for t in np.linspace(0, len(keys) - 1, max(n * 3, n))]
    seen = set()
    for j in idxs:
        if j in seen:
            continue
        seen.add(j)
        p = f"{CACHE}/{keys[j]}.npy"
        if os.path.exists(p):
            out.append(keys[j])
        if len(out) >= n:
            break
    return out


def mid_planes(arr):
    """3 orthogonal mid-slices (axial, coronal, sagittal), oriented for display."""
    c = SIZE // 2
    ax = np.rot90(arr[:, :, c])      # axial (z mid)
    co = np.rot90(arr[:, c, :])      # coronal (y mid)
    sa = np.rot90(arr[c, :, :])      # sagittal (x mid)
    return ax, co, sa


def main():
    cells = load_cells()
    plane_names = ["axial", "coronal", "sagittal"]

    # --- per-plane montages: rows = cells, cols = N_PER_CELL samples -----------
    for pi, pname in enumerate(plane_names):
        nrows = len(cells)
        fig, axes = plt.subplots(nrows, N_PER_CELL, figsize=(2.4 * N_PER_CELL, 2.4 * nrows))
        if nrows == 1:
            axes = axes[None, :]
        for ri, (cell, keys) in enumerate(cells.items()):
            chosen = pick(keys, N_PER_CELL)
            for ci in range(N_PER_CELL):
                ax = axes[ri, ci]
                ax.set_xticks([]); ax.set_yticks([])
                if ci < len(chosen):
                    arr = np.load(f"{CACHE}/{chosen[ci]}.npy").astype(np.float32)
                    img = mid_planes(arr)[pi]
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                    sid = chosen[ci].split("/")[-1]
                    ax.set_title(sid, fontsize=6)
                else:
                    ax.axis("off")
                if ci == 0:
                    ax.set_ylabel(f"{cell[0]}/{cell[1]}\n(n={len(keys)})",
                                  fontsize=9, rotation=0, ha="right", va="center",
                                  labelpad=38, fontweight="bold")
        fig.suptitle(f"Cache QC — {pname} mid-plane (192^3 [0,1])  |  13 cells x {N_PER_CELL} samples",
                     fontsize=12, y=0.997)
        fig.tight_layout(rect=[0.04, 0, 1, 0.99])
        fp = f"{OUT}/montage_cache_{pname}.png"
        fig.savefig(fp, dpi=110, bbox_inches="tight"); plt.close(fig)
        print("wrote", fp, flush=True)

    # --- combined grid: rows = cells, cols = 3 planes of 1 representative ------
    nrows = len(cells)
    fig, axes = plt.subplots(nrows, 3, figsize=(2.4 * 3, 2.2 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    for ri, (cell, keys) in enumerate(cells.items()):
        chosen = pick(keys, 1)
        arr = np.load(f"{CACHE}/{chosen[0]}.npy").astype(np.float32) if chosen else np.zeros((SIZE,)*3)
        planes = mid_planes(arr)
        for ci in range(3):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(planes[ci], cmap="gray", vmin=0, vmax=1)
            if ri == 0:
                ax.set_title(plane_names[ci], fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{cell[0]}/{cell[1]}\n(n={len(keys)})",
                              fontsize=9, rotation=0, ha="right", va="center",
                              labelpad=38, fontweight="bold")
    fig.suptitle("Cache QC grid — all 13 cohort x modality cells (1 sample, 3 planes)",
                 fontsize=12, y=0.997)
    fig.tight_layout(rect=[0.05, 0, 1, 0.99])
    fp = f"{OUT}/montage_cache_grid.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", fp, flush=True)


if __name__ == "__main__":
    main()
