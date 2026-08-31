#!/usr/bin/env python3
"""Standalone FID mean/cov decomposition over ALREADY-CACHED .pt features (CPU, no GPU).

Loads real_features/ + recon_features/ (or gen_features/) *.pt (each a 3-tuple of
per-volume plane features xy/yz/zx), vstacks per plane, and reports FID split into:
  mean-term ||mu_r - mu_g||^2   (coherent/systematic bias, e.g. uniform blur)
  cov-term  tr(S_r + S_g - 2 sqrt(S_r S_g))   (spread/diversity mismatch)
Total == monai FIDMetric (reuses monai _cov/_sqrtm). No `patches` import, no GPU.

  python scripts/fid_decompose_cached.py <features_dir> [--synth recon_features|gen_features]
where <features_dir> contains real_features/ and the synth dir.
"""
import argparse, glob, os, sys, time
import torch
from monai.metrics.fid import _cov, _sqrtm


def frechet_terms(y_pred, y):
    y = y.double(); y_pred = y_pred.double()
    mu_x = torch.mean(y_pred, dim=0); sigma_x = _cov(y_pred, rowvar=False)
    mu_y = torch.mean(y, dim=0);      sigma_y = _cov(y, rowvar=False)
    diff = mu_x - mu_y
    covmean = _sqrtm(sigma_x.mm(sigma_y))
    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma_x.size(0), dtype=mu_x.dtype) * 1e-6
        covmean = _sqrtm((sigma_x + offset).mm(sigma_y + offset))
    if torch.is_complex(covmean):
        covmean = covmean.real
    mean_term = float(diff.dot(diff))
    cov_term = float(torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * torch.trace(covmean))
    return mean_term + cov_term, mean_term, cov_term


def load_plane_stacks(fdir, max_vols=0):
    files = sorted(glob.glob(os.path.join(fdir, "*.pt")))
    if not files:
        sys.exit(f"no .pt in {fdir}")
    if max_vols and len(files) > max_vols:
        step = len(files) / max_vols            # even spacing (deterministic)
        files = [files[int(i * step)] for i in range(max_vols)]
    xy, yz, zx = [], [], []
    for i, f in enumerate(files):
        t = torch.load(f, weights_only=True, map_location="cpu")
        xy.append(t[0]); yz.append(t[1]); zx.append(t[2])
        if (i + 1) % 500 == 0:
            print(f"    loaded {i+1}/{len(files)}", flush=True)
    return torch.vstack(xy), torch.vstack(yz), torch.vstack(zx), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_dir")
    ap.add_argument("--synth", default="recon_features")
    ap.add_argument("--max_vols", type=int, default=0, help="subsample this many volumes (0=all); mean/cov %% is stable in N")
    args = ap.parse_args()

    t0 = time.time()
    real_dir = os.path.join(args.features_dir, "real_features")
    synth_dir = os.path.join(args.features_dir, args.synth)
    print(f"[load] real={real_dir}")
    rxy, ryz, rzx, nr = load_plane_stacks(real_dir, args.max_vols)
    print(f"[load] synth={synth_dir}")
    sxy, syz, szx, ns = load_plane_stacks(synth_dir, args.max_vols)
    print(f"[load] real vols={nr} synth vols={ns}  planes real {tuple(rxy.shape)}/{tuple(ryz.shape)}/{tuple(rzx.shape)}  ({time.time()-t0:.0f}s)")

    planes = (("XY", sxy, rxy), ("YZ", syz, ryz), ("ZX", szx, rzx))
    m_sum = c_sum = t_sum = 0.0
    print(f"\n{'plane':>6} {'total':>10} {'mean-term':>12} {'(%)':>7} {'cov-term':>12} {'(%)':>7}")
    for name, s, r in planes:
        tot, m, c = frechet_terms(s, r)
        m_sum += m; c_sum += c; t_sum += tot
        print(f"{name:>6} {tot:>10.4f} {m:>12.4f} {100*m/tot:>6.1f}% {c:>12.4f} {100*c/tot:>6.1f}%")
    print(f"{'AVG':>6} {t_sum/3:>10.4f} {m_sum/3:>12.4f} {100*m_sum/t_sum:>6.1f}% {c_sum/3:>12.4f} {100*c_sum/t_sum:>6.1f}%")
    print(f"\n[done] {time.time()-t0:.0f}s   (mean-term% high => coherent/systematic bias e.g. blur; cov-term% high => spread/diversity mismatch)")


if __name__ == "__main__":
    main()
