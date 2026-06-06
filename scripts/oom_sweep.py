#!/usr/bin/env python3
"""Single-GPU OOM + throughput sweep for the pooled VAE.

For each (patch_size, batch_size) it builds the REAL training stack — the MAISI
autoencoder (from configs/pooled/model_fm.json), the PatchDiscriminator, and the
3D PerceptualLoss — then runs WARMUP + TIMED full bf16 forward+backward+step for
BOTH the generator and the discriminator (mirroring train_VAE.py), and records
peak GPU memory (allocated AND reserved) plus steady-state throughput
(samples/s). The largest batch that does not OOM at each patch is the per-GPU cap.

Two outputs matter:
  * peak_reserved drives OOM headroom (caching-allocator footprint; > allocated).
  * samples/s vs batch answers SM-saturation: if samples/s keeps rising with
    batch, the GPU was under-utilized at small batch and a larger batch buys real
    wall-clock under an EPOCH-MATCHED budget (fixed data exposure). If samples/s
    plateaus, it's compute-bound and batch only changes the optimization.

NOTE: torch.compile is OFF here (real training compiles), so absolute mem is
conservative-high and absolute samples/s is a lower bound; the *relative* batch
scaling trend (the SM-saturation answer) carries over.

Use the result to set stage1(64^3)/stage2(128^3) patch_size + batch_size +
grad_accum, then derive max_train_steps = epochs * ceil(N_train / (batch*gpus*accum)).

Run on ONE h100 (see scripts/oom_sweep.sh). Does not touch the cache or train.
"""
import argparse
import json
import time
from argparse import Namespace

import torch
from torch.nn import L1Loss
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import define_instance

PATCHES = [64, 128]    # stage1=64^3, stage2=128^3 (MAISI-faithful curriculum)
MAX_BATCH = 32         # search batch in 1..MAX_BATCH per patch (OOM ends it early)
WARMUP = 2             # untimed steps (allocator/cudnn ramp)
TIMED = 6              # timed steps for steady-state samples/s
DTYPE = torch.bfloat16


def build_models(model_cfg, device):
    args = Namespace(**model_cfg)
    ae = define_instance(args, "autoencoder_def").to(device)
    disc = PatchDiscriminator(spatial_dims=args.spatial_dims, num_layers_d=3,
                              channels=32, in_channels=1, out_channels=1,
                              norm="INSTANCE").to(device)
    perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    return ae, disc, perceptual


def one_step(ae, disc, perceptual, l1, adv, opt_g, opt_d, x):
    # generator
    opt_g.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=DTYPE):
        recon, z_mu, z_sigma = ae(x)
        kl = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + 1e-8) - 1)
        kl = kl / x.shape[0]
        p = perceptual(recon.float(), x.float())
        logits_fake = disc(recon.contiguous().float())[-1]
        g_adv = adv(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = l1(recon, x) + 5e-4 * kl + 0.3 * p + 0.1 * g_adv
    loss_g.backward()
    opt_g.step()
    # discriminator
    opt_d.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=DTYPE):
        logits_fake = disc(recon.contiguous().detach())[-1]
        logits_real = disc(x.contiguous().detach())[-1]
        loss_d = 0.5 * (adv(logits_fake, target_is_real=False, for_discriminator=True)
                        + adv(logits_real, target_is_real=True, for_discriminator=True))
    loss_d.backward()
    opt_d.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config_path", default="configs/pooled/model_fm.json")
    ap.add_argument("--patches", default=",".join(str(p) for p in PATCHES),
                    help="comma-separated patch sizes")
    ap.add_argument("--max_batch", type=int, default=MAX_BATCH)
    a = ap.parse_args()
    patches = [int(p) for p in a.patches.split(",")]
    assert torch.cuda.is_available(), "need a GPU"
    device = torch.device("cuda:0")
    model_cfg = json.load(open(a.model_config_path))
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}  total={total_mem:.0f}GB  dtype={DTYPE}  "
          f"warmup={WARMUP} timed={TIMED}", flush=True)

    l1 = L1Loss()
    adv = PatchAdversarialLoss(criterion="least_squares")
    results = {}
    for patch in patches:
        # per-bs rows: (bs, peak_alloc, peak_reserved, samples_per_s, ms_per_step)
        rows = []
        for bs in range(1, a.max_batch + 1):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                ae, disc, perceptual = build_models(model_cfg, device)
                opt_g = torch.optim.AdamW(ae.parameters(), lr=1e-4)
                opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)
                x = torch.rand(bs, 1, patch, patch, patch, device=device)
                for _ in range(WARMUP):                       # untimed ramp
                    one_step(ae, disc, perceptual, l1, adv, opt_g, opt_d, x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()          # steady-state peak only
                t0 = time.perf_counter()
                for _ in range(TIMED):
                    one_step(ae, disc, perceptual, l1, adv, opt_g, opt_d, x)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                peak_a = torch.cuda.max_memory_allocated() / 1024**3
                peak_r = torch.cuda.max_memory_reserved() / 1024**3
                sps = (bs * TIMED) / dt
                ms = 1000.0 * dt / TIMED
                rows.append((bs, peak_a, peak_r, sps, ms))
                print(f"  patch {patch:3d}^3  bs {bs:2d}  OK   "
                      f"alloc={peak_a:5.1f}GB  reserved={peak_r:5.1f}GB  "
                      f"{sps:6.2f} samp/s  {ms:7.1f} ms/step", flush=True)
                del ae, disc, perceptual, opt_g, opt_d, x
            except torch.cuda.OutOfMemoryError:
                print(f"  patch {patch:3d}^3  bs {bs:2d}  OOM", flush=True)
                del ae, disc, perceptual
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"  patch {patch:3d}^3  bs {bs:2d}  ERR {repr(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
                break
        results[patch] = rows
    print("\n===== OOM + THROUGHPUT SWEEP (per single GPU, bf16, no-compile) =====")
    for p in patches:
        rows = results[p]
        if not rows:
            print(f"\npatch {p}^3: no batch fit"); continue
        cap = rows[-1][0]
        print(f"\npatch {p}^3  (max_batch={cap})")
        print(f"  {'bs':>3} {'alloc(GB)':>10} {'reserved(GB)':>13} "
              f"{'samp/s':>8} {'ms/step':>9} {'samp/s/bs':>10}")
        for bs, pa, pr, sps, ms in rows:
            # samp/s/bs flat => compute-bound; rising-then-flat => saturation point
            print(f"  {bs:>3} {pa:>10.1f} {pr:>13.1f} {sps:>8.2f} {ms:>9.1f} {sps/bs:>10.3f}")


if __name__ == "__main__":
    main()
