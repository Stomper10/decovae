#!/usr/bin/env python3
"""Single-GPU OOM sweep for the pooled VAE.

For each (patch_size, batch_size) it builds the REAL training stack — the MAISI
autoencoder (from configs/pooled/model_fm.json), the PatchDiscriminator, and the
3D PerceptualLoss — then runs one full bf16 forward+backward+step for BOTH the
generator and the discriminator (mirroring train_VAE.py), and records peak GPU
memory. The largest batch that does not OOM at each patch is the per-GPU cap.

Use the result to set stage1/stage2 patch_size + batch_size + grad_accum, then
derive max_train_steps = epochs * ceil(N_train / (batch*gpus*accum)).

Run on ONE h100 (see scripts/oom_sweep.sh). Does not touch the cache or train.
"""
import argparse
import json
from argparse import Namespace

import torch
from torch.nn import L1Loss
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import define_instance

PATCHES = [96, 128, 160, 192]
MAX_BATCH = 8          # search batch in 1..MAX_BATCH per patch
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
    a = ap.parse_args()
    assert torch.cuda.is_available(), "need a GPU"
    device = torch.device("cuda:0")
    model_cfg = json.load(open(a.model_config_path))
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}  total={total_mem:.0f}GB  dtype={DTYPE}", flush=True)

    l1 = L1Loss()
    adv = PatchAdversarialLoss(criterion="least_squares")
    results = {}
    for patch in PATCHES:
        max_ok, peak_ok = 0, 0.0
        for bs in range(1, MAX_BATCH + 1):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                ae, disc, perceptual = build_models(model_cfg, device)
                opt_g = torch.optim.AdamW(ae.parameters(), lr=1e-4)
                opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)
                x = torch.rand(bs, 1, patch, patch, patch, device=device)
                one_step(ae, disc, perceptual, l1, adv, opt_g, opt_d, x)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1024**3
                max_ok, peak_ok = bs, peak
                print(f"  patch {patch:3d}^3  bs {bs}  OK   peak={peak:.1f}GB", flush=True)
                del ae, disc, perceptual, opt_g, opt_d, x
            except torch.cuda.OutOfMemoryError:
                print(f"  patch {patch:3d}^3  bs {bs}  OOM", flush=True)
                del ae, disc, perceptual
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"  patch {patch:3d}^3  bs {bs}  ERR {repr(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
                break
        results[patch] = (max_ok, peak_ok)
    print("\n===== OOM SWEEP RESULT (per single GPU, bf16) =====")
    print(f"{'patch':>8} {'max_batch':>10} {'peak@max(GB)':>14}")
    for p in PATCHES:
        mb, pk = results[p]
        print(f"{p:>6}^3 {mb:>10} {pk:>14.1f}")


if __name__ == "__main__":
    main()
