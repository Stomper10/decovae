"""Synthesize UKB-like volumes conditioned on age, for brain-age augmentation.

Pipeline (mirrors ``compute_metric.run_generation`` with eval_mode=real_vs_gen):

    age_list ──► reverse-diffuse latent UNet (meta_tensor=age) ──► VAE.decode ──► NIfTI

Output: ``<output_dir>/{volumes/*.nii.gz, synth_age_index.csv}``. The CSV
columns (``eid, rel_path, age``) match the schema expected by
``downstream.brain_age_dataset.make_dataset`` so the regressor can ingest
the synth dir via ``SYNTH_CSV`` / ``SYNTH_DIR`` on its launcher.

NOTE: this is a skeleton. It assumes the conditional UNet has been trained
with ``include_meta_input=True`` and ``meta_tensor=[norm_age]``. Until the
stage2 + UNet weights land, run with ``--dry_run`` to validate path wiring
without invoking the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.cuda.amp import autocast

from datasets import get_adapter
from scripts.utils import define_instance


def _normalize_age(age: float, age_min: float, age_max: float) -> float:
    return (age - age_min) / (age_max - age_min + 1e-12)


def _build_meta(cond_dict, cond_cfg, use_token_set, device, norm_age):
    """meta_tensor for one sample: token-set dict (cond_cat/cont/presence) when
    conditioning is enabled, else the legacy 1-D scalar age. Mirrors
    compute_metric.run_generation so train/inference conditioning match."""
    if not use_token_set:
        return torch.tensor([[norm_age]], dtype=torch.float16, device=device)
    from patches.token_set_encoder import encode_token_set
    ci, cv, pr = encode_token_set(cond_dict, cond_cfg["attributes"])
    return {
        "cond_cat": torch.tensor([ci], dtype=torch.long, device=device),
        "cond_cont": torch.tensor([cv], dtype=torch.float32, device=device),
        "cond_presence": torch.tensor([pr], dtype=torch.bool, device=device),
    }


def sample_age_distribution(real_csv: str, n: int, seed: int,
                            method: str = "stratified") -> np.ndarray:
    import pandas as pd
    ages = pd.read_csv(real_csv)["age"].to_numpy()
    rng = np.random.default_rng(seed)
    if method == "uniform":
        return rng.uniform(ages.min(), ages.max(), size=n)
    quantiles = np.quantile(ages, np.linspace(0, 1, 6))  # quintile bins
    samples = []
    per_bin = n // 5
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        samples.append(rng.uniform(lo, hi, size=per_bin))
    samples.append(rng.uniform(ages.min(), ages.max(), size=n - per_bin * 5))
    out = np.concatenate(samples)
    rng.shuffle(out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_adapter", default="ukb_20252")
    p.add_argument("--dataset_config_path", required=True)
    p.add_argument("--model_config_path", required=True,
                   help="model_fm.json with include_meta_input=True UNet variant.")
    p.add_argument("--inference_config_path", required=True,
                   help="diff_train_inf.json — provides noise_scheduler.")
    p.add_argument("--pretrained_vae_path", required=True)
    p.add_argument("--pretrained_unet_path", required=True)
    p.add_argument("--latent_stats_csv", required=True,
                   help="analysis/latent_stats.csv produced by extract_emb.py")
    p.add_argument("--real_csv", required=True,
                   help="Real train CSV — age distribution + min/max source.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=22000)
    p.add_argument("--num_inference_steps", type=int, default=1000)
    p.add_argument("--stochastic_scale", type=float, default=0.0)
    p.add_argument("--age_sample_method", choices=["stratified", "uniform"], default="stratified")
    p.add_argument("--modality", default="T1",
                   help="modality token for the generated volumes (token-set conditioning).")
    p.add_argument("--cohort", default=None,
                   help="optional cohort token (A-variant model); omitted → absent.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true",
                   help="Skip model load/inference, only validate paths + CSV emit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    vol_dir = out_dir / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "synth_age_index.csv"

    with open(args.dataset_config_path) as f:
        ds_cfg = json.load(f)
    with open(args.model_config_path) as f:
        model_cfg = json.load(f)
    with open(args.inference_config_path) as f:
        inf_cfg = json.load(f)

    import pandas as pd
    real_df = pd.read_csv(args.real_csv)
    age_min, age_max = float(real_df["age"].min()), float(real_df["age"].max())

    ages = sample_age_distribution(args.real_csv, args.num_samples, args.seed,
                                   method=args.age_sample_method)

    # Write index CSV up front so dry runs verify schema even without model.
    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eid", "rel_path", "age"])
        for i, age in enumerate(ages):
            eid = f"synth_{i:06d}"
            w.writerow([eid, f"volumes/{eid}.nii.gz", float(age)])
    print(f"[index] {index_path} ({len(ages)} rows)")

    if args.dry_run:
        print("[dry_run] skipping model load + inference.")
        return

    # Compose args namespace expected by define_instance (matches compute_metric).
    cfg_args = argparse.Namespace(**{**vars(args), **ds_cfg, **model_cfg, **inf_cfg})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = define_instance(cfg_args, "autoencoder_def").to(device)
    ae_ckpt = torch.load(os.path.join(args.pretrained_vae_path, "model.pt"),
                         map_location=device, weights_only=False)
    autoencoder.load_state_dict(ae_ckpt["autoencoder"])
    autoencoder.eval()

    unet = define_instance(cfg_args, "diffusion_unet_def").to(device)
    unet_ckpt = torch.load(os.path.join(args.pretrained_unet_path, "model.pt"),
                           map_location=device, weights_only=False)
    unet.load_state_dict(unet_ckpt.get("unet", unet_ckpt.get("unet_state_dict")), strict=True)
    unet.eval()

    noise_scheduler = define_instance(cfg_args, "noise_scheduler")
    latent_shape = ds_cfg["latent_shape"]
    noise_scheduler.set_timesteps(num_inference_steps=args.num_inference_steps,
                                  input_img_size_numel=int(np.prod(latent_shape)))

    stats = pd.read_csv(args.latent_stats_csv)
    global_mean = float(stats.iloc[0]["global_mean"])
    scale_factor = float(stats.iloc[0]["scale_factor"])

    latent_channels = cfg_args.latent_channels
    inference_spacing = ds_cfg.get("inference_spacing", [1.0, 1.0, 1.0])
    spacing_tensor = torch.tensor([[s * 1e2 for s in inference_spacing]],
                                  dtype=torch.float16, device=device)

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:],
                                    torch.tensor([0], dtype=all_timesteps.dtype)))

    cond_cfg = model_cfg.get("conditioning")
    use_token_set = bool(cond_cfg) and bool(cond_cfg.get("enabled", False))
    print(f"[cond] use_token_set={use_token_set} modality={args.modality} cohort={args.cohort}")

    for i, age in enumerate(ages):
        eid = f"synth_{i:06d}"
        out_path = vol_dir / f"{eid}.nii.gz"
        if out_path.exists():
            continue
        torch.manual_seed(args.seed + i)
        norm_age = _normalize_age(float(age), age_min, age_max)
        # condition on (modality, age[, cohort]); sex/dx/cdrsb left absent so the
        # model fills them (it was trained with per-token CFG drop → robust to absence).
        cond_dict = {"modality": args.modality, "age": float(age)}
        if args.cohort:
            cond_dict["cohort"] = args.cohort
        meta_tensor = _build_meta(cond_dict, cond_cfg, use_token_set, device, norm_age)
        latent = torch.randn((1, latent_channels, *latent_shape), device=device)
        with torch.no_grad(), autocast(dtype=torch.float16, enabled=args.amp):
            for t, next_t in zip(all_timesteps, all_next_timesteps):
                model_out = unet(x=latent,
                                 timesteps=torch.tensor((t,), device=device),
                                 spacing_tensor=spacing_tensor,
                                 meta_tensor=meta_tensor)
                latent, _ = noise_scheduler.step(model_out, t, latent, next_t,
                                                 args.stochastic_scale)
            vol = autoencoder.decode_stage_2_outputs((latent / scale_factor) + global_mean)
        arr = vol.squeeze().float().cpu().numpy()
        nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), out_path)
        if (i + 1) % 50 == 0:
            print(f"[gen] {i + 1}/{len(ages)} (last age={age:.1f})", flush=True)


if __name__ == "__main__":
    main()
