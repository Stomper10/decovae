"""Unified VAE embedding extraction + latent analysis pipeline.

Stages:
  - extract  : run VAE encoder on UKB MRI volumes, save mu/sigma/sampled-z
               (multi-GPU via torchrun)
  - geometry : compute latent geometry metrics over saved embeddings
               (single process; spawns one I/O thread per visible GPU)
  - stat     : compute global latent statistics over saved embeddings
               (single process; multiprocessing.Pool over CPU workers)

All outputs land under --work_dir:
  {work_dir}/embeddings/{eid}_emb.nii.gz
  {work_dir}/embeddings/{eid}_mu.npy
  {work_dir}/embeddings/{eid}_sigma.npy
  {work_dir}/embeddings/{eid}_emb.nii.gz.json
  {work_dir}/analysis/latent_geometry.csv
  {work_dir}/analysis/latent_stats.csv
"""
from __future__ import annotations

import os
import json
import logging
import argparse
import warnings
import threading
import contextlib
import numpy as np
import pandas as pd
import nibabel as nib
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist

import monai
from monai.transforms import Compose

from scripts.config_utils import load_json
from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.utils import define_instance
from datasets import get_adapter


# ============================================================================
# Shared helpers
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Experiment dir; embeddings/ and analysis/ live here.")
    parser.add_argument("--stages", type=str, default="extract,geometry,stat",
                        help="Comma-separated subset of: extract, geometry, stat.")
    parser.add_argument("--dataset_config_path", type=str, required=True,
                        help="Path to dataset config (e.g. configs/ukb_20252/dataset.json).")
    # extract-only — defaults pulled from dataset_config_path when omitted.
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Source NIfTI dir. Defaults to dataset.data_dir if unset.")
    parser.add_argument("--trained_autoencoder_path", type=str, default=None,
                        help="VAE checkpoint .pt (required for extract).")
    parser.add_argument("--train_label_dir", type=str, default=None,
                        help="train.csv path. Defaults to dataset.train_label_dir if unset.")
    parser.add_argument("--valid_label_dir", type=str, default=None,
                        help="valid.csv path. Defaults to dataset.valid_label_dir if unset.")
    # Required for stage=extract; ignored by geometry/stat.
    parser.add_argument("--config_environment", type=str, default=None)
    parser.add_argument("--config_diff_train_inf", type=str, default=None)
    parser.add_argument("--config_model_fm", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    # analysis
    parser.add_argument("--max_geometry_samples", type=int, default=5000)
    parser.add_argument("--num_stat_workers", type=int, default=32)

    args = parser.parse_args()

    # Merge dataset config into args; CLI overrides win.
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
    dataset_dict = load_json(args.dataset_config_path)
    for k, v in dataset_dict.items():
        setattr(args, k, v)
    for k, v in cli_overrides.items():
        setattr(args, k, v)
    return args


def parse_stages(s: str) -> list[str]:
    valid = {"extract", "geometry", "stat"}
    out = [t.strip() for t in s.split(",") if t.strip()]
    bad = [t for t in out if t not in valid]
    if bad:
        raise ValueError(f"Unknown stages: {bad}. Valid: {sorted(valid)}")
    return out


def embeddings_dir(work_dir: str) -> str:
    return os.path.join(work_dir, "embeddings")


def analysis_dir(work_dir: str) -> str:
    return os.path.join(work_dir, "analysis")


def run_name_of(work_dir: str) -> str:
    return os.path.basename(os.path.normpath(work_dir))


# ============================================================================
# Stage 1: extract  (adapted from maisi_embedding_UNET.py)
# ============================================================================
def _intensity_transform(intensity_norm: dict):
    norm_type = intensity_norm.get("type", "none").lower()
    if norm_type == "percentile":
        return monai.transforms.ScaleIntensityRangePercentilesd(
            keys="image",
            lower=intensity_norm["lower"],
            upper=intensity_norm["upper"],
            b_min=intensity_norm["b_min"],
            b_max=intensity_norm["b_max"],
            clip=intensity_norm.get("clip", False),
        )
    if norm_type == "range":
        return monai.transforms.ScaleIntensityRanged(
            keys="image",
            a_min=intensity_norm["a_min"],
            a_max=intensity_norm["a_max"],
            b_min=intensity_norm["b_min"],
            b_max=intensity_norm["b_max"],
            clip=intensity_norm.get("clip", True),
        )
    return None  # "none" → no intensity normalization


def create_transforms(
    orientation_axcodes: str,
    intensity_norm: dict,
    dim: tuple | None = None,
    cached: bool = False,
) -> Compose:
    if cached:
        # Already-preprocessed cache .npy (single-channel, final 192^3, [0,1]):
        # load + add channel + fp32 only. No orient/resize/intensity (and NO
        # round-to-128 resize) — encode at the native cache resolution.
        return Compose([
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
        ])
    base = [
        monai.transforms.LoadImaged(keys="image"),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.Orientationd(keys="image", axcodes=orientation_axcodes),
    ]
    if dim is None:
        return Compose(base)

    intensity_t = _intensity_transform(intensity_norm)
    pipeline = base + [monai.transforms.EnsureTyped(keys="image", dtype=torch.float32)]
    if intensity_t is not None:
        pipeline.append(intensity_t)
    pipeline.append(monai.transforms.Resized(keys="image", spatial_size=dim, mode="trilinear"))
    return Compose(pipeline)


def round_number(number: int, base_number: int = 128) -> int:
    return int(max(round(float(number) / float(base_number)), 1.0) * float(base_number))


def load_filenames(data_list_path: str, key: str) -> list:
    with open(data_list_path, "r") as f:
        json_data = json.load(f)
    return [item["image"] for item in json_data[key]]


def process_file(filepath, ext_args, autoencoder, device, plain_transforms,
                 new_transforms, logger, adapter, cached=False):
    sid = adapter.extract_subject_id(filepath)
    out_filename = os.path.join(ext_args.embedding_base_dir, adapter.embedding_filename(sid))
    mu_filename = os.path.join(ext_args.embedding_base_dir, adapter.mu_filename(sid))
    sigma_filename = os.path.join(ext_args.embedding_base_dir, adapter.sigma_filename(sid))

    if (os.path.isfile(out_filename)
            and os.path.isfile(mu_filename)
            and os.path.isfile(sigma_filename)):
        return

    test_data = {"image": os.path.join(ext_args.data_base_dir, filepath)}
    new_data = new_transforms(test_data)
    nda_image = new_data["image"]
    new_affine = nda_image.meta["affine"].numpy()
    if cached:
        # cache .npy has no NIfTI header; it is final 192^3 @ 1mm.
        dim = list(nda_image.shape[1:4])
        spacing = [1.0, 1.0, 1.0]
    else:
        nda = plain_transforms(test_data)["image"]
        dim = [int(nda.meta["dim"][i]) for i in range(1, 4)]
        spacing = [float(nda.meta["pixdim"][i]) for i in range(1, 4)]
    logger.info(f"dim: {dim}, spacing: {spacing}")
    nda_image = nda_image.numpy().squeeze()
    logger.info(f"encode dim: {nda_image.shape}")

    try:
        Path(out_filename).parent.mkdir(parents=True, exist_ok=True)
        # D1: cache/pooled latents extracted in fp32 (no autocast) — these are
        # reused for all diffusion training + scale_factor. Raw datasets keep the
        # original fp16 autocast path (MIUA reproducibility).
        encode_ctx = contextlib.nullcontext() if cached else torch.amp.autocast("cuda")
        with encode_ctx:
            pt_nda = torch.from_numpy(nda_image).float().to(device).unsqueeze(0).unsqueeze(0)
            z_mu, z_sigma = autoencoder.encode(pt_nda)

            if not os.path.isfile(mu_filename):
                np.save(mu_filename, z_mu.squeeze().cpu().detach().numpy())
                logger.info(f"Saved mu: {mu_filename}")

            if not os.path.isfile(sigma_filename):
                np.save(sigma_filename, z_sigma.squeeze().cpu().detach().numpy())
                logger.info(f"Saved sigma: {sigma_filename}")

            if not os.path.isfile(out_filename):
                z = autoencoder.sampling(z_mu, z_sigma)
                out_nda = z.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
                nib.save(nib.Nifti1Image(np.float32(out_nda), affine=new_affine), out_filename)
                logger.info(f"Saved embedding: {out_filename}")
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")


@torch.inference_mode()
def _encode_all(ext_args, num_gpus, adapter):
    local_rank, world_size, device = initialize_distributed()
    logger = setup_logging("extract")
    logger.info(f"Using device {device} (rank {local_rank}/{world_size})")

    autoencoder = define_instance(ext_args, "autoencoder_def").to(device)
    try:
        ckpt = torch.load(ext_args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(ckpt["autoencoder"], strict=True)
    except Exception as e:
        logger.error(f"Failed to load autoencoder: {e}")
        raise

    Path(ext_args.embedding_base_dir).mkdir(parents=True, exist_ok=True)
    cached = bool(getattr(ext_args, "cached_input", False))
    plain_transforms = create_transforms(
        orientation_axcodes=ext_args.orientation_axcodes,
        intensity_norm=ext_args.intensity_norm,
        dim=None,
        cached=cached,
    )
    # For cached input there is no round-to-128 resize — encode at native 192^3.
    cached_transforms = plain_transforms if cached else None

    for split, image_list, list_path, key in [
        ("training", getattr(ext_args, "train_image_list", None), ext_args.json_data_list, "training"),
        ("validation", getattr(ext_args, "val_image_list", None), ext_args.val_json_data_list, "validation"),
    ]:
        # prefer the in-memory, per-rank-rebuilt list (race-free); fall back to the
        # on-disk json only if a caller didn't populate it.
        filenames = image_list if image_list is not None else load_filenames(list_path, key)
        logger.info(f"{split}: {len(filenames)} files (cached={cached})")
        for i, filepath in enumerate(filenames):
            if i % world_size != local_rank:
                continue
            if cached:
                new_transforms = cached_transforms
            else:
                new_dim = tuple(
                    round_number(
                        int(plain_transforms(
                            {"image": os.path.join(ext_args.data_base_dir, filepath)}
                        )["image"].meta["dim"][j])
                    )
                    for j in range(1, 4)
                )
                new_transforms = create_transforms(
                    orientation_axcodes=ext_args.orientation_axcodes,
                    intensity_norm=ext_args.intensity_norm,
                    dim=new_dim,
                )
            process_file(filepath, ext_args, autoencoder, device,
                         plain_transforms, new_transforms, logger, adapter, cached=cached)


def _list_gz_files(folder_path: str) -> list[str]:
    out = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".gz"):
                out.append(os.path.join(root, f))
    return out


def _create_json_files(gz_files: list[str], df: pd.DataFrame, adapter):
    for gz in gz_files:
        img = nib.load(gz)
        dimensions = list(img.shape[:3])
        spacing = [float(s) for s in img.header.get_zooms()[:3]]
        sid = adapter.extract_subject_id(gz)
        # Match against the id column whatever its dtype: try int first,
        # fall back to string equality so adapters can use either.
        try:
            mask = df[adapter.id_column] == int(sid)
        except (TypeError, ValueError):
            mask = df[adapter.id_column].astype(str) == str(sid)
        row = df.loc[mask]
        if len(row) == 0:
            continue
        data = {
            "dim": dimensions,
            "spacing": spacing,
            "top_region_index": [0, 0, 0, 0],
            "bottom_region_index": [0, 0, 0, 0],
            "cond": adapter.derive_conditions(row.iloc[0]),
        }
        with open(gz + ".json", "w") as f:
            json.dump(data, f, indent=4)


def _norm_label_df(csv_path: str, adapter) -> pd.DataFrame:
    return adapter.normalize_label_df(pd.read_csv(csv_path))


def run_extract(args: argparse.Namespace) -> int:
    """Returns local_rank (0 if non-distributed)."""
    for k in ("data_dir", "trained_autoencoder_path",
              "train_label_dir", "valid_label_dir",
              "config_environment", "config_diff_train_inf", "config_model_fm"):
        if getattr(args, k) is None:
            raise ValueError(f"--{k} required for stage 'extract'")

    # Pooled diffusion targets exclude vae_only rows (BraTS-T1c): the VAE saw
    # them, but the diffusion modality vocab is {T1,T2,FLAIR}, so no T1c latents.
    adapter_kwargs = {"include_vae_only": False} if args.dataset_adapter == "pooled" else {}
    adapter = get_adapter(args.dataset_adapter, **adapter_kwargs)

    out_dir = embeddings_dir(args.work_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    # Build datalists at work_dir (rank 0 only writes; all read).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_list = os.path.join(args.work_dir, "train_files.json")
    valid_list = os.path.join(args.work_dir, "valid_files.json")

    # Build split file lists from the STATIC manifest CSV. Every rank rebuilds them
    # independently (deterministic, same order) so _encode_all never cross-reads a
    # rank-0-written json — that read raced under torchrun on a networked FS:
    #   (a) fresh dir   -> FileNotFoundError (existence not yet visible on other node)
    #   (b) prior file  -> STALE content read from NFS attr cache (silent: extracts
    #                       on an old manifest's slices)
    #   (c) mid-write   -> truncated -> JSONDecodeError
    # The init_process_group rendezvous already orders rank-0-write before the reads
    # at the PROCESS level, so a dist.barrier() doesn't help (the gap is FS metadata
    # visibility, not process ordering). Rebuilding from the static CSV sidesteps it.
    # rank 0 still writes the json for downstream consumers (train_UNET).
    train_images = [x["image"] for x in adapter.load_manifest(args.train_label_dir, args.data_dir)]
    valid_images = [x["image"] for x in adapter.load_manifest(args.valid_label_dir, args.data_dir)]
    if local_rank == 0:
        with open(train_list, "w") as f:
            json.dump({"training": [{"image": x} for x in train_images]}, f)
        with open(valid_list, "w") as f:
            json.dump({"validation": [{"image": x} for x in valid_images]}, f)

    # Build configs from JSON; override paths to absolute work_dir-rooted ones.
    ext_args = load_config(args.config_environment, args.config_diff_train_inf,
                           args.config_model_fm)
    ext_args.embedding_base_dir = out_dir
    ext_args.json_data_list = train_list
    ext_args.val_json_data_list = valid_list
    ext_args.data_base_dir = args.data_dir
    ext_args.trained_autoencoder_path = args.trained_autoencoder_path
    # MAISI's `norm_float16` hard-casts GroupNorm output to fp16, crashing the
    # next conv under fp32 (no autocast) — same bug compute_metric.sh patches in
    # its merged-config step. extract uses fp32 for cached pooled .npy inputs.
    if isinstance(getattr(ext_args, "autoencoder_def", None), dict):
        ext_args.autoencoder_def["norm_float16"] = False
    if isinstance(getattr(ext_args, "mask_generation_autoencoder_def", None), dict):
        ext_args.mask_generation_autoencoder_def["norm_float16"] = False
    # Thread dataset preprocessing into ext_args so create_transforms can read them.
    ext_args.orientation_axcodes = args.orientation_axcodes
    ext_args.intensity_norm = args.intensity_norm
    ext_args.cached_input = bool(getattr(args, "cached_input", False))
    # In-memory split lists (rebuilt per-rank above) — _encode_all consumes these
    # instead of re-reading the raced json off disk.
    ext_args.train_image_list = train_images
    ext_args.val_image_list = valid_images

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    _encode_all(ext_args, args.num_gpus, adapter)

    if dist.is_initialized():
        dist.barrier()
        local_rank = dist.get_rank()

    # Post-processing on rank 0 only: per-embedding metadata JSON.
    if local_rank == 0:
        df_train = _norm_label_df(args.train_label_dir, adapter)
        df_valid = _norm_label_df(args.valid_label_dir, adapter)
        df_all = pd.concat([df_train, df_valid], ignore_index=True)
        gz_files = _list_gz_files(out_dir)
        print(f"[extract] writing {len(gz_files)} metadata json files")
        _create_json_files(gz_files, df_all, adapter)

    return local_rank


# ============================================================================
# Stage 2: geometry  (adapted from analyze_latent_geometry_mri.py)
# ============================================================================
def _load_chunk(rank, paths_chunk, results_list, idx):
    device = torch.device(f"cuda:{rank}")
    chunk_latents = []
    for path in tqdm(paths_chunk, desc=f"GPU {rank} Loading",
                     position=rank + 1, leave=False):
        try:
            tensor = torch.from_numpy(nib.load(path).get_fdata()).float()
            if tensor.shape[0] in (4, 8, 16):
                tensor = tensor.permute(1, 2, 3, 0)
            C = tensor.shape[-1]
            chunk_latents.append(tensor.reshape(-1, C).to(device))
        except Exception:
            continue
    if chunk_latents:
        results_list[idx] = torch.cat(chunk_latents, dim=0)


def run_geometry(args: argparse.Namespace):
    if args.train_label_dir is None:
        raise ValueError("--train_label_dir required for stage 'geometry'")

    emb_dir = embeddings_dir(args.work_dir)
    out_dir = analysis_dir(args.work_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_name = run_name_of(args.work_dir)

    num_gpus = min(4, torch.cuda.device_count())
    if num_gpus == 0:
        raise RuntimeError("Geometry analysis requires at least one CUDA GPU.")
    print(f"[geometry] using {num_gpus} GPUs")

    adapter = get_adapter(args.dataset_adapter)
    df = _norm_label_df(args.train_label_dir, adapter)
    target_ids = df[adapter.id_column].astype(str).tolist()

    train_emb_paths = []
    for sid in target_ids:
        p = os.path.join(emb_dir, adapter.embedding_filename(sid))
        if os.path.exists(p):
            train_emb_paths.append(p)
        if len(train_emb_paths) >= args.max_geometry_samples:
            break
    print(f"[geometry] found {len(train_emb_paths)} embeddings (cap={args.max_geometry_samples})")

    chunk_size = (len(train_emb_paths) + num_gpus - 1) // num_gpus
    chunks = [train_emb_paths[i * chunk_size:(i + 1) * chunk_size]
              for i in range(num_gpus)]
    results = [None] * num_gpus
    threads = []
    for i in range(num_gpus):
        if chunks[i]:
            t = threading.Thread(target=_load_chunk, args=(i, chunks[i], results, i))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    print("\n[geometry] merging chunks to cuda:0...")
    valid = [r.to("cuda:0") for r in results if r is not None]
    raw_flat = torch.cat(valid, dim=0)
    print(f"[geometry] total voxel matrix shape: {raw_flat.shape}")

    global_mean = raw_flat.mean()
    global_std = raw_flat.std()
    scale_factor = 1.0 / global_std.clamp(min=1e-8)
    diff_flat = (raw_flat - global_mean) * scale_factor
    print(f"  global_mean={global_mean.item():.4f} "
          f"global_std={global_std.item():.4f} "
          f"scale_factor={scale_factor.item():.4f}")

    device = torch.device("cuda:0")
    C = raw_flat.shape[1]
    off_diag = ~torch.eye(C, dtype=torch.bool, device=device)

    # Per-channel mean for proper Pearson correlation. Distinct from the
    # scalar global_mean above (used by UNet downstream); the scalar mean
    # cannot decorrelate channels.
    channel_mean = raw_flat.mean(dim=0, keepdim=True)

    # [1] raw_cos_sim — channel cosine similarity (no centering).
    # Use the X^T X / sqrt(diag) outer-product trick instead of F.normalize
    # to avoid allocating a (N, C)-shaped normed copy of raw_flat (was OOM
    # at N=327M).
    xtx = torch.mm(raw_flat.t(), raw_flat)
    d = xtx.diag().clamp(min=1e-8).sqrt()
    raw_corr = xtx / (d[:, None] * d[None, :])
    raw_cos_sim = raw_corr[off_diag].abs().mean().item()
    print(f"[1] raw_cos_sim={raw_cos_sim:.4f}  (no centering)")
    del xtx, d, raw_corr

    # [2] diff_cos_sim — Pearson R off-diagonal abs mean. Center raw_flat
    # in-place to avoid a second 5GB tensor; downstream uses diff_flat
    # (constructed earlier, independent of raw_flat).
    raw_flat.sub_(channel_mean)
    xtx_c = torch.mm(raw_flat.t(), raw_flat)
    d = xtx_c.diag().clamp(min=1e-8).sqrt()
    pearson_R = xtx_c / (d[:, None] * d[None, :])
    diff_cos_sim = pearson_R[off_diag].abs().mean().item()
    print(f"[2] diff_cos_sim={diff_cos_sim:.4f}  (Pearson R off-diagonal)")
    del raw_flat, xtx_c, d, pearson_R
    torch.cuda.empty_cache()

    N = diff_flat.shape[0]
    sample_size = min(N, 1_000_000)
    indices = torch.randint(0, N, (sample_size,), device=device)
    diff_sample = diff_flat[indices]

    l2_norms = torch.norm(diff_sample, p=2, dim=1)
    cov = torch.cov(diff_sample.t())
    eigvals = torch.linalg.eigvalsh(cov).real
    evr = torch.sort(eigvals, descending=True).values / eigvals.sum()
    entropy = -torch.sum(evr * torch.log(evr + 1e-8))
    effective_rank = torch.exp(entropy).item()
    mean_l2 = l2_norms.mean().item()
    print(f"[3] mean_l2_norm={mean_l2:.4f} (ideal ~{C ** 0.5:.2f}); "
          f"effective_rank={effective_rank:.2f} / {C}")

    pair_samples = 50_000
    idx_A = torch.randint(0, N, (pair_samples,), device=device)
    idx_B = torch.randint(0, N, (pair_samples,), device=device)
    vec_A = F.normalize(diff_flat[idx_A], p=2, dim=1)
    vec_B = F.normalize(diff_flat[idx_B], p=2, dim=1)
    avg_pair_cos_sim = (vec_A * vec_B).sum(dim=1).abs().mean().item()
    print(f"[4] avg_pairwise_cos_sim={avg_pair_cos_sim:.4f}")

    row = {
        "run_name": run_name,
        "num_samples": len(train_emb_paths),
        "channels": C,
        "global_mean": global_mean.item(),
        "global_std": global_std.item(),
        "scale_factor": scale_factor.item(),
        "raw_cos_sim": raw_cos_sim,
        "diff_cos_sim": diff_cos_sim,
        "mean_l2_norm": mean_l2,
        "effective_rank": effective_rank,
        "avg_pairwise_cos_sim": avg_pair_cos_sim,
    }
    out_csv = os.path.join(out_dir, "latent_geometry.csv")
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"[geometry] wrote {out_csv}")


# ============================================================================
# Stage 3: stat  (adapted from analyze_latent_stat.py)
# ============================================================================
def _stat_chunk(file_info_list):
    voxel_count = 0
    sum_z = 0.0
    sum_sq_z = 0.0
    sum_sigma = 0.0
    processed = 0
    for emb_path, sigma_path in file_info_list:
        if not os.path.exists(emb_path):
            continue
        try:
            z = nib.load(emb_path).get_fdata().astype(np.float64)
            cur_sigma_sum = 0.0
            if sigma_path and os.path.exists(sigma_path):
                cur_sigma_sum = float(np.sum(np.load(sigma_path).astype(np.float64)))
            voxel_count += z.size
            sum_z += float(np.sum(z))
            sum_sq_z += float(np.sum(z ** 2))
            sum_sigma += cur_sigma_sum
            processed += 1
        except Exception:
            continue
    return voxel_count, sum_z, sum_sq_z, sum_sigma, processed


def run_stat(args: argparse.Namespace):
    if args.train_label_dir is None:
        raise ValueError("--train_label_dir required for stage 'stat'")

    emb_dir = embeddings_dir(args.work_dir)
    out_dir = analysis_dir(args.work_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_name = run_name_of(args.work_dir)

    adapter = get_adapter(args.dataset_adapter)
    df = _norm_label_df(args.train_label_dir, adapter)
    target_ids = df[adapter.id_column].astype(str).tolist()
    print(f"[stat] {len(target_ids)} target ids; scanning {emb_dir} "
          f"with {args.num_stat_workers} workers")

    file_info = [
        (os.path.join(emb_dir, adapter.embedding_filename(sid)),
         os.path.join(emb_dir, adapter.sigma_filename(sid)))
        for sid in target_ids
    ]
    chunk = len(file_info) // args.num_stat_workers + 1
    chunks = [file_info[i:i + chunk] for i in range(0, len(file_info), chunk)]

    total_v = sum_z = sum_sq = sum_s = processed = 0
    with mp.Pool(processes=args.num_stat_workers) as pool:
        for v, sz, ssq, ss, p in tqdm(pool.imap(_stat_chunk, chunks),
                                       total=len(chunks), desc=f"stat[{run_name}]"):
            total_v += v
            sum_z += sz
            sum_sq += ssq
            sum_s += ss
            processed += p

    if total_v == 0:
        print("[stat] no embeddings found; skipping CSV write")
        return

    global_mean = sum_z / total_v
    global_var = max((sum_sq / total_v) - global_mean ** 2, 0.0)
    global_std = float(np.sqrt(global_var))
    scaling_factor = 1.0 / (global_std + 1e-8)
    avg_single_sigma = sum_s / total_v

    row = {
        "run_name": run_name,
        "scaling_factor": scaling_factor,
        "global_mean": global_mean,
        "global_std": global_std,
        "avg_single_sigma": avg_single_sigma,
        "num_train_samples": processed,
    }
    out_csv = os.path.join(out_dir, "latent_stats.csv")
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"[stat] {row}")
    print(f"[stat] wrote {out_csv}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    stages = parse_stages(args.stages)
    print(f"[extract_emb] work_dir={args.work_dir} stages={stages}")

    local_rank = 0
    if "extract" in stages:
        local_rank = run_extract(args)
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        if "geometry" in stages:
            run_geometry(args)
        if "stat" in stages:
            run_stat(args)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
