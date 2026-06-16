"""Unified UNET (latent diffusion) training script.

Mirrors the structure of train_VAE.py: every run lives under
<output_dir>/<run_name>/ with a fixed sub-tree (analysis, embeddings, logs,
outputs, weights/unet). global_mean / scaling_factor are read from
<exp>/analysis/latent_stats.csv (produced upstream by extract_emb.py) — they
are no longer expected in JSON configs.
"""

from __future__ import annotations

import os
import sys
import json
import yaml
import wandb
import signal
import argparse
import warnings
import pandas as pd
from tqdm import trange

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

import monai
from monai.transforms import Compose
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, DistributedSampler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.schedulers.rectified_flow import RFlowScheduler

from scripts.config_utils import load_json
from scripts.utils import define_instance, count_parameters
from datasets import get_adapter
import patches  # noqa: F401  # registers DiffusionModelUNetMaisiV2 / RFlowSchedulerV2 for ConfigParser

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# DDP / IO utilities (train_VAE.py와 동일)
# ---------------------------------------------------------------------------
def setup_ddp():
    # Launch via torchrun. LOCAL_RANK / RANK / WORLD_SIZE are set by torchrun;
    # we fall back to single-process defaults if absent, but dist.init_process_group
    # will then require a rendezvous backend to be available.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    try:
        dev_cnt = torch.cuda.device_count()
    except Exception as e:
        dev_cnt = f"<err: {e}>"
    import socket
    print(f"[ddp-init] host={socket.gethostname()} rank={rank} local_rank={local_rank} "
          f"CUDA_VISIBLE_DEVICES={cvd} torch.cuda.device_count()={dev_cnt}", flush=True)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def cleanup_ddp():
    dist.destroy_process_group()


def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


SHUTDOWN_REQUESTED = False
def graceful_shutdown(signum, frame):
    global SHUTDOWN_REQUESTED
    print(f"\n[!] Received signal {signum}. Requesting graceful shutdown...", flush=True)
    SHUTDOWN_REQUESTED = True
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


def get_run_name(manual_name=None, default_prefix="manual"):
    job_id = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME")
    if manual_name:
        return manual_name
    elif job_id and job_name:
        return f"{job_name}_{job_id}"
    elif job_id:
        return f"slurm_{job_id}"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{default_prefix}_{timestamp}"


# ---------------------------------------------------------------------------
# Config / experiment-dir helpers
# ---------------------------------------------------------------------------
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config_path", type=str, required=True,
                        help="Path to the dataset-level config (e.g. configs/ukb_20252/dataset.json).")
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--train_config_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cpus_per_task", type=int, default=8,
                        help="Number of CPUs allocated per task by Slurm.")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Stage base directory. Required (passed by launcher).")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity. Overrides dataset.wandb_entity.")
    parser.add_argument("--train_label_dir", type=str, default=None,
                        help="Pooled manifest CSV (only used for pooled imbalance sampling).")
    args = parser.parse_args()

    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run_name to be specified.")
    args.run_name = get_run_name(args.run_name)

    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}

    dataset_dict = load_json(args.dataset_config_path)
    for k, v in dataset_dict.items():
        setattr(args, k, v)

    model_config_dict = load_json(args.model_config_path)
    for k, v in model_config_dict.items():
        setattr(args, k, v)
    train_config_dict = load_json(args.train_config_path)
    for k, v in train_config_dict["diffusion_unet_inference"].items():
        setattr(args, k, v)
    for k, v in train_config_dict["diffusion_unet_train"].items():
        setattr(args, k, v)

    for k, v in cli_overrides.items():
        setattr(args, k, v)
    return args


def setup_experiment_dirs(base_dir: str, run_name: str):
    """실험 디렉토리 트리 생성 후 경로 dict 반환.

    구조:
        <base>/<run_name>/
            ├── analysis/        # latent_stats.csv (extract_emb.py 산출)
            ├── embeddings/
            ├── logs/
            ├── outputs/
            └── weights/
                └── unet/        # UNET 체크포인트 + config.json
    """
    exp_dir = os.path.join(base_dir, run_name)
    paths = {
        "exp": exp_dir,
        "analysis": os.path.join(exp_dir, "analysis"),
        "embeddings": os.path.join(exp_dir, "embeddings"),
        "logs": os.path.join(exp_dir, "logs"),
        "outputs": os.path.join(exp_dir, "outputs"),
        "weights": os.path.join(exp_dir, "weights"),
        "unet": os.path.join(exp_dir, "weights", "unet"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def save_used_config(args, unet_dir: str):
    """이번 실험에 실제로 사용된 최종 config (CSV에서 읽은 stats 포함)를 weights/unet/에 스냅샷."""
    snap_path = os.path.join(unet_dir, "config.json")
    serializable = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = repr(v)
    with open(snap_path, "w") as f:
        json.dump(serializable, f, indent=2, sort_keys=False)
    print(f"[Config] Saved used config snapshot to {snap_path}", flush=True)


def load_latent_stats(analysis_dir: str):
    """`<exp>/analysis/latent_stats.csv`에서 (global_mean, scaling_factor)를 읽어온다.

    extract_emb.py가 단일 row로 기록하므로 첫 행만 사용한다.
    내부 변수명은 scale_factor지만 CSV 컬럼명은 scaling_factor로 통일되어 있다.
    """
    csv_path = os.path.join(analysis_dir, "latent_stats.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"latent_stats.csv not found at {csv_path}. "
            "Run extract_emb.py with --stages stat first."
        )
    df = pd.read_csv(csv_path)
    return float(df["global_mean"].iloc[0]), float(df["scaling_factor"].iloc[0])


def resume_from_latest(unet, optimizer, lr_scheduler, scaler, output_dir, device):
    if not os.path.exists(output_dir):
        print("[Resume] No checkpoint directory found. Starting fresh.")
        return 0, float("inf"), 0

    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
        and os.path.isdir(os.path.join(output_dir, d))
        and os.path.exists(os.path.join(output_dir, d, "model.pt"))
    ]
    if len(ckpts) == 0:
        print("[Resume] No checkpoint found in directory. Starting fresh.")
        return 0, float("inf"), 0

    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
    latest_ckpt_dir = ckpts[-1]
    print(f"[Resume] Resuming from checkpoint: {latest_ckpt_dir}")

    path = os.path.join(output_dir, latest_ckpt_dir, "model.pt")
    ckpt = torch.load(path, map_location=device)

    # "unet_state_dict" / "lr_scheduler" are legacy key names; new checkpoints use
    # "unet" / "scheduler" to match train_VAE.py and train_DFT.py.
    unet.load_state_dict(ckpt.get("unet", ckpt.get("unet_state_dict")))
    optimizer.load_state_dict(ckpt["optimizer"])
    sched_state = ckpt.get("scheduler", ckpt.get("lr_scheduler"))
    if sched_state is not None: lr_scheduler.load_state_dict(sched_state)
    if "scaler" in ckpt and scaler is not None: scaler.load_state_dict(ckpt["scaler"])

    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf")), ckpt.get("epoch", 0)


def reduce_mean_scalar(x):
    t = x.detach().reshape(1).to(torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()


# ---------------------------------------------------------------------------
# Data helpers (UNET-specific: latent embedding inputs)
# ---------------------------------------------------------------------------
def load_filenames(data_list_path: str, data_key: str, adapter) -> list:
    with open(data_list_path, "r") as f:
        json_data = json.load(f)
    filenames = json_data[data_key]
    # adapter.extract_subject_id + adapter.embedding_filename — extract_emb.py와 동일 컨벤션.
    return [adapter.embedding_filename(adapter.extract_subject_id(item["image"]))
            for item in filenames]


class TokenSetCondLoader:
    """Reads the per-volume typed-token JSON ``cond`` dict and emits fixed-slot
    tensors: ``cond_cat`` (n_cat long), ``cond_cont`` (n_cont float, [0,1]-normed),
    ``cond_presence`` (n_attr bool, cats-then-conts). Absent value → presence
    False (NOT a fake 0). Order matches TokenSetEncoder."""

    def __init__(self, attributes: list[dict]):
        self.attributes = attributes

    def __call__(self, data):
        from patches.token_set_encoder import encode_token_set
        d = dict(data)
        with open(d["cond"]) as f:
            cond = json.load(f)["cond"]
        cat_idx, cont_val, pres = encode_token_set(cond, self.attributes)
        d["cond_cat"] = torch.tensor(cat_idx, dtype=torch.long)
        d["cond_cont"] = torch.tensor(cont_val, dtype=torch.float32)
        d["cond_presence"] = torch.tensor(pres, dtype=torch.bool)
        return d


def cfg_drop_presence(presence, full_p, token_p, keep_idx=(0,)):
    """Classifier-free-guidance dropout on the presence mask (training only).
    Whole-set drop (→ null) with prob full_p; independent per-token drop with
    prob token_p — BOTH keep the keep_idx slots (modality at index 0), so the
    "unconditional" baseline is modality-only (decision A: modality is always
    conditioned). Operates only on already-present tokens — never fabricates absent."""
    pres = presence.clone()
    B, n = pres.shape
    if token_p > 0:
        drop = torch.rand(B, n, device=pres.device) < token_p
        for k in keep_idx:
            drop[:, k] = False
        pres = pres & ~drop
    if full_p > 0:
        whole = torch.rand(B, device=pres.device) < full_p
        pres[whole] = False
        for k in keep_idx:                       # keep modality even on whole-set drop (A)
            pres[whole, k] = presence[whole, k]
    return pres


def prepare_transform(include_body_region: bool = False, cond_attributes=None, include_spacing: bool = True):
    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    data_transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
    ]
    # spacing is constant (1mm-isotropic) across the pooled corpus → uninformative;
    # gated by the UNet's include_spacing_input (off for pooled). Mirrors include_body_region.
    if include_spacing:
        data_transforms_list += [
            monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
        ]
    if cond_attributes is not None:
        # typed token-set: dict cond → fixed-slot cat/cont/presence tensors
        data_transforms_list.append(TokenSetCondLoader(cond_attributes))
    else:
        # legacy: scalar age conditioning (cond[0]) — MIUA / per-cohort path
        data_transforms_list += [
            monai.transforms.Lambdad(keys="cond", func=lambda x: _load_data_from_file(x, "cond")),
            monai.transforms.Lambdad(keys="cond", func=lambda x: torch.tensor([x[0]], dtype=torch.float32)),
        ]
    if include_body_region:
        data_transforms_list += [
            monai.transforms.Lambdad(keys="top_region_index",
                                     func=lambda x: _load_data_from_file(x, "top_region_index")),
            monai.transforms.Lambdad(keys="bottom_region_index",
                                     func=lambda x: _load_data_from_file(x, "bottom_region_index")),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        ]
    return Compose(data_transforms_list)


def build_file_list(filenames, embedding_base_dir, include_body_region, include_spacing=True):
    files = []
    for fname in filenames:
        str_img = os.path.join(embedding_base_dir, fname)
        if not os.path.exists(str_img):
            continue
        str_info = str_img + ".json"
        item = {"image": str_img, "cond": str_info}
        if include_spacing:
            item["spacing"] = str_info
        if include_body_region:
            item["top_region_index"] = str_info
            item["bottom_region_index"] = str_info
        files.append(item)
    return files


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------
def main():
    setup_ddp()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank == 0:
        print("=" * 50)
        print(f"✅ Training started with a total of {world_size} GPUs across all nodes.")
        print(f"   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
              f"torch.cuda.device_count()={torch.cuda.device_count()}")
        print("=" * 50)
        print_config()

    args = load_config()
    adapter = get_adapter(args.dataset_adapter)
    device = torch.device(f"cuda:{local_rank}")

    # bf16: autocast on, no GradScaler (only fp16 needs loss scaling).
    wd = getattr(args, "weight_dtype", "fp16")
    amp_dtype = torch.bfloat16 if wd == "bf16" else torch.float16
    use_scaler = args.amp and amp_dtype == torch.float16
    # Typed token-set conditioning (pooled). enabled=false / absent → legacy
    # (1-D meta) or unconditional path — preserves MIUA reproducibility.
    cond_cfg = getattr(args, "conditioning", None)
    use_token_set = bool(cond_cfg) and bool(cond_cfg.get("enabled", False))
    cfg_full_p = float(cond_cfg.get("cfg_drop_prob", 0.0)) if use_token_set else 0.0
    cfg_token_p = float(cond_cfg.get("per_token_drop_prob", 0.0)) if use_token_set else 0.0

    if rank == 0:
        print(f"[Opt] Using gradient accumulation with {args.gradient_accumulation_steps} steps.")
        print(f"[Cond] use_token_set={use_token_set} weight_dtype={wd} "
              f"(cfg_full={cfg_full_p}, cfg_token={cfg_token_p})")

    set_determinism(seed=args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Speed > bit-exact reproducibility. See train_VAE.py for the full note.
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    # 모든 rank가 같은 트리를 알고 있어야 resume / 데이터 로드가 안전.
    exp_paths = setup_experiment_dirs(args.output_dir, args.run_name)
    weights_unet_dir = exp_paths["unet"]
    embedding_base_dir = exp_paths["embeddings"]
    train_json = os.path.join(exp_paths["exp"], "train_files.json")
    valid_json = os.path.join(exp_paths["exp"], "valid_files.json")

    # ------------------------------------------------------------------
    # Latent stats (rank 0이 CSV 읽고 broadcast)
    # ------------------------------------------------------------------
    global_mean = torch.zeros(1, device=device)
    scale_factor = torch.zeros(1, device=device)
    if rank == 0:
        gmean_host, sfac_host = load_latent_stats(exp_paths["analysis"])
        global_mean[0] = gmean_host
        scale_factor[0] = sfac_host
    dist.broadcast(global_mean, src=0)
    dist.broadcast(scale_factor, src=0)
    if rank == 0:
        print(f"[Stats] global_mean = {global_mean.item():.6f}", flush=True)
        print(f"[Stats] scale_factor = {scale_factor.item():.6f}", flush=True)

    # 실제 사용된 값을 args에 박아 두고 config snapshot 저장.
    if rank == 0:
        args.global_mean = float(global_mean.item())
        args.scale_factor = float(scale_factor.item())
        print("[Config] Loaded hyperparameters:")
        print(yaml.dump(vars(args), sort_keys=False))
        save_used_config(args, weights_unet_dir)
        print(f"[Paths] exp_dir    = {exp_paths['exp']}")
        print(f"[Paths] logs_dir   = {exp_paths['logs']}")
        print(f"[Paths] unet_ckpts = {weights_unet_dir}")
        if args.report_to:
            wandb.init(project=args.wandb_project_unet, entity=args.wandb_entity,
                       config=args, name=args.run_name)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    unet = define_instance(args, "diffusion_unet_def").to(device)
    noise_scheduler = define_instance(args, "noise_scheduler")
    include_body_region = unet.include_top_region_index_input
    include_spacing = unet.include_spacing_input
    include_modality = unet.num_class_embeds is not None
    num_train_timesteps = args.noise_scheduler["num_train_timesteps"]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    filenames_train = load_filenames(train_json, "training", adapter)
    train_files = build_file_list(filenames_train, embedding_base_dir, include_body_region, include_spacing)

    filenames_valid = load_filenames(valid_json, "validation", adapter)[:args.num_valid]
    valid_files = build_file_list(filenames_valid, embedding_base_dir, include_body_region, include_spacing)

    if rank == 0:
        print(f"Total number of training data is {len(train_files)}.")
        print(f"Total number of validation data is {len(valid_files)}.")

    data_transform = prepare_transform(
        include_body_region=include_body_region,
        cond_attributes=cond_cfg["attributes"] if use_token_set else None,
        include_spacing=include_spacing,
    )

    workers_per_gpu = args.cpus_per_task // world_size
    train_dataset = CacheDataset(data=train_files, transform=data_transform,
                                 cache_rate=args.cache_rate, num_workers=workers_per_gpu)
    if (getattr(args, "imbalance_sampling", False) and args.dataset_adapter == "pooled"
            and getattr(args, "train_label_dir", None)):
        from datasets.sampling import DistributedWeightedSampler, temperature_weights
        tau = float(getattr(args, "sampling_tau", 0.5))
        sid2cell = adapter.sid_to_cell(args.train_label_dir, "diffusion")  # cohort×modality×dx
        cells = [sid2cell.get(adapter.extract_subject_id(it["image"]), "na") for it in train_files]
        weights = temperature_weights(cells, tau)
        train_sampler = DistributedWeightedSampler(weights, world_size, rank, seed=args.seed)
        if rank == 0:
            from collections import Counter
            print(f"[Sampler] imbalance τ={tau}, {len(set(cells))} diffusion cells "
                  f"(cohort×modality×dx); cell sizes={dict(Counter(cells))}")
    else:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=workers_per_gpu,
        sampler=train_sampler, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    steps_per_epoch = len(train_loader)

    val_total = len(valid_files)
    per_rank = (val_total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(val_total, start + per_rank)
    val_files_shard = valid_files[start:end]
    dataset_val = CacheDataset(data=val_files_shard, transform=data_transform,
                               cache_rate=0.0, num_workers=workers_per_gpu)
    valid_loader = DataLoader(
        dataset_val, batch_size=args.val_batch_size, num_workers=workers_per_gpu,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    # ------------------------------------------------------------------
    # Optimizer / scheduler / scaler / loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=args.lr, fused=True)
    total_opt_steps = (args.max_train_steps - args.pretrained_steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    if rank == 0: print("total_opt_steps:", total_opt_steps)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_opt_steps, power=2.0)
    scaler = GradScaler() if use_scaler else None
    if args.loss_type == "l1":
        loss_pt = torch.nn.L1Loss()
    elif args.loss_type == "l2":
        loss_pt = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss_type: {args.loss_type}")

    start_step, best_val_loss, start_epoch = 0, float("inf"), 0
    if args.resume:
        start_step, best_val_loss, start_epoch = resume_from_latest(
            unet, optimizer, lr_scheduler, scaler, weights_unet_dir, device)
    dist.barrier(device_ids=[local_rank])

    if rank == 0: print("Compiling models with torch.compile()...")
    unet = torch.compile(unet)
    dist.barrier(device_ids=[local_rank])
    unet = DDP(unet, device_ids=[local_rank], find_unused_parameters=False)

    sync_vec = torch.tensor([start_step, float(best_val_loss)], device=device, dtype=torch.float32)
    dist.broadcast(sync_vec, src=0)
    start_step, best_val_loss = int(sync_vec[0].item()), float(sync_vec[1].item())

    if rank == 0:
        print(f"# start_epoch: {start_epoch}")
        param_counts = count_parameters(unet)
        print(f"# UNET's Trainable parameters: {param_counts['trainable']:,}")

    def infinite_loader(loader, sampler, start_epoch=0):
        epoch = start_epoch
        while True:
            sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1

    train_iter = infinite_loader(train_loader, train_sampler, start_epoch)
    progress_bar = trange(start_step, args.max_train_steps + 1,
                          desc=f"Training on Rank {rank}",
                          initial=start_step, total=args.max_train_steps + 1,
                          disable=(rank != 0))

    # ==================================================================
    # Training loop
    # ==================================================================
    for step in progress_bar:
        unet.train()
        batch = next(train_iter)
        images = (batch["image"].to(device, non_blocking=True).contiguous() - global_mean) * scale_factor

        if include_body_region:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
        if include_modality:
            modality_tensor = torch.ones((len(images),), dtype=torch.long).to(device)
        spacing_tensor = batch["spacing"].to(device, non_blocking=True) if include_spacing else None
        if use_token_set:
            presence = cfg_drop_presence(
                batch["cond_presence"].to(device, non_blocking=True), cfg_full_p, cfg_token_p)
            meta_tensor = {
                "cond_cat": batch["cond_cat"].to(device, non_blocking=True),
                "cond_cont": batch["cond_cont"].to(device, non_blocking=True),
                "cond_presence": presence,
            }
        else:
            meta_tensor = batch["cond"].to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
                "meta_tensor": meta_tensor,
            }
            if include_body_region:
                unet_inputs.update({
                    "top_region_index_tensor": top_region_index_tensor,
                    "bottom_region_index_tensor": bottom_region_index_tensor,
                })
            if include_modality:
                unet_inputs.update({"class_labels": modality_tensor})

            model_output = unet(**unet_inputs)

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                model_gt = images - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from "
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]"
                )

            loss = loss_pt(model_output.float(), model_gt.float())
            loss = loss / args.gradient_accumulation_steps

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_log = reduce_mean_scalar(loss) * args.gradient_accumulation_steps

        if rank == 0:
            progress_bar.set_postfix({'Total_loss': f"{loss_log:.4f}"})
            if args.report_to and step % 100 == 0:
                log_data = {
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/loss_g_total": loss_log,
                }
                wandb.log(log_data, step=step)

        # ==============================================================
        # Validation
        # ==============================================================
        did_validate = False
        if (step % args.validation_steps == 0 or step == args.max_train_steps) and step > start_step:
            did_validate = True
            unet.eval()
            val_epoch_loss = 0.0
            num_val_batches_local = 0

            with torch.no_grad():
                for val_batch in valid_loader:
                    val_images = (val_batch["image"].to(device, non_blocking=True) - global_mean) * scale_factor
                    spacing_tensor = val_batch["spacing"].to(device, non_blocking=True) if include_spacing else None
                    if use_token_set:
                        # no CFG drop at validation — use the true token set
                        meta_tensor = {
                            "cond_cat": val_batch["cond_cat"].to(device, non_blocking=True),
                            "cond_cont": val_batch["cond_cont"].to(device, non_blocking=True),
                            "cond_presence": val_batch["cond_presence"].to(device, non_blocking=True),
                        }
                    else:
                        meta_tensor = val_batch["cond"].to(device, non_blocking=True)
                    with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                        noise = torch.randn_like(val_images)
                        timesteps = torch.randint(0, num_train_timesteps, (val_images.shape[0],), device=val_images.device).long()
                        noisy_latent = noise_scheduler.add_noise(original_samples=val_images, noise=noise, timesteps=timesteps)

                        unet_inputs = {
                            "x": noisy_latent,
                            "timesteps": timesteps,
                            "spacing_tensor": spacing_tensor,
                            "meta_tensor": meta_tensor,
                        }
                        if include_body_region:
                            top_region_index_tensor = val_batch["top_region_index"].to(device)
                            bottom_region_index_tensor = val_batch["bottom_region_index"].to(device)
                            unet_inputs.update({
                                "top_region_index_tensor": top_region_index_tensor,
                                "bottom_region_index_tensor": bottom_region_index_tensor,
                            })
                        if include_modality:
                            modality_tensor = torch.ones((len(val_images),), dtype=torch.long).to(device)
                            unet_inputs.update({"class_labels": modality_tensor})

                        model_output = unet.module(**unet_inputs)

                        if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                            model_gt = noise
                        elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                            model_gt = val_images
                        elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                            model_gt = val_images - noise
                        else:
                            raise ValueError(
                                "noise scheduler prediction type has to be chosen from "
                                f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]"
                            )

                        val_epoch_loss += loss_pt(model_output.float(), model_gt.float())

                    num_val_batches_local += 1

            val_metrics = torch.tensor([val_epoch_loss, num_val_batches_local], device=device)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)

            total_batches = val_metrics[-1].item()
            avg_loss = val_metrics[0].item() / total_batches if total_batches > 0 else 0.0

            if rank == 0:
                print(f"\nStep {step} Total Val Loss (Avg across all ranks): {avg_loss:.4f}", flush=True)
                if args.report_to:
                    wandb.log({"valid/total_loss": avg_loss,
                               "valid/scale_factor": scale_factor.item()}, step=step)

                if avg_loss < best_val_loss:
                    torch.cuda.synchronize(device)
                    unet_state_dict = unet.module._orig_mod.state_dict()
                    current_epoch = step // steps_per_epoch
                    best_val_loss = float(avg_loss)
                    state = {
                        "unet": unet_state_dict,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "num_train_timesteps": num_train_timesteps,
                        "scale_factor": scale_factor,
                        "global_mean": global_mean,
                        "epoch": current_epoch,
                    }
                    if use_scaler:
                        state["scaler"] = scaler.state_dict()

                    best_dir = os.path.join(weights_unet_dir, "best-checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    atomic_save(state, os.path.join(best_dir, "model.pt"))
                    print(f"[best] updated at step {step}: {best_val_loss:.6f}", flush=True)
                else:
                    print(f"[not best] not updated at step {step}: {avg_loss:.6f}", flush=True)

            _best = torch.tensor([best_val_loss], device=device, dtype=torch.float32)
            dist.broadcast(_best, src=0)
            best_val_loss = float(_best.item())

        # --------------------------------------------------------------
        # Periodic / shutdown checkpoint
        # --------------------------------------------------------------
        is_time_to_save = (step % args.checkpointing_steps == 0 and step > start_step)
        if (is_time_to_save or SHUTDOWN_REQUESTED) and rank == 0:
            torch.cuda.synchronize(device)
            unet_state_dict = unet.module._orig_mod.state_dict()
            current_epoch = step // steps_per_epoch
            state = {
                "unet": unet_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "step": step,
                "best_val_loss": float(best_val_loss),
                "num_train_timesteps": num_train_timesteps,
                "scale_factor": scale_factor,
                "global_mean": global_mean,
                "epoch": current_epoch,
            }
            if use_scaler:
                state["scaler"] = scaler.state_dict()

            ckpt_dir = os.path.join(weights_unet_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            atomic_save(state, os.path.join(ckpt_dir, "model.pt"))
            print(f"\nSaved Step {step} checkpoint to {ckpt_dir}", flush=True)

        shutdown_tensor = torch.tensor([1 if (rank == 0 and SHUTDOWN_REQUESTED) else 0], device=device)
        dist.broadcast(shutdown_tensor, src=0)
        if shutdown_tensor.item() == 1:
            if rank == 0:
                print("Shutdown signal received and synced across all ranks. Exiting training loop gracefully.")
            break
        if did_validate:
            dist.barrier(device_ids=[local_rank])

    if SHUTDOWN_REQUESTED:
        print("Graceful shutdown initiated, exiting with code 1 to trigger requeue.")
        sys.exit(1)

    dist.barrier(device_ids=[local_rank])
    cleanup_ddp()


if __name__ == '__main__':
    main()
