"""Unified VAE training script.

Consolidates four legacy scripts (base / +Cov / +Cor / +Cov+Var) into a single
weight-driven entrypoint. Auxiliary regularizers on the latent mean (z_mu) are
toggled by their `--lambda_*` weight: a value > 0 activates the term; the
default 0.0 short-circuits its computation entirely. This keeps the GPU path
identical to the base run when no auxiliary losses are requested, and makes
later migration to a config system (e.g. Hydra) trivial.
"""

import os
import sys
import json
import time
import yaml
import wandb
import signal
import argparse
import warnings
import numpy as np
from collections import defaultdict
from tqdm import trange

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.config import print_config
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import PatchDiscriminator
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.data import CacheDataset, DataLoader, DistributedSampler

from scripts.config_utils import load_json
from scripts.transforms import VAE_Transform
from scripts.utils import define_instance, dynamic_infer, count_parameters
from scripts.utils_plot import find_label_center_loc, get_xyz_plot
from datasets import get_adapter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Auxiliary latent-space losses (Covariance / Correlation / Variance)
# ---------------------------------------------------------------------------
# 모든 보조 loss는 z_mu(latent mean)에 대해 FP32로 계산한다. autocast 비활성화는
# off-diagonal 합산이 fp16에서 수치적으로 불안정하기 때문이다.

def _flatten_latent(z: torch.Tensor) -> torch.Tensor:
    """[B, C, D, H, W] -> [N, C] where N = B*D*H*W. NaN/Inf-safe."""
    B, C, D, H, W = z.shape
    z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)
    return z_flat


def compute_covariance_loss(z_flat: torch.Tensor) -> torch.Tensor:
    """채널 간 공분산의 off-diagonal L2."""
    N = z_flat.shape[0]
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    cov_matrix = (z_centered.T @ z_centered) / (N - 1 + 1e-6)
    off_diagonal = cov_matrix - torch.diag(torch.diag(cov_matrix))
    return (off_diagonal ** 2).mean()


def compute_correlation_loss(z_flat: torch.Tensor) -> torch.Tensor:
    """채널 간 상관행렬의 off-diagonal L2."""
    N = z_flat.shape[0]
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    z_std = z_centered.std(dim=0, keepdim=True) + 1e-8
    z_normalized = z_centered / z_std
    corr_matrix = (z_normalized.T @ z_normalized) / (N - 1)
    off_diagonal = corr_matrix - torch.diag(torch.diag(corr_matrix))
    return (off_diagonal ** 2).mean()


def compute_variance_loss(z_flat: torch.Tensor, target_var: float = 1.0) -> torch.Tensor:
    """채널별 분산을 target_var로 끌어당겨 rank collapse를 방지."""
    N = z_flat.shape[0]
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    channel_vars = torch.sum(z_centered ** 2, dim=0) / (N - 1 + 1e-6)
    return ((channel_vars - target_var) ** 2).mean()


def compute_aux_losses(
    z: torch.Tensor,
    lambda_cov: float,
    lambda_cor: float,
    lambda_var: float,
    target_var: float = 1.0,
):
    """활성화된(>0) 보조 loss만 계산해 dict로 반환.

    Returns:
        dict[str, Tensor]: 키는 {"cov_loss", "cor_loss", "var_loss"} 중 활성화된 것만 포함.
            아무것도 활성화되지 않으면 빈 dict (zero GPU work).
    """
    active = {}
    if not (lambda_cov > 0.0 or lambda_cor > 0.0 or lambda_var > 0.0):
        return active

    with torch.autocast(device_type="cuda", enabled=False):
        z = z.float()
        z_flat = _flatten_latent(z)

        if torch.isnan(z_flat).any() or torch.isinf(z_flat).any():
            zero = torch.tensor(0.0, device=z.device, requires_grad=True)
            if lambda_cov > 0.0: active["cov_loss"] = zero
            if lambda_cor > 0.0: active["cor_loss"] = zero
            if lambda_var > 0.0: active["var_loss"] = zero
            return active

        if lambda_cov > 0.0:
            active["cov_loss"] = compute_covariance_loss(z_flat)
        if lambda_cor > 0.0:
            active["cor_loss"] = compute_correlation_loss(z_flat)
        if lambda_var > 0.0:
            active["var_loss"] = compute_variance_loss(z_flat, target_var=target_var)

    return active


def aux_weighted_sum(aux: dict, lambda_cov: float, lambda_cor: float, lambda_var: float):
    """compute_aux_losses 결과 dict를 lambda 가중합으로 환원."""
    total = 0.0
    if "cov_loss" in aux: total = total + lambda_cov * aux["cov_loss"]
    if "cor_loss" in aux: total = total + lambda_cor * aux["cor_loss"]
    if "var_loss" in aux: total = total + lambda_var * aux["var_loss"]
    return total  # int(0) or Tensor; addable to loss_g either way


# ---------------------------------------------------------------------------
# DDP / IO utilities (base 스크립트와 동일)
# ---------------------------------------------------------------------------
def setup_ddp():
    # Launch via torchrun (or sbatch + srun + torchrun). LOCAL_RANK / RANK / WORLD_SIZE
    # are set by torchrun; we fall back to single-process defaults if absent, but
    # dist.init_process_group will then require a rendezvous backend to be available.
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


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config_path", type=str, required=True,
                        help="Path to the dataset-level config (e.g. configs/ukb_20252/dataset.json).")
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--train_config_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init_weights_from", type=str, default=None,
                        help="Stage2: init the autoencoder+discriminator from a stage1 "
                             "checkpoint (model weights only; fresh optimizer/scheduler, "
                             "start_step=0). Path to a model.pt or a weights/vae dir. "
                             "Ignored when resuming this stage's own checkpoint.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cpus_per_task", type=int, default=8,
                        help="Number of CPUs allocated per task by Slurm.")
    # Weight-driven aux losses. 0.0 = inactive (no GPU work, no logging).
    parser.add_argument("--lambda_cov", type=float, default=0.0,
                        help="Weight for covariance off-diagonal penalty on z_mu.")
    parser.add_argument("--lambda_cor", type=float, default=0.0,
                        help="Weight for correlation off-diagonal penalty on z_mu.")
    parser.add_argument("--lambda_var", type=float, default=0.0,
                        help="Weight for per-channel variance penalty on z_mu.")
    parser.add_argument("--target_var", type=float, default=1.0,
                        help="Target per-channel variance for the variance penalty.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Stage base directory. If set, overrides custom_config.output_dir from JSON.")
    # Per-user paths / identity. Passed by launcher from env.local.sh; if unset, dataset.json values (typically null) win.
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Source NIfTI dir. Overrides dataset.data_dir.")
    parser.add_argument("--train_label_dir", type=str, default=None,
                        help="train.csv path. Overrides dataset.train_label_dir.")
    parser.add_argument("--valid_label_dir", type=str, default=None,
                        help="valid.csv path. Overrides dataset.valid_label_dir.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity. Overrides dataset.wandb_entity.")
    args = parser.parse_args()

    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run_name to be specified.")
    args.run_name = get_run_name(args.run_name)

    # CLI에서 받은 값 보존 -> JSON merge가 덮어쓰지 못하게 한다.
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}

    dataset_dict = load_json(args.dataset_config_path)
    for k, v in dataset_dict.items():
        setattr(args, k, v)

    config_dict = load_json(args.model_config_path)
    for k, v in config_dict.items():
        setattr(args, k, v)
    config_train_dict = load_json(args.train_config_path)
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
    for k, v in config_train_dict["custom_config"].items():
        setattr(args, k, v)

    # CLI 오버라이드 재적용 (특히 --output_dir).
    for k, v in cli_overrides.items():
        setattr(args, k, v)
    return args


def setup_experiment_dirs(base_dir: str, run_name: str):
    """실험 디렉토리 트리 생성 후 경로 dict 반환.

    구조:
        <base>/<run_name>/
            ├── embeddings/
            ├── logs/
            ├── outputs/
            └── weights/
                └── vae/        # VAE 체크포인트 + config_used.json
    """
    exp_dir = os.path.join(base_dir, run_name)
    paths = {
        "exp": exp_dir,
        "embeddings": os.path.join(exp_dir, "embeddings"),
        "logs": os.path.join(exp_dir, "logs"),
        "outputs": os.path.join(exp_dir, "outputs"),
        "weights": os.path.join(exp_dir, "weights"),
        "vae": os.path.join(exp_dir, "weights", "vae"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def save_used_config(args, vae_dir: str):
    """이번 실험에 실제로 사용된 최종 config (CLI lambda 포함)를 weights/vae/에 스냅샷."""
    snap_path = os.path.join(vae_dir, "config.json")
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


def resume_from_latest(autoencoder, discriminator,
                       optimizer_g, optimizer_d,
                       scheduler_g, scheduler_d,
                       scaler_g, scaler_d,
                       output_dir, device):
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

    autoencoder.load_state_dict(ckpt["autoencoder"])
    discriminator.load_state_dict(ckpt["discriminator"])
    optimizer_g.load_state_dict(ckpt["optimizer_g"])
    optimizer_d.load_state_dict(ckpt["optimizer_d"])
    if "scheduler_g" in ckpt: scheduler_g.load_state_dict(ckpt["scheduler_g"])
    if "scheduler_d" in ckpt: scheduler_d.load_state_dict(ckpt["scheduler_d"])
    if "scaler_g" in ckpt and scaler_g is not None: scaler_g.load_state_dict(ckpt["scaler_g"])
    if "scaler_d" in ckpt and scaler_d is not None: scaler_d.load_state_dict(ckpt["scaler_d"])

    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf")), ckpt.get("epoch", 0)


def load_init_weights(autoencoder, discriminator, ckpt_path, device):
    """Stage2 init from a stage1 checkpoint: load ONLY model weights (autoencoder
    + discriminator); optimizer / scheduler / step start fresh. This is the MAISI
    two-stage recipe — stage2 refines the stage1 autoencoder at a larger patch
    under its own restarted LR schedule, so total training = stage1 + stage2 (NOT
    a single run). ``ckpt_path`` may be a model.pt file or a weights dir holding
    best-checkpoint/model.pt (preferred) or checkpoint-*/model.pt (latest)."""
    p = ckpt_path
    if os.path.isdir(p):
        best = os.path.join(p, "best-checkpoint", "model.pt")
        if os.path.exists(best):
            p = best
        else:
            cks = [d for d in os.listdir(p) if d.startswith("checkpoint-")
                   and os.path.exists(os.path.join(p, d, "model.pt"))]
            if not cks:
                raise FileNotFoundError(f"[init] no checkpoint found under {ckpt_path}")
            latest = sorted(cks, key=lambda x: int(x.split("-")[1]))[-1]
            p = os.path.join(p, latest, "model.pt")
    ckpt = torch.load(p, map_location=device)
    autoencoder.load_state_dict(ckpt["autoencoder"])
    discriminator.load_state_dict(ckpt["discriminator"])
    print(f"[init] loaded stage1 model weights from {p} "
          f"(fresh optimizer/scheduler, start_step=0)", flush=True)


def prepare_image_for_logging(image_tensor, center_loc):
    image_tensor_cpu = image_tensor.cpu()
    vis_img_np = get_xyz_plot(image_tensor_cpu, center_loc, mask_bool=False)
    min_val, max_val = vis_img_np.min(), vis_img_np.max()
    if max_val - min_val > 1e-6:
        vis_img_np = (vis_img_np - min_val) / (max_val - min_val)
    else:
        vis_img_np = np.zeros_like(vis_img_np)
    vis_img_uint8 = (vis_img_np * 255).astype(np.uint8)
    return wandb.Image(vis_img_uint8)


def reduce_mean_scalar(x):
    t = x.detach().reshape(1).to(torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()


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
    if args.weight_dtype == "fp16":
        weight_dtype, amp_dtype = torch.float16, torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype, amp_dtype = torch.bfloat16, torch.bfloat16
    else:
        weight_dtype, amp_dtype = torch.float32, torch.float16
    # bf16 autocast needs no gradient scaling; only fp16 does.
    use_scaler = args.amp and amp_dtype == torch.float16
    adv_warmup = int(getattr(args, "adv_warmup_steps", 0) or 0)

    # 활성화된 보조 loss 키만 따로 저장. 학습 루프 / logging에서 재사용한다.
    aux_active_keys = []
    if args.lambda_cov > 0.0: aux_active_keys.append("cov_loss")
    if args.lambda_cor > 0.0: aux_active_keys.append("cor_loss")
    if args.lambda_var > 0.0: aux_active_keys.append("var_loss")

    if rank == 0:
        print(f"[Opt] Using gradient accumulation with {args.gradient_accumulation_steps} steps.")
        if aux_active_keys:
            print(f"[Aux] Active latent regularizers: {aux_active_keys} "
                  f"(lambda_cov={args.lambda_cov}, lambda_cor={args.lambda_cor}, lambda_var={args.lambda_var})")
        else:
            print("[Aux] No latent regularizers active (base VAE objective).")

    set_determinism(seed=args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Speed > bit-exact reproducibility: cudnn picks faster non-deterministic
    # kernels and benchmark autotunes per input shape. Seed fixes RNG-driven
    # ops (augmentations, initialisation) but cuDNN convolutions may still
    # diverge bit-for-bit across runs on the same seed.
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    # 모든 rank가 같은 경로 트리를 알고 있어야 resume 등에서 안전. mkdir 자체는
    # 멱등이라 모든 rank에서 호출해도 무해.
    exp_paths = setup_experiment_dirs(args.output_dir, args.run_name)
    weights_vae_dir = exp_paths["vae"]

    if rank == 0:
        print("[Config] Loaded hyperparameters:")
        print(yaml.dump(vars(args), sort_keys=False))
        save_used_config(args, weights_vae_dir)
        print(f"[Paths] exp_dir   = {exp_paths['exp']}")
        print(f"[Paths] logs_dir  = {exp_paths['logs']}")
        print(f"[Paths] vae_ckpts = {weights_vae_dir}")
        if args.report_to:
            wandb.init(project=args.wandb_project_vae, entity=args.wandb_entity,
                       config=args, name=args.run_name)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_files = adapter.load_manifest(args.train_label_dir, args.data_dir)
    # Pooled valid CSV is cohort-ordered (UKB-dominated); use balanced
    # cell-stratified selection so the in-loop recon monitor sees every cohort.
    if getattr(args, "dataset_adapter", None) == "pooled" and hasattr(adapter, "load_manifest_stratified"):
        val_files = adapter.load_manifest_stratified(
            args.valid_label_dir, args.data_dir, n=args.num_valid, stage="vae", seed=args.seed)
    else:
        val_files = adapter.load_manifest(args.valid_label_dir, args.data_dir, n=args.num_valid)
    # Per-cohort val recon (pooled): cell label aligned to val_files order. None
    # for adapters that don't tag cells → per-cell logging is skipped.
    val_cells = [f.get("cell") for f in val_files]
    canonical_cells = sorted({c for c in val_cells if c is not None})

    cached_input = getattr(args, "cached_input", False)
    train_transform = VAE_Transform(
        is_train=True, random_aug=args.random_aug, k=4,
        patch_size=args.patch_size, output_dtype=weight_dtype,
        spacing_type=args.spacing_type, image_keys=["image"],
        resolution=args.resolution,
        intensity_norm=args.intensity_norm,
        orientation_axcodes=args.orientation_axcodes,
        select_channel=args.select_channel,
        cached=cached_input,
    )
    val_transform = VAE_Transform(
        is_train=False, random_aug=False, k=4,
        output_dtype=weight_dtype, image_keys=["image"],
        resolution=args.resolution,
        intensity_norm=args.intensity_norm,
        orientation_axcodes=args.orientation_axcodes,
        select_channel=args.select_channel,
        cached=cached_input,
    )

    if rank == 0: print(f"Total number of training data is {len(train_files)}.")
    workers_per_gpu = args.cpus_per_task // world_size
    train_dataset = CacheDataset(data=train_files, transform=train_transform,
                                 cache_rate=args.cache, num_workers=workers_per_gpu)
    if getattr(args, "imbalance_sampling", False) and args.dataset_adapter == "pooled":
        from datasets.sampling import DistributedWeightedSampler, temperature_weights
        tau = float(getattr(args, "sampling_tau", 0.5))
        cells = adapter.cell_labels(args.train_label_dir, "vae")  # cohort×modality
        weights = temperature_weights(cells, tau)
        train_sampler = DistributedWeightedSampler(weights, world_size, rank, seed=args.seed)
        if rank == 0:
            from collections import Counter
            print(f"[Sampler] imbalance temperature τ={tau}, {len(set(cells))} VAE cells "
                  f"(cohort×modality); cell sizes={dict(Counter(cells))}")
    else:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    dataloader_train = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=workers_per_gpu,
        sampler=train_sampler, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    steps_per_epoch = len(dataloader_train)

    if rank == 0: print(f"Total number of validation data is {len(val_files)}.")
    val_total = len(val_files)
    per_rank = (val_total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(val_total, start + per_rank)
    val_files_shard = val_files[start:end]
    val_cells_shard = val_cells[start:end]
    dataset_val = CacheDataset(data=val_files_shard, transform=val_transform,
                               cache_rate=0.0, num_workers=workers_per_gpu)
    dataloader_val = DataLoader(
        dataset_val, batch_size=args.val_batch_size, num_workers=workers_per_gpu,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    if rank == 0:
        print("# Train Transform")
        for i, t in enumerate(train_transform.transform_dict[args.modality].transforms):
            print(f"[{i}] {t}")
        print("\n# Validation Transform")
        for i, t in enumerate(val_transform.transform_dict[args.modality].transforms):
            print(f"[{i}] {t}")

    # ------------------------------------------------------------------
    # Models / optimizers
    # ------------------------------------------------------------------
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims, num_layers_d=3, channels=32,
        in_channels=1, out_channels=1, norm="INSTANCE",
    ).to(device)

    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), lr=args.lr,
                                    weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=args.lr,
                                    weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    total_opt_steps = (args.max_train_steps - args.pretrained_steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    if rank == 0: print("total_opt_steps:", total_opt_steps)
    scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=total_opt_steps)
    scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=total_opt_steps)
    scaler_g, scaler_d = (GradScaler(), GradScaler()) if use_scaler else (None, None)

    start_step, best_val_loss, start_epoch = 0, float("inf"), 0
    resumed = False
    if args.resume:
        start_step, best_val_loss, start_epoch = resume_from_latest(
            autoencoder, discriminator, optimizer_g, optimizer_d,
            scheduler_g, scheduler_d, scaler_g, scaler_d, weights_vae_dir, device)
        resumed = start_step > 0
    # stage2: if this stage has no checkpoint of its own yet, initialise from the
    # stage1 weights. On requeue (own checkpoint present) resume wins and init is
    # skipped, so we never clobber stage2 progress.
    if not resumed and getattr(args, "init_weights_from", None):
        load_init_weights(autoencoder, discriminator, args.init_weights_from, device)
    dist.barrier(device_ids=[local_rank])

    if rank == 0: print("Compiling models with torch.compile()...")
    autoencoder = torch.compile(autoencoder)
    discriminator = torch.compile(discriminator)
    dist.barrier(device_ids=[local_rank])
    autoencoder = DDP(autoencoder, device_ids=[local_rank], find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)

    sync_vec = torch.tensor([start_step, float(best_val_loss)], device=device, dtype=torch.float32)
    dist.broadcast(sync_vec, src=0)
    start_step, best_val_loss = int(sync_vec[0].item()), float(sync_vec[1].item())

    # ------------------------------------------------------------------
    # Loss heads
    # ------------------------------------------------------------------
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                     is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

    if rank == 0:
        print("# start_epoch:", start_epoch)
        param_counts = count_parameters(autoencoder.module)
        print(f"# autoencoder's Trainable parameters: {param_counts['trainable']:,}")
        param_counts = count_parameters(discriminator.module)
        print(f"# discriminator's Trainable parameters: {param_counts['trainable']:,}")

    def infinite_loader(loader, sampler, start_epoch=0):
        epoch = start_epoch
        while True:
            sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1

    train_iter = infinite_loader(dataloader_train, train_sampler, start_epoch)
    progress_bar = trange(start_step, args.max_train_steps + 1,
                          desc=f"Training on Rank {rank}",
                          initial=start_step, total=args.max_train_steps + 1,
                          disable=(rank != 0))

    # Reused every validation pass; construct once outside the loop.
    val_inferer = SlidingWindowInferer(
        roi_size=args.val_sliding_window_patch_size, sw_batch_size=1,
        overlap=0.5, device=device, sw_device=device,
    )

    # Running stats for periodic logging (grad norms, throughput). Initialised so
    # the first logging step (before any optimizer step / during warmup) is valid.
    gnorm_g = torch.zeros((), device=device)
    gnorm_d = torch.zeros((), device=device)
    t_window = time.time()

    # ==================================================================
    # Training loop
    # ==================================================================
    for step in progress_bar:
        autoencoder.train()
        discriminator.train()
        batch = next(train_iter)
        images = batch["image"].to(device, non_blocking=True).contiguous()

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
            reconstruction, z_mu, z_sigma = autoencoder(images)

        z_mu_f = z_mu.float()
        z_sigma_f = torch.clamp(z_sigma.float(), min=1e-8)
        logvar = 2.0 * torch.log(z_sigma_f)
        logvar = torch.clamp(logvar, min=-30.0, max=10.0)
        kl = 0.5 * (torch.exp(logvar) + z_mu_f ** 2 - 1.0 - logvar)
        kl_loss = kl.mean()

        # 활성화된 보조 loss만 계산. 빈 dict면 GPU work도 0.
        aux_losses = compute_aux_losses(
            z_mu, args.lambda_cov, args.lambda_cor, args.lambda_var,
            target_var=args.target_var,
        )

        adv_active = step >= adv_warmup
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
            losses = {
                "recons_loss": intensity_loss(reconstruction, images),
                "kl_loss": kl_loss,
                "p_loss": loss_perceptual(reconstruction.float(), images.float()),
            }
            losses.update(aux_losses)

            # Adversarial warmup: AE learns reconstruction first; no adv gradient
            # to the generator (and the discriminator is frozen) until adv_warmup
            # steps. Skip the generator's discriminator forward during warmup —
            # it contributes 0 (adv_w=0) but would otherwise cost a full disc
            # forward + graph every warmup step. adv_warmup=0 (MIUA) → original.
            if adv_active:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.zeros((), device=device)
            adv_w = args.adv_weight if adv_active else 0.0
            # Total = Recon + kl_w*KL + perc_w*P + adv_w*Adv + Σ λ_i * aux_i
            loss_g = (
                losses["recons_loss"]
                + args.kl_weight * losses["kl_loss"]
                + args.perceptual_weight * losses["p_loss"]
                + adv_w * generator_loss
                + aux_weighted_sum(aux_losses, args.lambda_cov, args.lambda_cor, args.lambda_var)
            )
            loss_g = loss_g / args.gradient_accumulation_steps

        if use_scaler:
            scaler_g.scale(loss_g).backward()
        else:
            loss_g.backward()

        if adv_active:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * args.disc_weight
                loss_d = loss_d / args.gradient_accumulation_steps
            if use_scaler:
                scaler_d.scale(loss_d).backward()
            else:
                loss_d.backward()
        else:
            loss_d = loss_d_fake = loss_d_real = torch.zeros((), device=device)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Generator
            if use_scaler:
                scaler_g.unscale_(optimizer_g)
                gnorm_g = clip_grad_norm_(autoencoder.parameters(), 1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                gnorm_g = clip_grad_norm_(autoencoder.parameters(), 1.0)
                optimizer_g.step()
            scheduler_g.step()
            optimizer_g.zero_grad(set_to_none=True)

            # Discriminator (frozen during adversarial warmup)
            if adv_active:
                if use_scaler:
                    scaler_d.unscale_(optimizer_d)
                    gnorm_d = clip_grad_norm_(discriminator.parameters(), 1.0)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    gnorm_d = clip_grad_norm_(discriminator.parameters(), 1.0)
                    optimizer_d.step()
                scheduler_d.step()
                optimizer_d.zero_grad(set_to_none=True)

        # --------------------------------------------------------------
        # Logging. Each reduce_mean_scalar is an all_reduce (collective) + a
        # .item() (CPU-GPU sync); doing ~8 of them every step serializes the
        # pipeline. Gate the whole block on the logging cadence so non-logging
        # steps incur zero collective/sync overhead. The progress-bar loss
        # postfix updates with the same cadence. (reduce_* must be called by ALL
        # ranks — keep them outside the rank-0 guard.)
        # --------------------------------------------------------------
        if step % 100 == 0:
            gas = args.gradient_accumulation_steps
            lg = reduce_mean_scalar(loss_g) * gas
            ld = reduce_mean_scalar(loss_d) * gas
            avg_recons_loss = reduce_mean_scalar(losses["recons_loss"]) * gas
            avg_kl_loss = reduce_mean_scalar(losses["kl_loss"]) * gas
            avg_p_loss = reduce_mean_scalar(losses["p_loss"]) * gas
            avg_gen_loss = reduce_mean_scalar(generator_loss) * gas
            avg_dfake_loss = reduce_mean_scalar(loss_d_fake) * gas
            avg_dreal_loss = reduce_mean_scalar(loss_d_real) * gas
            # 활성화된 보조 loss만 reduce -> 비활성 항목에 대한 통신 / 로깅 비용 0
            avg_aux = {k: reduce_mean_scalar(v) * gas for k, v in aux_losses.items()}
            # Monitors (all ranks must call reduce): latent channel decorrelation
            # (the L_cor objective, logged even when lambda_cor=0 to compare the
            # baseline against L_SID/L_VAD) + pre-clip grad norms (divergence watch).
            corr_mon = reduce_mean_scalar(
                compute_correlation_loss(_flatten_latent(z_mu.detach().float())))
            gnorm_g_r = reduce_mean_scalar(gnorm_g)
            gnorm_d_r = reduce_mean_scalar(gnorm_d)

            if rank == 0:
                progress_bar.set_postfix({'Total_g_loss': f"{lg:.4f}", 'Total_d_loss': f"{ld:.4f}"})
                if args.report_to:
                    log_data = {
                        "train/learning_rate": scheduler_g.get_last_lr()[0],
                        "train/loss_g_total": lg,
                        "train/loss_d_total": ld,
                        "train/z_mu_mean": z_mu_f.mean().item(),
                        "train/z_sigma_mean": z_sigma_f.mean().item(),
                        "train/generator/recons_loss": avg_recons_loss,
                        "train/generator/kl_loss": avg_kl_loss,
                        "train/generator/p_loss": avg_p_loss,
                        "train/discriminator/adv_g_loss": avg_gen_loss,
                        "train/discriminator/d_fake_loss": avg_dfake_loss,
                        "train/discriminator/d_real_loss": avg_dreal_loss,
                        "train/latent/offdiag_corr": corr_mon,
                        "train/grad_norm/generator": gnorm_g_r,
                        "train/grad_norm/discriminator": gnorm_d_r,
                    }
                    # Dynamic logging: 활성화된 보조 loss만 추가.
                    for k, v in avg_aux.items():
                        log_data[f"train/generator/{k}"] = v
                    # Throughput / memory (for tuning + cross-method compute budget
                    # accounting, e.g. vs 3D MedDiffusion). Windowed over the 100-step
                    # logging interval; total GPU-hours come from `sacct` (handles requeue).
                    now = time.time()
                    dt = now - t_window
                    t_window = now
                    if dt > 0:
                        itps = 100.0 / dt
                        log_data["perf/it_per_s"] = itps
                        log_data["perf/samples_per_s"] = itps * args.batch_size * world_size
                    log_data["perf/peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    torch.cuda.reset_peak_memory_stats(device)
                    wandb.log(log_data, step=step)

        # ==============================================================
        # Validation
        # ==============================================================
        did_validate = False
        if step % args.validation_steps == 0 and step > start_step:
            did_validate = True
            autoencoder.eval()
            val_epoch_losses = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0,
                                "z_mu_mean": 0.0, "z_sigma_mean": 0.0}
            for k in aux_active_keys:
                val_epoch_losses[k] = 0.0
            num_val_batches_local = 0
            cell_recon_sum = defaultdict(float)   # per-cohort recon (val_batch_size=1)
            cell_recon_cnt = defaultdict(int)

            with torch.no_grad():
                for vi, val_batch in enumerate(dataloader_val):
                    val_images = val_batch["image"].to(device)
                    with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                        reconstruction, z_mu_val, z_sigma_val = dynamic_infer(val_inferer, autoencoder.module, val_images)

                    z_mu_val_f = z_mu_val.float()
                    z_sigma_val_f = torch.clamp(z_sigma_val.float(), min=1e-8)
                    logvar_val = 2.0 * torch.log(z_sigma_val_f)
                    logvar_val = torch.clamp(logvar_val, min=-30.0, max=10.0)
                    kl_val = 0.5 * (torch.exp(logvar_val) + z_mu_val_f ** 2 - 1.0 - logvar_val)
                    kl_loss_val = kl_val.mean().item()

                    aux_losses_val = compute_aux_losses(
                        z_mu_val, args.lambda_cov, args.lambda_cor, args.lambda_var,
                        target_var=args.target_var,
                    )

                    reconstruction = reconstruction.to(device)
                    val_images = val_images.to(device)
                    recon_l = intensity_loss(reconstruction, val_images).item()
                    cell = val_cells_shard[vi] if vi < len(val_cells_shard) else None
                    if cell is not None:
                        cell_recon_sum[cell] += recon_l
                        cell_recon_cnt[cell] += 1
                    val_epoch_losses["recons_loss"] += recon_l
                    val_epoch_losses["kl_loss"] += kl_loss_val
                    val_epoch_losses["p_loss"] += loss_perceptual(reconstruction.float(), val_images.float()).item()
                    val_epoch_losses["z_mu_mean"] += z_mu_val_f.mean().item()
                    val_epoch_losses["z_sigma_mean"] += z_sigma_val_f.mean().item()
                    for k, v in aux_losses_val.items():
                        val_epoch_losses[k] += v.item()
                    num_val_batches_local += 1

            # 고정 5개 metric + 활성화된 보조 loss + count. 항상 동일한 순서로 packing.
            ordered_keys = ["recons_loss", "kl_loss", "p_loss", "z_mu_mean", "z_sigma_mean"] + aux_active_keys
            metric_values = [val_epoch_losses[k] for k in ordered_keys] + [num_val_batches_local]
            val_metrics = torch.tensor(metric_values, device=device, dtype=torch.float32)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)

            total_batches = val_metrics[-1].item()
            denom = total_batches if total_batches > 0 else 1.0
            reduced = {k: val_metrics[i].item() / denom for i, k in enumerate(ordered_keys)}

            avg_recon_loss = reduced["recons_loss"]
            avg_kl_loss_v = reduced["kl_loss"]
            avg_p_loss_v = reduced["p_loss"]
            avg_z_mu_mean_v = reduced["z_mu_mean"]
            avg_z_sigma_mean_v = reduced["z_sigma_mean"]

            # Per-cohort recon: pack (sum, count) per canonical cell, all-reduce,
            # then average. canonical_cells is identical on every rank.
            per_cell_recon = {}
            if canonical_cells:
                cidx = {c: i for i, c in enumerate(canonical_cells)}
                nC = len(canonical_cells)
                cell_t = torch.zeros(2 * nC, device=device, dtype=torch.float32)
                for c, s in cell_recon_sum.items():
                    cell_t[cidx[c]] = s
                for c, n in cell_recon_cnt.items():
                    cell_t[nC + cidx[c]] = n
                dist.all_reduce(cell_t, op=dist.ReduceOp.SUM)
                for c, i in cidx.items():
                    n = cell_t[nC + i].item()
                    if n > 0:
                        per_cell_recon[c] = cell_t[i].item() / n

            val_loss_g = (
                avg_recon_loss
                + args.kl_weight * avg_kl_loss_v
                + args.perceptual_weight * avg_p_loss_v
                + args.lambda_cov * reduced.get("cov_loss", 0.0)
                + args.lambda_cor * reduced.get("cor_loss", 0.0)
                + args.lambda_var * reduced.get("var_loss", 0.0)
            )

            if rank == 0:
                print(f"\nStep {step} Total Val Loss: {val_loss_g:.4f}, "
                      f"z_mu: {avg_z_mu_mean_v:.4f}, z_sigma: {avg_z_sigma_mean_v:.4f}")
                if args.report_to:
                    log_data = {
                        "valid/total_loss": val_loss_g,
                        "valid/recon_loss": avg_recon_loss,
                        "valid/kl_loss": avg_kl_loss_v,
                        "valid/p_loss": avg_p_loss_v,
                        "valid/z_mu_mean": avg_z_mu_mean_v,
                        "valid/z_sigma_mean": avg_z_sigma_mean_v,
                    }
                    for k in aux_active_keys:
                        log_data[f"valid/{k}"] = reduced[k]
                    for c, v in per_cell_recon.items():
                        log_data[f"valid/recon_by_cell/{c}"] = v
                    if num_val_batches_local > 0:
                        std = z_mu_val.detach().float().flatten().std().clamp(min=1e-8)
                        log_data["valid/scale_factor"] = (1.0 / std).item()
                        center_loc = find_label_center_loc(val_images[0, 0, ...])
                        log_data["valid/original_image"] = prepare_image_for_logging(val_images[0], center_loc)
                        log_data["valid/reconstructed_image"] = prepare_image_for_logging(reconstruction[0], center_loc)
                    wandb.log(log_data, step=step)

                if val_loss_g < best_val_loss:
                    torch.cuda.synchronize(device)
                    autoencoder_state_dict = autoencoder.module._orig_mod.state_dict()
                    discriminator_state_dict = discriminator.module._orig_mod.state_dict()
                    current_epoch = step // steps_per_epoch
                    best_val_loss = float(val_loss_g)
                    state = {
                        "autoencoder": autoencoder_state_dict,
                        "discriminator": discriminator_state_dict,
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict(),
                        "scheduler_d": scheduler_d.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "epoch": current_epoch,
                    }
                    if args.amp:
                        state["scaler_g"] = scaler_g.state_dict()
                        state["scaler_d"] = scaler_d.state_dict()
                    best_dir = os.path.join(weights_vae_dir, "best-checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    atomic_save(state, os.path.join(best_dir, "model.pt"))
                    print(f"[best] updated at step {step}: {best_val_loss:.6f}", flush=True)
                else:
                    print(f"[not best] not updated at step {step}: {val_loss_g:.6f}", flush=True)

            _best = torch.tensor([best_val_loss], device=device, dtype=torch.float32)
            dist.broadcast(_best, src=0)
            best_val_loss = float(_best.item())

        # --------------------------------------------------------------
        # Periodic / shutdown checkpoint
        # --------------------------------------------------------------
        is_time_to_save = (step % args.checkpointing_steps == 0 and step > start_step)
        if (is_time_to_save or SHUTDOWN_REQUESTED) and rank == 0:
            torch.cuda.synchronize(device)
            autoencoder_state_dict = autoencoder.module._orig_mod.state_dict()
            discriminator_state_dict = discriminator.module._orig_mod.state_dict()
            current_epoch = step // steps_per_epoch
            state = {
                "autoencoder": autoencoder_state_dict,
                "discriminator": discriminator_state_dict,
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "step": step,
                "best_val_loss": float(best_val_loss),
                "epoch": current_epoch,
            }
            if use_scaler:
                state["scaler_g"] = scaler_g.state_dict()
                state["scaler_d"] = scaler_d.state_dict()
            ckpt_dir = os.path.join(weights_vae_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            atomic_save(state, os.path.join(ckpt_dir, "model.pt"))
            print(f"\nSaved checkpoint to {ckpt_dir}", flush=True)

        shutdown_tensor = torch.tensor([1 if (rank == 0 and SHUTDOWN_REQUESTED) else 0], device=device)
        dist.broadcast(shutdown_tensor, src=0)
        if shutdown_tensor.item() == 1:
            if rank == 0:
                print("Shutdown signal received and synced across all ranks. Exiting training loop gracefully.")
            break
        if did_validate:
            dist.barrier(device_ids=[local_rank])

    if SHUTDOWN_REQUESTED:
        sys.exit(0)

    dist.barrier(device_ids=[local_rank])
    cleanup_ddp()


if __name__ == '__main__':
    main()
