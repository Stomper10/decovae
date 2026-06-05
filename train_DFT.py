import os
import sys
import json
import yaml
import wandb
import signal
import types  # [ADDED] for forward method overriding
import argparse
import warnings
import numpy as np
from tqdm import trange

import torch
import torch.distributed as dist
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

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
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Stage base directory. If set, overrides custom_config.output_dir from JSON.")

    # Pretrained model path & Latent Noise Scale for Decoder Fine-tuning
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to the pre-trained model.pt")
    parser.add_argument("--latent_noise_scale", type=float, default=1.0,
                        help="Multiplier for latent noise sampling")
    parser.add_argument("--ploss_model", type=str, default='squeeze',
                        help="Model type of perceptual loss")

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
            ├── logs/
            ├── outputs/
            └── weights/
                └── vae/        # DFT 체크포인트 + config.json
    """
    exp_dir = os.path.join(base_dir, run_name)
    paths = {
        "exp": exp_dir,
        "logs": os.path.join(exp_dir, "logs"),
        "outputs": os.path.join(exp_dir, "outputs"),
        "weights": os.path.join(exp_dir, "weights"),
        "vae": os.path.join(exp_dir, "weights", "vae"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def save_used_config(args, vae_dir: str):
    """이번 실험에 실제로 사용된 최종 config (CLI 인자 포함)를 weights/vae/에 스냅샷."""
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

# [ADDED] Custom forward function to inject noise scale properly within DDP
def custom_forward(self, x, noise_scale=1.0):
    z_mu, z_sigma = self.encode(x)
    # Sampling with amplified noise
    noise = torch.randn_like(z_sigma)
    z = z_mu + noise * z_sigma * noise_scale
    reconstruction = self.decode(z)
    return reconstruction, z_mu, z_sigma

def main():
    setup_ddp()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank == 0:
        print("=" * 50)
        print(f"✅ Decoder Fine-tuning started with a total of {world_size} GPUs across all nodes.")
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

    set_determinism(seed=args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Speed > bit-exact reproducibility. See train_VAE.py for the full note.
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    # 모든 rank가 같은 경로 트리를 알고 있어야 resume 등에서 안전. mkdir 자체는
    # 멱등이라 모든 rank에서 호출해도 무해.
    exp_paths = setup_experiment_dirs(args.output_dir, args.run_name)
    weights_vae_dir = exp_paths["vae"]

    if rank == 0:
        print(f"[Opt] Latent Noise Scale: {args.latent_noise_scale}")
        print("[Config] Loaded hyperparameters:")
        print(yaml.dump(vars(args), sort_keys=False))
        save_used_config(args, weights_vae_dir)
        print(f"[Paths] exp_dir   = {exp_paths['exp']}")
        print(f"[Paths] logs_dir  = {exp_paths['logs']}")
        print(f"[Paths] vae_ckpts = {weights_vae_dir}")
        if args.report_to:
            wandb.init(project=args.wandb_project_dft, entity=args.wandb_entity,
                       config=args, name=args.run_name)

    # Data
    train_files = adapter.load_manifest(args.train_label_dir, args.data_dir)
    val_files = adapter.load_manifest(args.valid_label_dir, args.data_dir, n=args.num_valid)

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

    workers_per_gpu = args.cpus_per_task // world_size 
    train_dataset = CacheDataset(data=train_files, transform=train_transform, cache_rate=args.cache, num_workers=workers_per_gpu)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    dataloader_train = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=workers_per_gpu, 
        sampler=train_sampler, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2
    )
    steps_per_epoch = len(dataloader_train)
    
    val_total = len(val_files)
    per_rank = (val_total + world_size - 1) // world_size  
    start = rank * per_rank
    end = min(val_total, start + per_rank)
    val_files_shard = val_files[start:end]
    dataset_val = CacheDataset(data=val_files_shard, transform=val_transform, cache_rate=0.0, num_workers=workers_per_gpu)
    dataloader_val = DataLoader(
        dataset_val, batch_size=args.val_batch_size, num_workers=workers_per_gpu, 
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # ---------------------------------------------------------
    # [MODIFIED] Model Definition, Loading & Freezing
    # ---------------------------------------------------------
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    
    # Load Pre-trained Weights
    if rank == 0: print(f"Loading pretrained weights from {args.pretrained_model_path}...")
    ckpt = torch.load(args.pretrained_model_path, map_location=device)
    autoencoder.load_state_dict(ckpt["autoencoder"])

    # Override the forward method to inject noise_scale
    autoencoder.forward = types.MethodType(custom_forward, autoencoder)

    # Freeze Encoder, Unfreeze Decoder & Post-Quant Conv
    for name, param in autoencoder.named_parameters():
        if "decoder" in name or "quant_conv_d" in name or "post_quant" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims, num_layers_d=3, channels=32,
        in_channels=1, out_channels=1, norm="INSTANCE",
    ).to(device)
    # (Discriminator weights can also be loaded here if desired, otherwise it trains from scratch)

    # [MODIFIED] Optimizer now only updates trainable parameters (Decoder)
    # Materialize the filter generator so it survives optimizer.zero_grad() and the
    # autoencoder state-dict swap during checkpoint loading.
    trainable_ae_params = list(filter(lambda p: p.requires_grad, autoencoder.parameters()))
    optimizer_g = torch.optim.AdamW(params=trainable_ae_params, lr=args.lr, weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-6 if args.amp else 1e-8)
    
    total_opt_steps = (args.max_train_steps - args.pretrained_steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=total_opt_steps)
    scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=total_opt_steps)
    scaler_g, scaler_d = (GradScaler(), GradScaler()) if use_scaler else (None, None)

    start_step, best_val_loss, start_epoch = 0, float("inf"), 0
    if args.resume:
        # weights/vae/ 아래에 checkpoint-* 폴더가 있는지 확인
        ckpt_path = None

        if rank == 0:
            if os.path.exists(weights_vae_dir):
                checkpoints = [d for d in os.listdir(weights_vae_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    latest_ckpt = checkpoints[-1]
                    ckpt_path = os.path.join(weights_vae_dir, latest_ckpt, "model.pt")
            
        obj_list = [ckpt_path]
        dist.broadcast_object_list(obj_list, src=0)
        ckpt_path = obj_list[0]

        if ckpt_path and os.path.exists(ckpt_path):
            if rank == 0: print(f"🚀 Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            autoencoder.load_state_dict(checkpoint["autoencoder"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            scheduler_g.load_state_dict(checkpoint["scheduler_g"])
            scheduler_d.load_state_dict(checkpoint["scheduler_d"])
            
            if use_scaler and "scaler_g" in checkpoint:
                scaler_g.load_state_dict(checkpoint["scaler_g"])
                scaler_d.load_state_dict(checkpoint["scaler_d"])
            
            start_step = checkpoint["step"] + 1
            start_epoch = checkpoint.get("epoch", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        else:
            if rank == 0: print("⚠️ Resume enabled but no checkpoint found. Starting from scratch/pretrained.")

    dist.barrier(device_ids=[local_rank])

    if rank == 0: print("Compiling models with torch.compile()...")
    autoencoder = torch.compile(autoencoder)
    discriminator = torch.compile(discriminator)
    dist.barrier(device_ids=[local_rank])
    
    # find_unused_parameters=True is required because the encoder's output doesn't require grad
    autoencoder = DDP(autoencoder, device_ids=[local_rank], find_unused_parameters=True)
    discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)

    sync_vec = torch.tensor([start_step, float(best_val_loss)], device=device, dtype=torch.float32)
    dist.broadcast(sync_vec, src=0)
    start_step, best_val_loss = int(sync_vec[0].item()), float(sync_vec[1].item())

    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type=args.ploss_model, is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
        
    if rank == 0:
        param_counts = count_parameters(autoencoder.module)
        print(f"# autoencoder's Trainable parameters (Decoder only): {param_counts['trainable']:,}")

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
        roi_size=args.val_sliding_window_patch_size,
        sw_batch_size=1, overlap=0.5, device=device, sw_device=device,
    )

    for step in progress_bar:
        autoencoder.train()
        discriminator.train()
        batch = next(train_iter)
        images = batch["image"].to(device, non_blocking=True).contiguous()

        # [MODIFIED] Pass latent_noise_scale to the forward pass
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
            reconstruction, z_mu, z_sigma = autoencoder(images, noise_scale=args.latent_noise_scale)
            
            recons_loss = intensity_loss(reconstruction, images)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_lc = torch.tensor(0.0, device=device)
            if 'latcon' in args.run_name:
                z_mu_rec, _ = autoencoder.module.encode(reconstruction)
                
                z1 = z_mu.flatten(1).float()
                z2 = z_mu_rec.flatten(1).float()
                
                cos_sim = F.cosine_similarity(z1, z2, dim=1).mean()
                loss_lc = 1.0 - cos_sim

            # 3. 손실 딕셔너리 구성
            losses = {
                "recons_loss": recons_loss,
                "p_loss": p_loss,
                "lc_loss": loss_lc
            }
            
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            
            # [MODIFIED] loss_g now only contains Recons + Perceptual + Adversarial
            loss_g = losses["recons_loss"] + args.perceptual_weight * losses["p_loss"] + args.adv_weight * generator_loss
            if 'latcon' in args.run_name:
                loss_g += losses["lc_loss"]
            loss_g = loss_g / args.gradient_accumulation_steps
            
        if use_scaler:
            scaler_g.scale(loss_g).backward()
        else:
            loss_g.backward()

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

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if use_scaler:
                scaler_g.unscale_(optimizer_g)
                clip_grad_norm_(autoencoder.parameters(), 1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                clip_grad_norm_(autoencoder.parameters(), 1.0)
                optimizer_g.step()
            scheduler_g.step()
            optimizer_g.zero_grad(set_to_none=True)

            if use_scaler:
                scaler_d.unscale_(optimizer_d)
                clip_grad_norm_(discriminator.parameters(), 1.0)
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_d.step()
            scheduler_d.step()
            optimizer_d.zero_grad(set_to_none=True)

        lg = reduce_mean_scalar(loss_g) * args.gradient_accumulation_steps
        ld = reduce_mean_scalar(loss_d) * args.gradient_accumulation_steps
        avg_recons_loss = reduce_mean_scalar(losses["recons_loss"]) * args.gradient_accumulation_steps
        avg_p_loss = reduce_mean_scalar(losses["p_loss"]) * args.gradient_accumulation_steps
        avg_gen_loss = reduce_mean_scalar(generator_loss) * args.gradient_accumulation_steps
        avg_dfake_loss = reduce_mean_scalar(loss_d_fake) * args.gradient_accumulation_steps
        avg_dreal_loss = reduce_mean_scalar(loss_d_real) * args.gradient_accumulation_steps
        avg_lc_loss = 0.0
        if "lc_loss" in losses:
            avg_lc_loss = reduce_mean_scalar(losses["lc_loss"]) * args.gradient_accumulation_steps

        if rank == 0:
            progress_bar.set_postfix({
                'Total_g': f"{lg:.4f}", 
                'Total_d': f"{ld:.4f}", 
                'LC': f"{avg_lc_loss:.4f}"
            })

            if args.report_to and step % 100 == 0: 
                log_data = {
                    "train/learning_rate": scheduler_g.get_last_lr(),
                    "train/loss_g_total": lg,
                    "train/loss_d_total": ld,
                    "train/generator/recons_loss": avg_recons_loss,
                    "train/generator/p_loss": avg_p_loss,
                    "train/generator/lc_loss": avg_lc_loss, # LC loss 로그
                    "train/discriminator/adv_g_loss": avg_gen_loss,
                    "train/discriminator/d_fake_loss": avg_dfake_loss,
                    "train/discriminator/d_real_loss": avg_dreal_loss,
                }
                wandb.log(log_data, step=step)

        did_validate = False
        if step % args.validation_steps == 0 and step > start_step:
            did_validate = True
            autoencoder.eval()
            val_epoch_losses = {"recons_loss": 0, "p_loss": 0, "z_mu_mean": 0, "z_sigma_mean": 0}
            num_val_batches_local = 0
            with torch.no_grad():
                for val_batch in dataloader_val:
                    val_images = val_batch["image"].to(device)
                    with autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                        # [MODIFIED] Using module to bypass DDP, assuming dynamic_infer handles the rest.
                        # Validation can usually run with scale=1.0 or user-defined.
                        reconstruction, z_mu_val, z_sigma_val = dynamic_infer(val_inferer, autoencoder.module, val_images)

                    z_mu_val_f = z_mu_val.float()
                    z_sigma_val_f = torch.clamp(z_sigma_val.float(), min=1e-8)
                    
                    reconstruction = reconstruction.to(device)
                    val_images = val_images.to(device)
                    val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, val_images).item()
                    val_epoch_losses["p_loss"] += loss_perceptual(reconstruction.float(), val_images.float()).item()
                    val_epoch_losses["z_mu_mean"] += z_mu_val_f.mean().item()
                    val_epoch_losses["z_sigma_mean"] += z_sigma_val_f.mean().item()
                    num_val_batches_local += 1

            val_metrics = torch.tensor(
                [
                    val_epoch_losses["recons_loss"], 
                    val_epoch_losses["p_loss"], 
                    val_epoch_losses["z_mu_mean"], 
                    val_epoch_losses["z_sigma_mean"], 
                    num_val_batches_local
                ],
                device=device
            )
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
            total_batches = val_metrics[4].item()
            avg_recon_loss = val_metrics[0].item() / total_batches if total_batches > 0 else 0
            avg_p_loss = val_metrics[1].item() / total_batches if total_batches > 0 else 0
            avg_z_mu_mean = val_metrics[2].item() / total_batches if total_batches > 0 else 0
            avg_z_sigma_mean = val_metrics[3].item() / total_batches if total_batches > 0 else 0
            
            # [MODIFIED] Validation loss calculation (No KL)
            val_loss_g = avg_recon_loss + args.perceptual_weight * avg_p_loss

            if rank == 0:                
                print(f"\nStep {step} Total Val Loss: {val_loss_g:.4f}, z_mu: {avg_z_mu_mean:.4f}, z_sigma: {avg_z_sigma_mean:.4f}") 
                if args.report_to:
                    log_data = {
                        "valid/total_loss": val_loss_g,
                        "valid/recon_loss": avg_recon_loss,
                        "valid/p_loss": avg_p_loss,
                        "valid/z_mu_mean": avg_z_mu_mean,
                        "valid/z_sigma_mean": avg_z_sigma_mean,
                    }
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
                        "epoch": current_epoch
                    }
                    if use_scaler:
                        state["scaler_g"] = scaler_g.state_dict()
                        state["scaler_d"] = scaler_d.state_dict()
                    best_dir = os.path.join(weights_vae_dir, "best-checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    atomic_save(state, os.path.join(best_dir, "model.pt"))
                    print(f"[best] updated at step {step}: {best_val_loss:.6f}", flush=True)

            _best = torch.tensor([best_val_loss], device=device, dtype=torch.float32)
            dist.broadcast(_best, src=0)
            best_val_loss = float(_best.item())

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
                "epoch": current_epoch
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
            break
        if did_validate:
            dist.barrier(device_ids=[local_rank])

    if SHUTDOWN_REQUESTED:
        sys.exit(0)
    
    dist.barrier(device_ids=[local_rank]) 
    cleanup_ddp()

if __name__ == '__main__':
    main()
