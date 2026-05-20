"""ControlNet (mask-conditional) latent-diffusion training script.

Mirrors `train_UNET.py` scaffolding (DDP, resume, atomic checkpoints, etc.),
but trains a `ControlNetMaisi` on top of a *frozen* trained UNet. Adapted
from NV-Generate-CTMR ``scripts/train_controlnet.py``:

  1. Load trained UNet via ``--trained_diffusion_path``
  2. Freeze UNet (``requires_grad=False``); set ``eval()``
  3. Build ControlNet via ``define_instance(args, "controlnet_def")``
  4. Initialize ControlNet weights from UNet via ``copy_model_state``
  5. Each step: ``controlnet(noisy_latent, t, mask) → down_res + mid_res``
     fed into ``unet(noisy_latent, t, *additional_residuals)`` for the
     diffusion loss

Use case: BraTS tumor-mask conditional generation (downstream synth-aug).
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
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange

import monai
from monai.transforms import Compose
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, DistributedSampler
from monai.networks import copy_model_state
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.schedulers.rectified_flow import RFlowScheduler

from scripts.config_utils import load_json
from scripts.utils import define_instance, count_parameters
from datasets import get_adapter
import patches  # noqa: F401  # registers DiffusionModelUNetMaisiV2 + RFlowSchedulerV2

# Reuse train_UNET.py utilities (DDP setup, save helpers, resume scaffolding).
# load_config is overridden below because controlnet_train.json uses a
# different top-level block (`controlnet_train`) and we add an extra CLI
# flag for the trained UNet path.
from train_UNET import (
    setup_ddp, cleanup_ddp, atomic_save, get_run_name,
    setup_experiment_dirs, save_used_config, load_latent_stats,
    reduce_mean_scalar, load_filenames, prepare_transform, build_file_list,
)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--train_config_path", type=str, required=True)
    parser.add_argument("--trained_diffusion_path", type=str, required=True,
                        help="Path to base UNet checkpoint (model.pt). Frozen.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cpus_per_task", type=int, default=8)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run_name to be specified.")
    args.run_name = get_run_name(args.run_name)

    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}

    for path_attr in ("dataset_config_path", "model_config_path"):
        for k, v in load_json(getattr(args, path_attr)).items():
            setattr(args, k, v)

    train_cfg = load_json(args.train_config_path)
    for k, v in train_cfg["controlnet_train"].items():
        setattr(args, k, v)

    for k, v in cli_overrides.items():
        setattr(args, k, v)
    return args

warnings.filterwarnings("ignore")


SHUTDOWN_REQUESTED = False
def graceful_shutdown(signum, frame):
    global SHUTDOWN_REQUESTED
    print(f"\n[!] Received signal {signum}. Requesting graceful shutdown...", flush=True)
    SHUTDOWN_REQUESTED = True
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


# ---------------------------------------------------------------------------
# Mask handling
# ---------------------------------------------------------------------------
def binarize_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Multi-channel binary encoding of a multi-class integer mask.

    Shape (B, 1, *spatial) → (B, num_classes, *spatial). One channel per
    non-background class (label 0 dropped). For BraTS-GLI 2023 the call
    site uses num_classes=4 (NCR/ED/ET against bg).
    """
    if labels.shape[1] != 1:
        raise ValueError(f"expected (B,1,*) mask, got {tuple(labels.shape)}")
    labels = labels.long().squeeze(1)
    one_hot = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
    # last dim is class; move to channel dim
    return one_hot.permute(0, -1, *range(1, labels.ndim)).contiguous()


def build_controlnet_file_list(filenames, embedding_base_dir, mask_dir,
                                mask_lookup: dict[str, str], include_body_region: bool):
    """Extend train_UNET.build_file_list with a 'label' key for the mask path."""
    files = []
    for fname in filenames:
        emb_path = os.path.join(embedding_base_dir, fname)
        if not os.path.exists(emb_path):
            continue
        info_path = emb_path + ".json"
        # Embedding file follows ``{subject_id}_emb.nii.gz``. Recover subject id
        # via the stem and look up the BraTS mask path the CSV builder emitted.
        subject_id = os.path.basename(fname).replace("_emb.nii.gz", "")
        if subject_id not in mask_lookup:
            continue
        mask_path = os.path.join(mask_dir, mask_lookup[subject_id])
        item = {"image": emb_path, "spacing": info_path, "cond": info_path,
                "label": mask_path}
        if include_body_region:
            item["top_region_index"] = info_path
            item["bottom_region_index"] = info_path
        files.append(item)
    return files


def prepare_controlnet_transform(include_body_region: bool = False) -> Compose:
    """train_UNET.prepare_transform + extra LoadImaged on the 'label' (mask)."""
    base = prepare_transform(include_body_region=include_body_region)
    # Compose chains the existing transforms; just append mask-loading ops.
    extra = [
        monai.transforms.LoadImaged(keys=["label"]),
        monai.transforms.EnsureChannelFirstd(keys=["label"]),
    ]
    return Compose(list(base.transforms) + extra)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_ddp()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    args = load_config()
    adapter = get_adapter(args.dataset_adapter)
    device = torch.device(f"cuda:{local_rank}")

    set_determinism(seed=args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    exp_paths = setup_experiment_dirs(args.output_dir, args.run_name)
    weights_dir = os.path.join(exp_paths["exp"], "weights", "controlnet")
    os.makedirs(weights_dir, exist_ok=True)
    embedding_base_dir = exp_paths["embeddings"]
    train_json = os.path.join(exp_paths["exp"], "train_files.json")
    valid_json = os.path.join(exp_paths["exp"], "valid_files.json")

    # Latent stats
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
        args.global_mean = float(global_mean.item())
        args.scale_factor = float(scale_factor.item())
        save_used_config(args, weights_dir)
        if args.report_to:
            wandb.init(project=getattr(args, "wandb_project_controlnet",
                                       getattr(args, "wandb_project_unet", "controlnet")),
                       entity=args.wandb_entity, config=args, name=args.run_name)

    # ------------------------------------------------------------------
    # Models — load trained UNet, build ControlNet
    # ------------------------------------------------------------------
    unet = define_instance(args, "diffusion_unet_def").to(device)
    if not args.trained_diffusion_path or not os.path.exists(args.trained_diffusion_path):
        raise ValueError(
            f"--trained_diffusion_path must point at a trained base UNet checkpoint "
            f"(got '{args.trained_diffusion_path}')."
        )
    diff_ckpt = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
    unet_state = diff_ckpt.get("unet", diff_ckpt.get("unet_state_dict"))
    unet.load_state_dict(unet_state, strict=False)
    if rank == 0:
        print(f"[unet] loaded from {args.trained_diffusion_path}", flush=True)
    for p in unet.parameters():
        p.requires_grad = False
    unet.eval()

    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    if rank == 0:
        print("[controlnet] initialized from UNet weights (copy_model_state)", flush=True)

    noise_scheduler = define_instance(args, "noise_scheduler")
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    num_train_timesteps = args.noise_scheduler["num_train_timesteps"]

    # ------------------------------------------------------------------
    # Data — embeddings (cond + mask via 'label' key)
    # ------------------------------------------------------------------
    filenames_train = load_filenames(train_json, "training", adapter)
    filenames_valid = load_filenames(valid_json, "validation", adapter)[:args.num_valid]

    # Build subject_id → rel_path_seg lookup from BraTS CSV.
    train_label_df = pd.read_csv(args.train_label_dir)
    valid_label_df = pd.read_csv(args.valid_label_dir)
    mask_lookup_train = dict(zip(train_label_df["eid"], train_label_df["rel_path_seg"]))
    mask_lookup_valid = dict(zip(valid_label_df["eid"], valid_label_df["rel_path_seg"]))

    train_files = build_controlnet_file_list(filenames_train, embedding_base_dir,
                                              args.data_dir, mask_lookup_train,
                                              include_body_region)
    valid_files = build_controlnet_file_list(filenames_valid, embedding_base_dir,
                                              args.data_dir, mask_lookup_valid,
                                              include_body_region)
    if rank == 0:
        print(f"Total training: {len(train_files)} valid: {len(valid_files)}", flush=True)

    data_transform = prepare_controlnet_transform(include_body_region=include_body_region)

    workers_per_gpu = args.cpus_per_task // world_size
    train_dataset = CacheDataset(data=train_files, transform=data_transform,
                                 cache_rate=args.cache_rate, num_workers=workers_per_gpu)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=workers_per_gpu, sampler=train_sampler,
                              pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=2)
    steps_per_epoch = max(len(train_loader), 1)

    val_total = len(valid_files)
    per_rank = (val_total + world_size - 1) // world_size
    val_shard = valid_files[rank * per_rank: min(val_total, (rank + 1) * per_rank)]
    val_dataset = CacheDataset(data=val_shard, transform=data_transform,
                               cache_rate=0.0, num_workers=workers_per_gpu)
    valid_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                              num_workers=workers_per_gpu, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    # ------------------------------------------------------------------
    # Optimizer / loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)
    total_opt_steps = max(1, (args.max_train_steps + args.gradient_accumulation_steps - 1)
                          // args.gradient_accumulation_steps)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_opt_steps, power=2.0)
    scaler = GradScaler("cuda") if args.amp else None

    # ControlNet num_classes for the mask. Default 4 (BraTS-GLI bg + 3 tumor classes).
    num_mask_classes = getattr(args, "num_mask_classes", 4)
    # Loss weighting (tumor amplification, NV-style).
    weighted_loss = getattr(args, "weighted_loss", 1.0)
    weighted_loss_labels = getattr(args, "weighted_loss_label", [3])  # ET by default
    if rank == 0 and weighted_loss > 1.0:
        print(f"[loss] weighted_loss={weighted_loss} on labels {weighted_loss_labels}",
              flush=True)

    # ------------------------------------------------------------------
    # (Optional) resume from controlnet checkpoint
    # ------------------------------------------------------------------
    start_step, best_val_loss = 0, float("inf")
    if args.resume:
        latest = sorted(
            [d for d in os.listdir(weights_dir) if d.startswith("checkpoint-")],
            key=lambda d: int(d.split("-", 1)[1]),
        )
        if latest:
            ckpt_path = os.path.join(weights_dir, latest[-1], "model.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                controlnet.load_state_dict(ckpt["controlnet"], strict=True)
                if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt: lr_scheduler.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and scaler is not None: scaler.load_state_dict(ckpt["scaler"])
                start_step = int(ckpt.get("step", 0))
                best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
                if rank == 0:
                    print(f"[resume] from {ckpt_path} step={start_step}", flush=True)

    dist.barrier(device_ids=[local_rank])
    controlnet = DDP(controlnet, device_ids=[local_rank], find_unused_parameters=True)

    def infinite_loader(loader, sampler, start_epoch=0):
        epoch = start_epoch
        while True:
            sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1

    train_iter = infinite_loader(train_loader, train_sampler, start_step // steps_per_epoch)
    progress_bar = trange(start_step, args.max_train_steps + 1,
                          desc=f"Training on Rank {rank}",
                          initial=start_step, total=args.max_train_steps + 1,
                          disable=(rank != 0))

    # ==================================================================
    # Training loop — frozen UNet + trainable ControlNet
    # ==================================================================
    for step in progress_bar:
        controlnet.train()
        batch = next(train_iter)
        images = (batch["image"].to(device, non_blocking=True).contiguous() - global_mean) * scale_factor
        labels = batch["label"].to(device, non_blocking=True)
        spacing_tensor = batch["spacing"].to(device, non_blocking=True)
        meta_tensor = batch["cond"].to(device, non_blocking=True)

        # Match mask spatial to latent. Down to {latent_shape} via nearest;
        # ControlNet's conditioning_embedding then handles the in-network
        # 1× passthrough → channel expansion.
        # TODO: if conditioning_embedding has non-trivial downsample
        # (currently in MAISI ControlNet it does), set size=latent_shape * ratio.
        labels_resized = F.interpolate(labels, size=images.shape[2:], mode="nearest")
        controlnet_cond = binarize_labels(labels_resized, num_classes=num_mask_classes)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            noise = torch.randn_like(images)
            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],),
                                           device=images.device).long()
            noisy_latent = noise_scheduler.add_noise(original_samples=images,
                                                      noise=noise, timesteps=timesteps)

            cn_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "controlnet_cond": controlnet_cond,
            }
            down_res, mid_res = controlnet(**cn_inputs)

            unet_inputs = {
                "x": noisy_latent,
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
                "meta_tensor": meta_tensor,
                "down_block_additional_residuals": down_res,
                "mid_block_additional_residual": mid_res,
            }
            if include_body_region:
                unet_inputs.update({
                    "top_region_index_tensor": batch["top_region_index"].to(device),
                    "bottom_region_index_tensor": batch["bottom_region_index"].to(device),
                })
            if include_modality:
                unet_inputs.update({
                    "class_labels": torch.ones((len(images),), dtype=torch.long, device=device)
                })

            with torch.no_grad():
                # UNet is frozen — but we still need its gradient flow into the
                # ControlNet outputs. PL/MAISI handles this naturally via the
                # additional_residuals path; we simply call .eval()-mode UNet.
                pass
            model_output = unet(**unet_inputs)

            if isinstance(noise_scheduler, RFlowScheduler):
                model_gt = images - noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                model_gt = noise_scheduler.get_velocity(images, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction_type {noise_scheduler.prediction_type}")

            if weighted_loss > 1.0:
                weights = torch.ones_like(images)
                roi = torch.zeros((images.shape[0], 1) + images.shape[2:], device=device)
                interp_label = F.interpolate(labels, size=images.shape[2:], mode="nearest")
                for cls in weighted_loss_labels:
                    roi[interp_label == cls] = 1
                weights[roi.repeat(1, images.shape[1], 1, 1, 1) == 1] = weighted_loss
                loss = (F.l1_loss(model_output.float(), model_gt.float(),
                                  reduction="none") * weights).mean()
            else:
                loss = F.l1_loss(model_output.float(), model_gt.float())

            loss = loss / args.gradient_accumulation_steps

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.amp:
                scaler.unscale_(optimizer)
                clip_grad_norm_(controlnet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(controlnet.parameters(), 1.0)
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_log = reduce_mean_scalar(loss) * args.gradient_accumulation_steps
        if rank == 0:
            progress_bar.set_postfix({"loss": f"{loss_log:.4f}"})
            if args.report_to and step % 100 == 0:
                wandb.log({"train/learning_rate": lr_scheduler.get_last_lr()[0],
                           "train/loss": loss_log}, step=step)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        did_validate = False
        if (step % args.validation_steps == 0 or step == args.max_train_steps) and step > start_step:
            did_validate = True
            controlnet.eval()
            val_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for vb in valid_loader:
                    v_images = (vb["image"].to(device, non_blocking=True) - global_mean) * scale_factor
                    v_labels = vb["label"].to(device, non_blocking=True)
                    v_spacing = vb["spacing"].to(device, non_blocking=True)
                    v_meta = vb["cond"].to(device, non_blocking=True)
                    v_labels_r = F.interpolate(v_labels, size=v_images.shape[2:], mode="nearest")
                    v_cond = binarize_labels(v_labels_r, num_classes=num_mask_classes)
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                        v_noise = torch.randn_like(v_images)
                        v_t = torch.randint(0, num_train_timesteps, (v_images.shape[0],),
                                              device=device).long()
                        v_noisy = noise_scheduler.add_noise(original_samples=v_images,
                                                            noise=v_noise, timesteps=v_t)
                        d_res, m_res = controlnet.module(x=v_noisy, timesteps=v_t,
                                                         controlnet_cond=v_cond)
                        v_out = unet(x=v_noisy, timesteps=v_t,
                                     spacing_tensor=v_spacing, meta_tensor=v_meta,
                                     down_block_additional_residuals=d_res,
                                     mid_block_additional_residual=m_res)
                        if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                            v_gt = v_noise
                        elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                            v_gt = v_images
                        else:
                            v_gt = v_images - v_noise
                        val_loss += F.l1_loss(v_out.float(), v_gt.float()).item()
                    n_batches += 1

            vm = torch.tensor([val_loss, n_batches], device=device)
            dist.all_reduce(vm, op=dist.ReduceOp.SUM)
            avg_val = vm[0].item() / max(vm[1].item(), 1)
            if rank == 0:
                print(f"Step {step} Val Loss: {avg_val:.4f}", flush=True)
                if args.report_to:
                    wandb.log({"valid/total_loss": avg_val}, step=step)
                if avg_val < best_val_loss:
                    best_val_loss = float(avg_val)
                    state = {
                        "controlnet": controlnet.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "scale_factor": float(scale_factor.item()),
                        "global_mean": float(global_mean.item()),
                    }
                    if scaler is not None:
                        state["scaler"] = scaler.state_dict()
                    best_dir = os.path.join(weights_dir, "best-checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    atomic_save(state, os.path.join(best_dir, "model.pt"))
                    print(f"[best] updated at step {step}: {best_val_loss:.6f}", flush=True)

            _best = torch.tensor([best_val_loss], device=device, dtype=torch.float32)
            dist.broadcast(_best, src=0)
            best_val_loss = float(_best.item())

        # ------------------------------------------------------------------
        # Periodic checkpoint
        # ------------------------------------------------------------------
        if ((step % args.checkpointing_steps == 0 and step > start_step) or SHUTDOWN_REQUESTED) and rank == 0:
            torch.cuda.synchronize(device)
            state = {
                "controlnet": controlnet.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "step": step,
                "best_val_loss": float(best_val_loss),
                "scale_factor": float(scale_factor.item()),
                "global_mean": float(global_mean.item()),
            }
            if scaler is not None:
                state["scaler"] = scaler.state_dict()
            ckpt_dir = os.path.join(weights_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            atomic_save(state, os.path.join(ckpt_dir, "model.pt"))
            print(f"\nSaved Step {step} checkpoint to {ckpt_dir}", flush=True)

        shutdown_tensor = torch.tensor([1 if (rank == 0 and SHUTDOWN_REQUESTED) else 0], device=device)
        dist.broadcast(shutdown_tensor, src=0)
        if shutdown_tensor.item() == 1:
            if rank == 0:
                print("Shutdown received — exiting training loop.")
            break
        if did_validate:
            dist.barrier(device_ids=[local_rank])

    if SHUTDOWN_REQUESTED:
        sys.exit(1)

    dist.barrier(device_ids=[local_rank])
    cleanup_ddp()


if __name__ == "__main__":
    main()
