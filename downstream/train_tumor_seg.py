"""SegResNet tumor segmentor — train + eval entry point.

Mirrors ``train_brain_age.py``: real-only baseline by default, with
``--synthetic_csv`` / ``--synthetic_dir`` flags to mix mask-conditional
synthetic volumes once stage2 + UNET produce them via
``scripts/generate_synthetic_for_seg.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from downstream.seg_dataset import make_dataset


REGION_NAMES = ("TC", "WT", "ET")  # tumor core, whole tumor, enhancing tumor


def _setup_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_segresnet(in_channels: int = 1, num_classes: int = 3) -> SegResNet:
    return SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=in_channels,
        out_channels=num_classes,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        norm="instance",
        dropout_prob=0.2,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             roi_size: tuple[int, int, int]) -> dict:
    model.eval()
    dice = DiceMetric(include_background=True, reduction="mean_batch")
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = sliding_window_inference(x, roi_size, sw_batch_size=2, predictor=model,
                                          overlap=0.5, mode="gaussian")
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        dice(pred, y)
    per_region = dice.aggregate().cpu().numpy().tolist()
    dice.reset()
    return {name: float(v) for name, v in zip(REGION_NAMES, per_region)} | {
        "dice_mean": float(np.mean(per_region))
    }


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = _setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    with open(args.dataset_config_path) as f:
        ds_cfg = json.load(f)
    # resolution comes from the SEG config, not the dataset config. The dataset
    # config's 240x240x155 is the native BraTS grid, which SegResNet cannot take:
    # 155 is not a multiple of 8, so the third decoder skip-add is 40 against 39.
    # tumor_seg.json says 240x240x144 for exactly that reason, and before this it
    # was read for epochs/lr only, so the crash was guaranteed.
    seg_cfg = {}
    if args.seg_config_path:
        with open(args.seg_config_path) as f:
            seg_cfg = json.load(f)
    resolution = tuple(args.resolution or seg_cfg.get("resolution")
                       or ds_cfg.get("resolution", [240, 240, 155]))
    roi_size = tuple(args.roi_size or seg_cfg.get("roi_size", [128, 128, 128]))
    if is_main:
        print(f"  resolution    : {list(resolution)}", flush=True)
        print(f"  roi_size      : {list(roi_size)}  (train crop = val window)", flush=True)

    train_ds = make_dataset(real_csv=args.train_csv, data_dir=args.data_dir,
                            resolution=resolution, roi_size=roi_size, train=True,
                            synthetic_csv=args.synthetic_csv,
                            synthetic_dir=args.synthetic_dir,
                            real_limit=args.real_limit,
                            synth_limit=args.synth_limit,
                            cache_rate=args.cache_rate)
    val_ds = make_dataset(real_csv=args.valid_csv, data_dir=args.data_dir,
                          resolution=resolution, cache_rate=args.cache_rate)

    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_segresnet(in_channels=1, num_classes=len(REGION_NAMES)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = DiceCELoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)

    if is_main:
        out_dir = Path(args.output_dir) / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / "weights"
        ckpt_dir.mkdir(exist_ok=True)
        best_dice = -1.0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        # Validation is decoupled from the epoch count because the data-scaling
        # ablation has to hold OPTIMIZER STEPS fixed, not epochs: steps/epoch is
        # n/(bs*gpus), so 200 epochs is 25,000 steps at n=1000 but only 2,400 at
        # n=100, and a curve run that way measures undertraining as if it were data
        # scarcity. Matching steps means ~1,250 epochs at n=100, and validating every
        # one of those (126 volumes x 18 sliding windows, redundantly on all ranks)
        # would cost far more than the training it is meant to monitor.
        last_epoch = epoch == args.epochs - 1
        if not (last_epoch or (epoch + 1) % args.val_every == 0):
            continue
        metrics = evaluate(model.module if world_size > 1 else model, val_loader, device, roi_size)
        if is_main:
            row = {"epoch": epoch, "step": (epoch + 1) * len(train_loader),
                   "train_loss": running / max(n, 1), **metrics}
            print(json.dumps(row), flush=True)
            with open(out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if metrics["dice_mean"] > best_dice:
                best_dice = metrics["dice_mean"]
                torch.save({"model": (model.module if world_size > 1 else model).state_dict(),
                            "epoch": epoch, "val_dice_mean": metrics["dice_mean"]},
                           ckpt_dir / "best.pt")

    _cleanup_ddp()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_config_path", required=True)
    p.add_argument("--seg_config_path", default=None,
                   help="tumor_seg.json — source of resolution and roi_size.")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--valid_csv", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)

    p.add_argument("--synthetic_csv", default=None,
                   help="CSV of synthetic (rel_path, rel_path_seg) from mask-conditional VAE+UNet.")
    p.add_argument("--synthetic_dir", default=None)
    p.add_argument("--real_limit", type=int, default=None)
    p.add_argument("--synth_limit", type=int, default=None)

    p.add_argument("--resolution", type=int, nargs=3, default=None)
    p.add_argument("--roi_size", type=int, nargs=3, default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_rate", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--val_every", type=int, default=1,
                   help="Validate every N epochs (the last epoch always runs).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
