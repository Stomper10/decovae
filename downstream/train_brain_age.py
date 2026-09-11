"""SFCN brain-age regressor — train + eval entry point.

Run real-only baseline (default), then later re-run with
``--synthetic_csv`` / ``--synthetic_dir`` once stage2 weights produce
synthetic volumes via ``scripts/generate_synthetic_for_age.py``.

The script is structured for torchrun-style DDP. For single-GPU sanity
just invoke ``python downstream/train_brain_age.py ...`` directly.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from downstream.brain_age_dataset import make_dataset
from downstream.sfcn import build_sfcn


def _setup_ddp() -> tuple[int, int, int]:
    """Initialize torch.distributed if launched via torchrun. Returns (rank, world_size, local_rank)."""
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["age"].to(device, non_blocking=True).float()
            yhat = model(x)
            preds.append(yhat.detach().cpu())
            targets.append(y.detach().cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = float(np.sum((preds - targets) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(targets))}


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = _setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    _seed_everything(args.seed)
    # Validation / test are always REAL; training may be synthetic, which lives under
    # a different root. One --data_dir could not serve both.
    eval_dir = args.eval_data_dir or args.data_dir

    with open(args.dataset_config_path) as f:
        ds_cfg = json.load(f)
    resolution = tuple(args.resolution or ds_cfg.get("resolution", [160, 192, 160]))

    train_ds = make_dataset(real_csv=args.train_csv,
                            data_dir=args.data_dir,
                            resolution=resolution,
                            synthetic_csv=args.synthetic_csv,
                            synthetic_dir=args.synthetic_dir,
                            real_limit=args.real_limit,
                            synth_limit=args.synth_limit,
                            orientation_axcodes=ds_cfg.get("orientation_axcodes", "RAS"),
                            cache_rate=args.cache_rate)
    val_ds = make_dataset(real_csv=args.valid_csv,
                          data_dir=eval_dir,
                          resolution=resolution,
                          orientation_axcodes=ds_cfg.get("orientation_axcodes", "RAS"),
                          cache_rate=args.cache_rate)

    train_sampler = DistributedSampler(train_ds, seed=args.seed) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    out_dir = Path(args.output_dir) / args.run_name
    ckpt_dir = out_dir / "weights"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_fp = ckpt_dir / "last.pt"
    # Resume from the last completed epoch. Without this a walltime cut or a SIGUSR1
    # requeue restarted at epoch 0 and APPENDED to history.jsonl -- how one TSTR run
    # reported 17 epochs and another stopped at 13.
    resume = torch.load(last_fp, map_location="cpu", weights_only=False) if last_fp.is_file() else None

    model = build_sfcn(in_channels=1, dropout=args.dropout)
    if resume is not None:
        model.load_state_dict(resume["model"])
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.L1Loss()  # MAE; replace with MSE / soft-label-KL if pursuing original SFCN later
    start_epoch, best_mae = 0, float("inf")
    if resume is not None:
        optim.load_state_dict(resume["optim"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch, best_mae = resume["epoch"] + 1, resume["best_mae"]

    if is_main:
        history: list[dict] = []
        if resume is None:
            open(out_dir / "history.jsonl", "w").close()
            with open(out_dir / "run_meta.json", "w") as f:
                json.dump({"seed": args.seed, "epochs": args.epochs,
                           "train_csv": args.train_csv, "n_train": len(train_ds),
                           "valid_csv": args.valid_csv, "n_valid": len(val_ds),
                           "test_csv": args.test_csv,
                           "data_dir": args.data_dir, "eval_data_dir": eval_dir}, f, indent=2)
        print(f"[run] seed={args.seed} n_train={len(train_ds)} n_valid={len(val_ds)} "
              f"start_epoch={start_epoch}/{args.epochs}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["age"].to(device, non_blocking=True).float()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        if (epoch + 1) % args.val_every and epoch + 1 != args.epochs:
            # Not a validation epoch. At small n an epoch is a handful of steps, and
            # validating every one of hundreds of epochs costs more than the training.
            # last.pt is written at validation epochs, so a requeue loses at most
            # val_every epochs.
            continue
        metrics = evaluate(model.module if world_size > 1 else model, val_loader, device)
        if is_main:
            train_mae = running / max(n, 1)
            row = {"epoch": epoch, "train_mae": train_mae, **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            with open(out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                torch.save({"model": (model.module if world_size > 1 else model).state_dict(),
                            "epoch": epoch, "val_mae": metrics["mae"]},
                           ckpt_dir / "best.pt")
            torch.save({"model": (model.module if world_size > 1 else model).state_dict(),
                        "optim": optim.state_dict(), "scheduler": scheduler.state_dict(),
                        "epoch": epoch, "best_mae": best_mae, "seed": args.seed}, last_fp)

    # ---- final: score the SELECTED checkpoint on the held-out test set ----
    # Selection uses validation, so reporting validation would report the maximum
    # over epochs. Test is read once, here.
    if args.test_csv and is_main:
        import copy
        test_ds = make_dataset(real_csv=args.test_csv, data_dir=eval_dir,
                               resolution=resolution,
                               orientation_axcodes=ds_cfg.get("orientation_axcodes", "RAS"),
                               cache_rate=args.cache_rate)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        eval_model = copy.deepcopy(model.module if world_size > 1 else model)
        eval_model.load_state_dict(best["model"])
        eval_model.eval().to(device)
        row = {"split": "test", "from_epoch": best["epoch"], "seed": args.seed,
               **evaluate(eval_model, test_loader, device)}
        print(json.dumps(row), flush=True)
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(row, f, indent=2)

    _cleanup_ddp()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_config_path", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--valid_csv", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--eval_data_dir", default=None,
                   help="Root for --valid_csv/--test_csv; defaults to --data_dir. Set it when "
                        "training on synthetic volumes, which live under a different root.")
    p.add_argument("--test_csv", default=None,
                   help="Held-out real test set, scored once at the end with best.pt.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_every", type=int, default=1,
                   help="Validate (and checkpoint) every N epochs; the last epoch always validates.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)

    p.add_argument("--synthetic_csv", default=None,
                   help="CSV of synthetic (rel_path, age) generated by VAE+UNet.")
    p.add_argument("--synthetic_dir", default=None)
    p.add_argument("--real_limit", type=int, default=None,
                   help="Cap on real samples for data-regime ablations (e.g. 100/500/1k/5k).")
    p.add_argument("--synth_limit", type=int, default=None)

    p.add_argument("--resolution", type=int, nargs=3, default=None,
                   help="Override dataset.json resolution.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_rate", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
