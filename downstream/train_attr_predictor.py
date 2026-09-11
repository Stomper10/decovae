"""SFCN attribute classifier — train + eval entry point for adherence predictors.

Trains an SFCN classifier on REAL volumes to predict a categorical conditioning
attribute (sex; dx = AD/MCI/CN, or the 4-way healthy/MCI/AD/tumor). The resulting
``best.pt`` is consumed by ``scripts/adherence_eval.py`` to score whether the
diffusion model's generations honour their intended condition.

Structured like ``downstream/train_brain_age.py`` (torchrun DDP; single-GPU just
runs ``python downstream/train_attr_predictor.py ...``). The label_map (string →
class id) is passed as JSON, so the same script covers binary and multi-class
targets without a code change:

    --target sex --label_map '{"M":0,"F":1}'
    --target dx  --label_map '{"CN":0,"MCI":1,"AD":2}'
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from downstream.attr_dataset import make_attr_dataset
from downstream.sfcn import build_sfcn_classifier


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).long()
            logits = model(x)
            preds.append(logits.argmax(dim=1).detach().cpu())
            targets.append(y.detach().cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    acc = float((preds == targets).mean()) if len(targets) else 0.0
    # balanced accuracy (mean per-class recall) — robust to dx imbalance
    recalls = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            recalls.append(float((preds[mask] == c).mean()))
    bal_acc = float(np.mean(recalls)) if recalls else 0.0
    # confusion matrix (rows = true, cols = pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return {"acc": acc, "balanced_acc": bal_acc, "n": int(len(targets)),
            "confusion": cm.tolist()}


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

    label_map = json.loads(args.label_map)
    num_classes = len(set(label_map.values()))

    with open(args.dataset_config_path) as f:
        ds_cfg = json.load(f)
    resolution = tuple(args.resolution or ds_cfg.get("resolution", [160, 192, 160]))
    axcodes = ds_cfg.get("orientation_axcodes", "RAS")

    train_ds = make_attr_dataset(real_csv=args.train_csv, data_dir=args.data_dir,
                                 resolution=resolution, target_col=args.target,
                                 task="cls", label_map=label_map,
                                 real_limit=args.real_limit,
                                 orientation_axcodes=axcodes, cache_rate=args.cache_rate)
    val_ds = make_attr_dataset(real_csv=args.valid_csv, data_dir=eval_dir,
                               resolution=resolution, target_col=args.target,
                               task="cls", label_map=label_map,
                               orientation_axcodes=axcodes, cache_rate=args.cache_rate)

    # class weights from the (real) training label distribution → CE reweighting
    train_labels = [int(d["label"]) for d in train_ds.data]
    counts = Counter(train_labels)
    weights = torch.tensor(
        [len(train_labels) / (num_classes * max(counts.get(c, 0), 1))
         for c in range(num_classes)], dtype=torch.float32, device=device)

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
    # reported 17 epochs.
    resume = torch.load(last_fp, map_location="cpu", weights_only=False) if last_fp.is_file() else None

    model = build_sfcn_classifier(num_classes=num_classes, in_channels=1,
                                  dropout=args.dropout)
    if resume is not None:
        model.load_state_dict(resume["model"])
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights if args.class_weighted else None)
    start_epoch, best_metric = 0, -1.0
    if resume is not None:
        optim.load_state_dict(resume["optim"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch, best_metric = resume["epoch"] + 1, resume["best_metric"]

    if is_main:
        with open(out_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        if resume is None:
            open(out_dir / "history.jsonl", "w").close()
            with open(out_dir / "run_meta.json", "w") as f:
                json.dump({"seed": args.seed, "epochs": args.epochs, "target": args.target,
                           "train_csv": args.train_csv, "n_train": len(train_ds),
                           "valid_csv": args.valid_csv, "n_valid": len(val_ds),
                           "test_csv": args.test_csv, "train_dist": dict(counts),
                           "data_dir": args.data_dir, "eval_data_dir": eval_dir}, f, indent=2)
        print(f"[train] target={args.target} num_classes={num_classes} "
              f"label_map={label_map} train_dist={dict(counts)} seed={args.seed} "
              f"start_epoch={start_epoch}/{args.epochs}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).long()
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        metrics = evaluate(model.module if world_size > 1 else model, val_loader,
                           device, num_classes)
        if is_main:
            row = {"epoch": epoch, "train_loss": running / max(n, 1), **metrics}
            print(json.dumps(row), flush=True)
            with open(out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            # select on balanced accuracy (dx is imbalanced)
            if metrics["balanced_acc"] > best_metric:
                best_metric = metrics["balanced_acc"]
                torch.save({"model": (model.module if world_size > 1 else model).state_dict(),
                            "epoch": epoch, "target": args.target,
                            "label_map": label_map, "num_classes": num_classes,
                            "resolution": list(resolution),
                            "val_balanced_acc": metrics["balanced_acc"],
                            "val_acc": metrics["acc"]},
                           ckpt_dir / "best.pt")
            torch.save({"model": (model.module if world_size > 1 else model).state_dict(),
                        "optim": optim.state_dict(), "scheduler": scheduler.state_dict(),
                        "epoch": epoch, "best_metric": best_metric, "seed": args.seed}, last_fp)

    # ---- final: score the SELECTED checkpoint on the held-out test set ----
    if args.test_csv and is_main:
        import copy
        test_ds = make_attr_dataset(real_csv=args.test_csv, data_dir=eval_dir,
                                    resolution=resolution, target_col=args.target,
                                    task="cls", label_map=label_map,
                                    orientation_axcodes=axcodes,
                                    cache_rate=args.cache_rate)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        eval_model = copy.deepcopy(model.module if world_size > 1 else model)
        eval_model.load_state_dict(best["model"])
        eval_model.eval().to(device)
        tm = evaluate(eval_model, test_loader, device, num_classes)
        row = {"split": "test", "from_epoch": best["epoch"], "seed": args.seed, **tm}
        print(json.dumps(row), flush=True)
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(row, f, indent=2)

    _cleanup_ddp()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_config_path", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--valid_csv", required=True)
    p.add_argument("--test_csv", default=None,
                   help="Held-out test set. Selection stays on valid; this is scored "
                        "ONCE at the end with the selected best.pt. Plan section 6 "
                        "requires the downstream numbers to come from a held-out real "
                        "test because generation is conditioned on the same attribute "
                        "the predictor reads back -- scoring on the split that chose "
                        "the checkpoint would close that loop.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--eval_data_dir", default=None,
                   help="Root for --valid_csv/--test_csv; defaults to --data_dir. Set it when "
                        "training on synthetic volumes, which live under a different root.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)

    p.add_argument("--target", required=True, help="metadata column to predict (e.g. sex, dx).")
    p.add_argument("--label_map", required=True,
                   help='JSON string mapping category -> class id, e.g. \'{"M":0,"F":1}\'.')
    p.add_argument("--class_weighted", action="store_true",
                   help="reweight CrossEntropy by inverse class frequency (recommended for dx).")
    p.add_argument("--real_limit", type=int, default=None)

    p.add_argument("--resolution", type=int, nargs=3, default=None)
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
