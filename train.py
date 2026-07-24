"""
train.py
Training script for NAFNet-SR semiconductor image restoration.

Self-training / on-the-fly synthetic degradation (recommended):
    python train.py --self_train --clean_dir data/clean \
        --val_clean_dir data/val_clean

Static pre-paired degraded/clean data:
    python train.py --clean_dir data/clean --degraded_dir data/degraded \
        --val_clean_dir data/val_clean --val_degraded_dir data/val_degraded
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SelfTrainingDataset
from degradation import SemiconductorDegradationPipeline
from model import NAFNet_SR


class CharbonnierLoss(nn.Module):
    """Robust L1-like loss: sqrt((pred - target)^2 + eps^2). Less sensitive
    to outliers than MSE while remaining smooth/differentiable everywhere --
    well suited to targets synthesized with unclamped, out-of-bounds noise."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train NAFNet-SR for semiconductor image restoration")

    parser.add_argument("--clean_dir", type=str, required=True, help="Directory of clean training images")
    parser.add_argument("--degraded_dir", type=str, default=None, help="Directory of pre-degraded training images (static mode)")
    parser.add_argument("--val_clean_dir", type=str, default=None, help="Directory of clean validation images")
    parser.add_argument("--val_degraded_dir", type=str, default=None, help="Directory of pre-degraded validation images (static mode)")

    parser.add_argument("--self_train", action="store_true", help="Enable on-the-fly dynamic synthetic degradation during training")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Use mixed-precision training (default: on)")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable mixed-precision training")
    parser.add_argument("--output_path", type=str, default="final_model_weights.pt")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_dataloaders(args):
    pipeline = SemiconductorDegradationPipeline()

    train_dataset = SelfTrainingDataset(
        clean_dir=args.clean_dir,
        degraded_dir=args.degraded_dir,
        self_train=args.self_train,
        patch_size=args.patch_size,
        upscale_factor=args.upscale_factor,
        degradation_pipeline=pipeline,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if args.val_clean_dir is not None:
        val_dataset = SelfTrainingDataset(
            clean_dir=args.val_clean_dir,
            degraded_dir=args.val_degraded_dir,
            self_train=args.self_train,
            patch_size=args.patch_size,
            upscale_factor=args.upscale_factor,
            degradation_pipeline=pipeline,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader


@torch.no_grad()
def validate(model, val_loader, criterion, device, amp):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in val_loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            pred = model(degraded)
            loss = criterion(pred, clean)

        total_loss += loss.item() * degraded.size(0)
        total_samples += degraded.size(0)

    model.train()
    return total_loss / max(total_samples, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(args)
    print(f"Train samples: {len(train_loader.dataset)} | "
          f"Val samples: {len(val_loader.dataset) if val_loader else 0}")
    print(f"Self-training (on-the-fly degradation): {args.self_train}")

    model = NAFNet_SR(
        in_channels=1,
        out_channels=1,
        width=args.width,
        num_blocks=args.num_blocks,
        upscale_factor=args.upscale_factor,
    ).to(device)

    criterion = CharbonnierLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.9)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_train_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(degraded)
                loss = criterion(pred, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

            if (step + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")

        scheduler.step()
        avg_train_loss = running_loss / max(num_batches, 1)
        elapsed = time.time() - epoch_start

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device, use_amp)
            print(f"Epoch [{epoch}/{args.epochs}] avg_train_loss={avg_train_loss:.6f} "
                  f"val_loss={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.output_path)
                print(f"  -> New best val_loss {best_val_loss:.6f}, saved weights to {args.output_path}")
        else:
            print(f"Epoch [{epoch}/{args.epochs}] avg_train_loss={avg_train_loss:.6f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}s")

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), args.output_path)
                print(f"  -> New best train_loss {best_train_loss:.6f}, saved weights to {args.output_path}")

    print(f"Training complete. Best weights saved to {args.output_path}")


if __name__ == "__main__":
    main()
