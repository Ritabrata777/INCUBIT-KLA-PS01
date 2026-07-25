"""Quick smoke test for the self-training pipeline.

Generates a tiny folder of synthetic clean grayscale images, runs a handful of
training steps end-to-end (degradation -> NAFNet-SR -> Charbonnier -> AdamW),
and prints per-step loss plus images/second throughput.

Usage:
    python smoke_test.py                 # default: 8 imgs, 10 steps, bs=4, ps=128
    python smoke_test.py --steps 20 --batch_size 8 --patch_size 256
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import SelfTrainingDataset, seed_worker
from model import NAFNet_SR
from train import CharbonnierLoss


def make_sample_dir(root: Path, n: int, size: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        # Smooth low-freq pattern + a few bright "structures" -> semiconductor-ish.
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        base = 0.4 + 0.3 * np.sin(x / 17.0 + i) * np.cos(y / 23.0 - i)
        for _ in range(6):
            cx, cy = rng.integers(0, size, size=2)
            r = rng.integers(4, 12)
            mask = (x - cx) ** 2 + (y - cy) ** 2 < r * r
            base[mask] = 0.95
        arr = np.clip(base + 0.02 * rng.standard_normal(base.shape), 0, 1)
        Image.fromarray((arr * 255).astype(np.uint8), "L").save(root / f"img_{i:03d}.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2,
                   help="Steps excluded from throughput timing.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--sample_dir", type=str, default="",
                   help="Optional path to keep the sample folder. Temp dir by default.")
    p.add_argument("--image_backend", choices=["pil", "torchvision", "cv2"],
                   default="pil", help="Dataset image decoder backend.")
    p.add_argument("--cache_mode", choices=["none", "memory", "disk"],
                   default="none", help="Cache decoded clean images.")
    p.add_argument("--cache_dir", type=str, default=None)
    return p.parse_args()



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    tmp_ctx = None
    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
        cleanup = False
    else:
        tmp_ctx = tempfile.mkdtemp(prefix="smoke_clean_")
        sample_dir = Path(tmp_ctx)
        cleanup = True

    print(f"[setup] device={device} sample_dir={sample_dir}")
    make_sample_dir(sample_dir, args.num_images, args.image_size)

    ds = SelfTrainingDataset(
        clean_dir=str(sample_dir),
        self_train=True,
        patch_size=args.patch_size,
        augment=True,
        image_backend=args.image_backend,
        cache_mode=args.cache_mode,
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=torch.Generator().manual_seed(0),
        drop_last=True,
    )

    model = NAFNet_SR(in_channels=1, out_channels=1, width=args.width, upscale=1).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] NAFNet_SR width={args.width} params={n_params:.2f}M")

    optim = AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.9), weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = CharbonnierLoss()

    it = iter(loader)
    imgs_seen = 0
    t_start = None
    losses = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(degraded)
            loss = criterion(pred, clean)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()

        losses.append(loss.item())
        print(f"[step {step:3d}/{args.steps}] loss={loss.item():.4f}")

        if step == args.warmup:
            t_start = time.time()
            imgs_seen = 0
        elif step > args.warmup:
            imgs_seen += degraded.size(0)

    dt = (time.time() - t_start) if t_start is not None else 0.0
    thr = imgs_seen / dt if dt > 0 else float("nan")
    print("-" * 60)
    print(f"[summary] steps={args.steps} warmup={args.warmup} "
          f"batch_size={args.batch_size} patch={args.patch_size}")
    print(f"[summary] mean_loss={sum(losses)/len(losses):.4f} "
          f"first={losses[0]:.4f} last={losses[-1]:.4f}")
    print(f"[throughput] {imgs_seen} imgs in {dt:.2f}s -> {thr:.2f} img/s "
          f"({thr / max(args.batch_size,1):.2f} steps/s)")
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[gpu] peak_mem={mem:.2f} GiB")

    if cleanup:
        shutil.rmtree(tmp_ctx, ignore_errors=True)


if __name__ == "__main__":
    main()
