"""Training loop for NAFNet-SR semiconductor image restoration.

Example:
    python train.py --clean_dir data/clean --self_train --epochs 200 \
        --batch_size 16 --patch_size 256 --lr 2e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path


import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from dataset import SelfTrainingDataset, seed_worker
from model import NAFNet_SR


def set_reproducibility(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    """Seed python/numpy/torch and configure cuDNN for reproducibility.

    When ``deterministic`` is True, forces bit-exact reproducibility on CUDA
    by disabling cuDNN autotune, enabling deterministic algorithms, and
    setting the cuBLAS workspace config required by PyTorch. This is slower
    and may raise on ops without a deterministic implementation.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs a fixed workspace to be deterministic under CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[repro] use_deterministic_algorithms failed: {e}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)



class CharbonnierLoss(nn.Module):
    """L1-like loss with a small epsilon for smoothness."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g.view(1, 1, -1)


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5) -> float:
    """Mean SSIM over a batch of single-channel images in [0, 1]."""
    pred = pred.clamp(0, 1).float()
    target = target.clamp(0, 1).float()
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    device, dtype = pred.device, pred.dtype
    win1d = _gaussian_window(window_size, sigma, device, dtype)
    win_x = win1d.unsqueeze(2)          # (1,1,1,K)
    win_y = win1d.unsqueeze(3)          # (1,1,K,1)
    pad = window_size // 2

    def _blur(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.conv2d(x, win_y, padding=(pad, 0))
        x = torch.nn.functional.conv2d(x, win_x, padding=(0, pad))
        return x

    mu_p = _blur(pred)
    mu_t = _blur(target)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sigma_p2 = _blur(pred * pred) - mu_p2
    sigma_t2 = _blur(target * target) - mu_t2
    sigma_pt = _blur(pred * target) - mu_pt
    num = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)
    return (num / den).mean().item()


class CUDAPrefetcher:
    """Async H2D prefetcher that overlaps data loading with GPU compute.

    Uses a dedicated CUDA stream to copy the next batch to GPU while the
    current batch trains on the default stream. Falls back to a plain
    iterator on CPU-only devices.
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.enabled = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None
        self._iter = None
        self._next = None

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _to_device(self, batch):
        if not self.enabled:
            return batch
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _preload(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        if self.enabled:
            with torch.cuda.stream(self.stream):
                self._next = self._to_device(batch)
        else:
            self._next = batch

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self.enabled:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next
        # Ensure tensors aren't freed while consumer uses them.
        if self.enabled:
            for v in batch.values():
                if torch.is_tensor(v):
                    v.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch



def _percentile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round((q / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def make_tb_writer(args, subdir: str = ""):
    """Create a SummaryWriter under --tb_log_dir (optionally a sub-run dir).

    Returns None when TensorBoard logging is disabled or unavailable, so all
    call sites can stay `if writer is not None:`.
    """
    if not getattr(args, "tb_log_dir", None):
        return None
    log_dir = str(Path(args.tb_log_dir) / subdir) if subdir else args.tb_log_dir
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir,
                               flush_secs=getattr(args, "tb_flush_secs", 30))
        writer.add_text("args", "  \n".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
        print(f"[tb] logging to {log_dir}")
        return writer
    except Exception as e:  # pragma: no cover
        print(f"[tb] disabled ({e}); pip install tensorboard to enable")
        return None


def _tb_log_pipeline(writer, args, prefix: str, *, data_ms, compute_ms, step_ms,
                     throughput: float, extra: dict | None = None) -> None:
    """Log dataloader/compute/overlap/throughput series + summary + hparams.

    Per-step series go to `<prefix>/step/*` (step index as x-axis) so a single
    run can be inspected, while the scalar summaries and `add_hparams` entry
    make several runs directly comparable in the HPARAMS tab.
    """
    if writer is None:
        return
    for i, (d, c, s) in enumerate(zip(data_ms, compute_ms, step_ms)):
        writer.add_scalar(f"{prefix}/step/data_ms", d, i)
        writer.add_scalar(f"{prefix}/step/compute_ms", c, i)
        writer.add_scalar(f"{prefix}/step/step_ms", s, i)

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    d_mean, c_mean, s_mean = _mean(data_ms), _mean(compute_ms), _mean(step_ms)
    overlap_gain = max(0.0, d_mean + c_mean - s_mean)
    denom = min(d_mean, c_mean)
    overlap_eff = (overlap_gain / denom) if denom > 0 else 0.0
    ceiling = (args.batch_size / (max(d_mean, c_mean) / 1000.0)
               if max(d_mean, c_mean) > 0 else 0.0)
    d_s, c_s, s_s = sorted(data_ms), sorted(compute_ms), sorted(step_ms)

    metrics = {
        f"{prefix}/data_ms_mean": d_mean,
        f"{prefix}/data_ms_p95": _percentile(d_s, 95),
        f"{prefix}/compute_ms_mean": c_mean,
        f"{prefix}/compute_ms_p95": _percentile(c_s, 95),
        f"{prefix}/step_ms_mean": s_mean,
        f"{prefix}/step_ms_p95": _percentile(s_s, 95),
        f"{prefix}/overlap_gain_ms": overlap_gain,
        f"{prefix}/overlap_efficiency": overlap_eff,
        f"{prefix}/throughput_img_per_s": throughput,
        f"{prefix}/ceiling_img_per_s": ceiling,
        f"{prefix}/data_compute_ratio": (d_mean / c_mean) if c_mean > 0 else 0.0,
    }
    if extra:
        metrics.update({f"{prefix}/{k}": float(v) for k, v in extra.items()})
    for k, v in metrics.items():
        writer.add_scalar(k, v, 0)

    hparams = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": int(bool(args.persistent_workers)),
        "pin_memory": int(bool(args.pin_memory)),
        "image_backend": args.image_backend,
        "cache_mode": args.cache_mode,
        "width": args.width,
        "upscale": args.upscale,
        "patch_size": getattr(args, "patch_size", 0),
        "grad_accum_steps": getattr(args, "grad_accum_steps", 1),
    }
    try:
        writer.add_hparams(hparams, {k: float(v) for k, v in metrics.items()})
    except Exception as e:  # pragma: no cover
        print(f"[tb] add_hparams skipped ({e})")
    writer.flush()



def benchmark_dataloader(loader: DataLoader, device: torch.device, args) -> None:
    """Run a self-training-shaped loop and measure input-pipeline health.

    Reports per-step (ms):
      - data_wait: blocking time consumer spent waiting for the next batch
                   from the CUDAPrefetcher (near-zero => queue kept full).
      - compute:   forward + backward + optimizer + AMP scaler.
      - step:      end-to-end wall time = data_wait + compute.

    Also reports:
      - starvation ratio: fraction of steps where data_wait exceeded 5% of
        step time (i.e. the loader could not hide behind compute).
      - queue depth: nominal `num_workers * prefetch_factor` slots, plus a
        measured saturation drain (batches obtainable in <1 ms after a
        compute-heavy burst) — an empirical lower bound on how many batches
        the workers keep hot.
    """
    print(f"[bench] warmup={args.benchmark_warmup} steps={args.benchmark_steps} "
          f"batch_size={args.batch_size} workers={args.num_workers} "
          f"prefetch_factor={args.prefetch_factor} pin_memory={args.pin_memory}")

    # Build a tiny throwaway model matching the real training config so the
    # compute cost is realistic (this determines whether workers can hide).
    model = NAFNet_SR(in_channels=1, out_channels=1,
                      width=args.width, upscale=args.upscale).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                      betas=(0.9, 0.9))
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = CharbonnierLoss()

    prefetcher = CUDAPrefetcher(loader, device)
    it = iter(prefetcher)

    data_ms, compute_ms, step_ms = [], [], []
    total_needed = args.benchmark_warmup + args.benchmark_steps
    imgs_measured = 0
    t_bench_start = None

    for i in range(total_needed):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_step_start = time.perf_counter()

        t_wait_start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(prefetcher)
            batch = next(it)
        if device.type == "cuda":
            # Wait time includes the H2D copy the consumer syncs on.
            torch.cuda.synchronize(device)
        t_wait_end = time.perf_counter()

        degraded = batch["degraded"]
        clean = batch["clean"]

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(degraded)
            loss = criterion(pred, clean)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_step_end = time.perf_counter()

        if i >= args.benchmark_warmup:
            if t_bench_start is None:
                t_bench_start = t_step_start
            data_ms.append((t_wait_end - t_wait_start) * 1000.0)
            step_ms.append((t_step_end - t_step_start) * 1000.0)
            compute_ms.append(step_ms[-1] - data_ms[-1])
            imgs_measured += degraded.size(0)

    # ---- saturation probe: drain as many pre-ready batches as possible ----
    drain_ready = 0
    for _ in range(64):
        t0 = time.perf_counter()
        try:
            _ = next(it)
        except StopIteration:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if (time.perf_counter() - t0) * 1000.0 < 1.0:
            drain_ready += 1
        else:
            break

    def _summ(name, xs):
        xs_s = sorted(xs)
        return (f"  {name:<10s} mean={sum(xs)/len(xs):7.2f}  p50={_percentile(xs_s, 50):7.2f}  "
                f"p95={_percentile(xs_s, 95):7.2f}  max={xs_s[-1]:7.2f}  (ms)")

    total_wall = (time.perf_counter() - t_bench_start) if t_bench_start else 0.0
    throughput = imgs_measured / total_wall if total_wall > 0 else 0.0
    starved = sum(1 for d, s in zip(data_ms, step_ms) if s > 0 and d > 0.05 * s)
    starvation_ratio = starved / max(1, len(data_ms))
    nominal_queue = max(0, args.num_workers) * max(1, args.prefetch_factor)

    print("[bench] per-step timing:")
    print(_summ("data_wait", data_ms))
    print(_summ("compute",   compute_ms))
    print(_summ("step",      step_ms))
    print(f"[bench] throughput           = {throughput:8.1f} img/s "
          f"({imgs_measured} imgs in {total_wall:.2f}s)")
    print(f"[bench] starvation ratio    = {starvation_ratio*100:6.1f}%  "
          f"(steps where data_wait > 5% of step time)")
    print(f"[bench] queue depth nominal = {nominal_queue}  "
          f"(num_workers * prefetch_factor)")
    print(f"[bench] queue depth measured= {drain_ready}  "
          f"(batches ready in <1 ms after compute burst)")
    if device.type == "cuda":
        print(f"[bench] peak GPU mem        = "
              f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")

    writer = make_tb_writer(args, "bench")
    if writer is not None:
        _tb_log_pipeline(writer, args, "bench",
                         data_ms=data_ms, compute_ms=compute_ms, step_ms=step_ms,
                         throughput=throughput,
                         extra={"starvation_ratio": starvation_ratio,
                                "queue_depth_nominal": nominal_queue,
                                "queue_depth_measured": drain_ready,
                                "peak_gpu_mem_gib": (
                                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                                    if device.type == "cuda" else 0.0)})
        writer.close()



def profile_bottleneck(loader: DataLoader, device: torch.device, args) -> None:
    """Measure dataloader vs GPU-compute time and quantify prefetch overlap.

    Runs three regimes over the same batch shape:
      A. data-only:     iterate the DataLoader + H2D copy, no compute.
      B. compute-only:  fixed cached batch, forward+backward+optimizer.
      C. overlapped:    real training step through CUDAPrefetcher.

    Then reports:
      - t_data, t_compute, t_step (mean/p50/p95, ms)
      - overlap gain           = t_data + t_compute - t_step   (ms hidden)
      - overlap efficiency     = gain / min(t_data, t_compute) (0..1, higher is better)
      - bottleneck             = data-bound / compute-bound / balanced
      - achievable ceiling     = images / max(t_data, t_compute)   (perfect overlap)
      - observed throughput    = images / t_step
    """
    warmup = args.profile_bottleneck_warmup
    steps = args.profile_bottleneck_steps
    print(f"[profile] warmup={warmup} steps={steps} batch_size={args.batch_size} "
          f"workers={args.num_workers} prefetch_factor={args.prefetch_factor} "
          f"pin_memory={args.pin_memory}")

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # ---- A. data-only ---------------------------------------------------
    prefetcher = CUDAPrefetcher(loader, device)
    it = iter(prefetcher)
    data_ms = []
    sample_batch = None
    for i in range(warmup + steps):
        _sync()
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(prefetcher)
            batch = next(it)
        _sync()
        if i >= warmup:
            data_ms.append((time.perf_counter() - t0) * 1000.0)
        sample_batch = batch  # keep last for compute-only stage

    # ---- B. compute-only (fixed cached batch, no data pressure) ---------
    model = NAFNet_SR(in_channels=1, out_channels=1,
                      width=args.width, upscale=args.upscale).to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.9))
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = CharbonnierLoss()
    degraded = sample_batch["degraded"].detach().clone()
    clean = sample_batch["clean"].detach().clone()

    compute_ms = []
    for i in range(warmup + steps):
        _sync()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(degraded)
            loss = criterion(pred, clean)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _sync()
        if i >= warmup:
            compute_ms.append((time.perf_counter() - t0) * 1000.0)

    # ---- C. overlapped (real training step through CUDAPrefetcher) ------
    prefetcher = CUDAPrefetcher(loader, device)
    it = iter(prefetcher)
    step_ms = []
    imgs = 0
    t_bench_start = None
    for i in range(warmup + steps):
        _sync()
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(prefetcher)
            batch = next(it)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(batch["degraded"])
            loss = criterion(pred, batch["clean"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _sync()
        if i >= warmup:
            if t_bench_start is None:
                t_bench_start = t0
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            imgs += batch["degraded"].size(0)

    def _stats(xs):
        xs_s = sorted(xs)
        return (sum(xs) / len(xs),
                _percentile(xs_s, 50),
                _percentile(xs_s, 95))

    d_mean, d_p50, d_p95 = _stats(data_ms)
    c_mean, c_p50, c_p95 = _stats(compute_ms)
    s_mean, s_p50, s_p95 = _stats(step_ms)

    overlap_gain = max(0.0, d_mean + c_mean - s_mean)
    denom = min(d_mean, c_mean)
    overlap_eff = (overlap_gain / denom) if denom > 0 else 0.0
    # Bottleneck: whichever leg dominates the step time (>20% margin).
    if d_mean > c_mean * 1.2:
        bottleneck = f"DATA-bound (data {d_mean:.2f}ms > compute {c_mean:.2f}ms)"
    elif c_mean > d_mean * 1.2:
        bottleneck = f"COMPUTE-bound (compute {c_mean:.2f}ms > data {d_mean:.2f}ms)"
    else:
        bottleneck = f"BALANCED (data {d_mean:.2f}ms ~= compute {c_mean:.2f}ms)"

    ceiling = args.batch_size / (max(d_mean, c_mean) / 1000.0) if max(d_mean, c_mean) > 0 else 0.0
    total_wall = (time.perf_counter() - t_bench_start) if t_bench_start else 0.0
    observed = imgs / total_wall if total_wall > 0 else 0.0

    print("[profile] per-step timing (ms)  mean / p50 / p95")
    print(f"  data-only     {d_mean:7.2f} / {d_p50:7.2f} / {d_p95:7.2f}")
    print(f"  compute-only  {c_mean:7.2f} / {c_p50:7.2f} / {c_p95:7.2f}")
    print(f"  overlapped    {s_mean:7.2f} / {s_p50:7.2f} / {s_p95:7.2f}")
    print(f"[profile] overlap gain      = {overlap_gain:7.2f} ms/step "
          f"(hidden behind the longer leg)")
    print(f"[profile] overlap efficiency= {overlap_eff*100:6.1f}%  "
          f"(1.0 = shorter leg fully hidden)")
    print(f"[profile] bottleneck        = {bottleneck}")
    print(f"[profile] achievable ceiling= {ceiling:8.1f} img/s  "
          f"(images / max(data, compute))")
    print(f"[profile] observed throughput={observed:8.1f} img/s  "
          f"({imgs} imgs in {total_wall:.2f}s)")
    if device.type == "cuda":
        print(f"[profile] peak GPU mem      = "
              f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GiB")

    # Actionable hint
    if "DATA-bound" in bottleneck:
        print("[profile] hint: increase --num_workers / --prefetch_factor, enable "
              "--cache_mode memory|disk, or try --image_backend cv2/torchvision.")
    elif "COMPUTE-bound" in bottleneck:
        print("[profile] hint: workers are keeping up; scale --batch_size, --width, "
              "or --grad_accum_steps to use the headroom.")
    else:
        print("[profile] hint: pipeline is balanced; small gains possible from either side.")

    writer = make_tb_writer(args, "bottleneck")
    if writer is not None:
        bclass = ("data" if "DATA-bound" in bottleneck
                  else "compute" if "COMPUTE-bound" in bottleneck else "balanced")
        _tb_log_pipeline(writer, args, "bottleneck",
                         data_ms=data_ms, compute_ms=compute_ms, step_ms=step_ms,
                         throughput=observed,
                         extra={"bottleneck_data_bound": float(bclass == "data"),
                                "bottleneck_compute_bound": float(bclass == "compute"),
                                "peak_gpu_mem_gib": (
                                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                                    if device.type == "cuda" else 0.0)})
        writer.add_text("bottleneck", bottleneck)
        writer.close()







def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean_dir", type=str, required=True)
    p.add_argument("--degraded_dir", type=str, default=None)
    p.add_argument("--self_train", action="store_true",
                   help="Generate degraded inputs on-the-fly each epoch.")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Batches preloaded per worker (DataLoader prefetch_factor).")
    p.add_argument("--persistent_workers", action="store_true", default=True,
                   help="Keep DataLoader workers alive across epochs.")
    p.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    p.add_argument("--pin_memory", action="store_true", default=True,
                   help="Pin host memory for faster async H2D copies.")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--output", type=str, default="final_model_weights.pt")
    p.add_argument("--validate", action="store_true", default=True,
                   help="Run PSNR/SSIM validation each epoch.")
    p.add_argument("--no_validate", dest="validate", action="store_false")
    p.add_argument("--val_metric", choices=["psnr", "ssim"], default="psnr",
                   help="Metric used to select the best checkpoint.")
    p.add_argument("--val_interval", type=int, default=1,
                   help="Run validation every N epochs (>=1).")
    p.add_argument("--seed", type=int, default=42,
                   help="Master RNG seed for python/numpy/torch (CPU+CUDA) and workers.")
    p.add_argument("--deterministic", action="store_true",
                   help="Force deterministic algorithms and disable cuDNN autotune "
                        "for bit-exact-reproducible runs (slower).")
    p.add_argument("--cudnn_benchmark", action="store_true", default=True,
                   help="Enable cuDNN autotune (default). Ignored when --deterministic.")
    p.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--log_interval", type=int, default=20)

    p.add_argument("--profile_dataloader", action="store_true",
                   help="Run a short torch.profiler trace of the input pipeline and exit.")
    p.add_argument("--profile_steps", type=int, default=20,
                   help="Active profiler steps for --profile_dataloader.")
    p.add_argument("--profile_warmup", type=int, default=3)
    p.add_argument("--profile_out", type=str, default="profiler_trace.json",
                   help="Chrome trace output for --profile_dataloader.")
    p.add_argument("--benchmark_dataloader", action="store_true",
                   help="Run a self-training-shaped loop measuring data-wait, "
                        "queue depth, and end-to-end step time, then exit.")
    p.add_argument("--benchmark_steps", type=int, default=100,
                   help="Measured steps for --benchmark_dataloader (after warmup).")
    p.add_argument("--benchmark_warmup", type=int, default=10,
                   help="Warmup steps excluded from --benchmark_dataloader stats.")
    p.add_argument("--profile_bottleneck", action="store_true",
                   help="Measure data-only vs compute-only vs overlapped step time, "
                        "report prefetch overlap efficiency and bottleneck class, then exit.")
    p.add_argument("--profile_bottleneck_steps", type=int, default=50,
                   help="Measured steps for --profile_bottleneck (after warmup).")
    p.add_argument("--profile_bottleneck_warmup", type=int, default=5,
                   help="Warmup steps excluded from --profile_bottleneck stats.")

    p.add_argument("--image_backend", choices=["pil", "torchvision", "cv2"],
                   default="pil",
                   help="Image decoder backend used by the dataset workers.")
    p.add_argument("--cache_mode", choices=["none", "memory", "disk"],
                   default="none",
                   help="Cache decoded images to skip repeat disk I/O across epochs.")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory for --cache_mode disk (default: <clean_dir>/.decoded_cache).")
    p.add_argument("--tb_log_dir", type=str, default=None,
                   help="TensorBoard log dir (e.g. runs/exp1). Disabled if unset.")
    p.add_argument("--tb_flush_secs", type=int, default=30,
                   help="TensorBoard flush interval in seconds.")
    p.add_argument("--metrics_csv", type=str, default=None,
                   help="Append per-epoch metrics as CSV to this file.")
    p.add_argument("--metrics_json", type=str, default=None,
                   help="Append per-epoch metrics as JSON Lines (one row per epoch).")

    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Accumulate gradients over N micro-batches before "
                        "optimizer.step(). Effective batch = batch_size * N.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from. Restores model, "
                        "optimizer, scheduler, AMP scaler, best-metric, and epoch.")
    p.add_argument("--resume_reset_epoch", action="store_true",
                   help="Load weights/optim state but start again at epoch 1 "
                        "(useful for fine-tuning from --output or a best.pt).")
    return p.parse_args()





def profile_dataloader(loader: DataLoader, device: torch.device, args) -> None:
    """Run a short torch.profiler trace over the input pipeline.

    Measures per-batch wall time for (a) fetching from the DataLoader
    (worker decode + collate + IPC) and (b) H2D copy via CUDAPrefetcher,
    then prints a compact summary plus the top profiler ops.
    """
    from torch.profiler import profile, ProfilerActivity, schedule, record_function

    warmup = max(0, args.profile_warmup)
    active = max(1, args.profile_steps)
    total = warmup + active

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    fetch_times: list[float] = []
    h2d_times: list[float] = []
    prefetch_times: list[float] = []

    sched = schedule(wait=0, warmup=warmup, active=active, repeat=1)
    print(f"[profile] warmup={warmup} active={active} device={device} "
          f"num_workers={loader.num_workers} batch_size={loader.batch_size}")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with profile(activities=activities, schedule=sched, record_shapes=False,
                 profile_memory=False, with_stack=False) as prof:
        # --- Raw DataLoader fetch timing (no prefetcher) ---
        it = iter(loader)
        for i in range(total):
            sync()
            t0 = time.perf_counter()
            with record_function("dataloader_next"):
                batch = next(it)
            sync()
            t1 = time.perf_counter()
            with record_function("h2d_copy"):
                if device.type == "cuda":
                    batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
            sync()
            t2 = time.perf_counter()
            if i >= warmup:
                fetch_times.append((t1 - t0) * 1000.0)
                h2d_times.append((t2 - t1) * 1000.0)
            prof.step()

        # --- CUDAPrefetcher end-to-end batch delivery timing ---
        pf = iter(CUDAPrefetcher(loader, device))
        for i in range(total):
            sync()
            t0 = time.perf_counter()
            with record_function("prefetcher_next"):
                _ = next(pf)
            sync()
            t1 = time.perf_counter()
            if i >= warmup:
                prefetch_times.append((t1 - t0) * 1000.0)

    def _stats(name, xs):
        if not xs:
            print(f"  {name}: (no samples)")
            return
        xs_sorted = sorted(xs)
        mean = sum(xs) / len(xs)
        p50 = xs_sorted[len(xs) // 2]
        p95 = xs_sorted[min(len(xs) - 1, int(len(xs) * 0.95))]
        bs = loader.batch_size or 1
        ips = bs * 1000.0 / mean if mean > 0 else float("inf")
        print(f"  {name:22s} mean={mean:7.2f} ms  p50={p50:7.2f}  p95={p95:7.2f}  "
              f"(~{ips:8.1f} img/s)")

    print("\n[profile] per-batch timings (ms):")
    _stats("dataloader_next", fetch_times)
    _stats("h2d_copy", h2d_times)
    _stats("prefetcher_next (async)", prefetch_times)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("\n[profile] top ops by", sort_key)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    out = Path(args.profile_out)
    try:
        prof.export_chrome_trace(str(out))
        print(f"[profile] chrome trace written to {out}")
    except Exception as e:
        print(f"[profile] failed to export chrome trace: {e}")


METRICS_FIELDS = [
    "epoch", "global_step", "lr", "train_loss", "epoch_time_s",
    "throughput_img_per_s", "peak_gpu_mem_gib",
    "val_loss", "val_psnr", "val_ssim",
    "best_metric_name", "best_metric_value", "is_best",
]


def log_epoch_metrics(args, row: dict) -> None:
    """Append one epoch's metrics to CSV and/or JSONL sinks."""
    row = {k: row.get(k) for k in METRICS_FIELDS}
    if args.metrics_csv:
        path = Path(args.metrics_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
    if args.metrics_json:
        path = Path(args.metrics_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def main():


    args = parse_args()
    set_reproducibility(args.seed, args.deterministic, args.cudnn_benchmark)
    if args.deterministic:
        print(f"[repro] deterministic mode ON (seed={args.seed}, cudnn.benchmark=False)")
    else:
        print(f"[repro] seed={args.seed} cudnn.benchmark={torch.backends.cudnn.benchmark}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    full_ds = SelfTrainingDataset(
        clean_dir=args.clean_dir,
        degraded_dir=args.degraded_dir,
        self_train=args.self_train,
        patch_size=args.patch_size,
        augment=True,
        image_backend=args.image_backend,
        cache_mode=args.cache_mode,
        cache_dir=args.cache_dir,
    )
    val_len = max(1, int(len(full_ds) * args.val_split))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Dataset: total={len(full_ds)} train={train_len} val={val_len} "
          f"self_train={args.self_train}")

    nw = args.num_workers
    persistent = args.persistent_workers and nw > 0
    pin = args.pin_memory and device.type == "cuda"
    common = dict(pin_memory=pin, persistent_workers=persistent)
    if nw > 0:
        common["prefetch_factor"] = args.prefetch_factor
        common["worker_init_fn"] = seed_worker

    # Per-epoch generators — reseeded each epoch below so persistent workers
    # get fresh, non-overlapping seeds every epoch.
    train_gen = torch.Generator().manual_seed(args.seed)
    val_gen = torch.Generator().manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, drop_last=True, generator=train_gen, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, nw // 2), generator=val_gen, **common,
    )

    if args.profile_dataloader:
        profile_dataloader(train_loader, device, args)
        return

    if args.benchmark_dataloader:
        benchmark_dataloader(train_loader, device, args)
        return

    if args.profile_bottleneck:
        profile_bottleneck(train_loader, device, args)
        return



    model = NAFNet_SR(in_channels=1, out_channels=1, width=args.width,
                     upscale=args.upscale).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                      betas=(0.9, 0.9))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    criterion = CharbonnierLoss()

    best_metric = -1.0
    metric_name = args.val_metric
    out_path = Path(args.output)
    start_epoch = 1
    global_step = 0

    # ---- optional resume ----
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {ckpt_path}")
        print(f"[resume] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[resume] state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"[resume] scaler state incompatible, resetting ({e})")
        if not args.resume_reset_epoch:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
        prev_metric = args.val_metric
        prev_best_name = ckpt.get("best_metric", metric_name)
        if prev_best_name == metric_name:
            key = f"val_{metric_name}"
            if key in ckpt and isinstance(ckpt[key], (int, float)):
                best_metric = float(ckpt[key])
        print(f"[resume] resumed at epoch {start_epoch} (global_step={global_step}, "
              f"best {metric_name}={best_metric:.4f})")



    # ---- TensorBoard ----
    writer = make_tb_writer(args)


    accum = max(1, args.grad_accum_steps)
    if accum > 1:
        print(f"[grad-accum] {accum} micro-batches/step -> effective batch "
              f"= {args.batch_size * accum}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Reseed loader generator so worker torch/numpy/random RNGs advance
        # deterministically each epoch (fresh degradations, no overlap).
        train_gen.manual_seed(args.seed + epoch)
        model.train()
        t0 = time.time()
        running = 0.0
        imgs_seen = 0
        optimizer.zero_grad(set_to_none=True)
        steps_in_epoch = len(train_loader)
        # Input-pipeline instrumentation: data_wait is the blocking time spent
        # waiting on the prefetcher (already overlapped H2D), compute is the
        # remainder of the step. Cheap (no extra CUDA syncs beyond loss.item()).
        data_wait_ms, compute_ms_ep = [], []
        prefetch_iter = iter(CUDAPrefetcher(train_loader, device))
        step = 0
        while True:
            t_wait = time.perf_counter()
            try:
                batch = next(prefetch_iter)
            except StopIteration:
                break
            t_data_end = time.perf_counter()
            step += 1
            degraded = batch["degraded"]
            clean = batch["clean"]

            with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(degraded)
                loss = criterion(pred, clean)
                # Scale so the accumulated gradient matches a single large batch.
                loss_scaled = loss / accum
            scaler.scale(loss_scaled).backward()

            is_accum_boundary = (step % accum == 0) or (step == steps_in_epoch)
            if is_accum_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item()  # syncs, so compute timing below is real
            t_step_end = time.perf_counter()
            data_wait_ms.append((t_data_end - t_wait) * 1000.0)
            compute_ms_ep.append((t_step_end - t_data_end) * 1000.0)
            running += loss_val
            imgs_seen += degraded.size(0)
            global_step += 1
            if writer is not None:
                writer.add_scalar("train/loss_step", loss_val, global_step)
                writer.add_scalar("pipeline/data_wait_ms_step", data_wait_ms[-1], global_step)
                writer.add_scalar("pipeline/compute_ms_step", compute_ms_ep[-1], global_step)
            if step % args.log_interval == 0:
                print(f"epoch {epoch:3d} step {step:5d}/{steps_in_epoch} "
                      f"loss {running / step:.4f} lr {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        dt = time.time() - t0
        train_loss_epoch = running / max(1, len(train_loader))
        throughput = imgs_seen / dt if dt > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        n_steps = max(1, len(data_wait_ms))
        data_mean = sum(data_wait_ms) / n_steps
        compute_mean = sum(compute_ms_ep) / n_steps
        step_mean = data_mean + compute_mean
        # Loader cost hidden behind compute by the workers + CUDAPrefetcher:
        # any wait shorter than the compute leg means that much loader work was
        # absorbed for free; positive data_wait is unhidden (starving) time.
        overlap_gain = max(0.0, compute_mean - data_mean)

        data_frac = (data_mean / step_mean) if step_mean > 0 else 0.0
        print(f"[pipeline] epoch {epoch:3d} data_wait {data_mean:6.2f} ms  "
              f"compute {compute_mean:6.2f} ms  data_share {data_frac*100:5.1f}%  "
              f"{throughput:8.1f} img/s")

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)
            writer.add_scalar("train/throughput_img_per_s", throughput, epoch)
            writer.add_scalar("train/epoch_time_s", dt, epoch)
            d_s, c_s = sorted(data_wait_ms), sorted(compute_ms_ep)
            writer.add_scalar("pipeline/data_wait_ms_mean", data_mean, epoch)
            writer.add_scalar("pipeline/data_wait_ms_p95", _percentile(d_s, 95), epoch)
            writer.add_scalar("pipeline/compute_ms_mean", compute_mean, epoch)
            writer.add_scalar("pipeline/compute_ms_p95", _percentile(c_s, 95), epoch)
            writer.add_scalar("pipeline/step_ms_mean", step_mean, epoch)
            writer.add_scalar("pipeline/data_share", data_frac, epoch)
            writer.add_scalar("pipeline/overlap_gain_ms", overlap_gain, epoch)
            writer.add_scalar("pipeline/throughput_img_per_s", throughput, epoch)
            writer.add_scalar(
                "pipeline/ceiling_img_per_s",
                args.batch_size / (max(data_mean, compute_mean) / 1000.0)
                if max(data_mean, compute_mean) > 0 else 0.0, epoch)
            writer.add_scalars("pipeline/time_split_ms",
                               {"data_wait": data_mean, "compute": compute_mean}, epoch)
            if device.type == "cuda":
                writer.add_scalar("gpu/peak_mem_gib",
                                  torch.cuda.max_memory_allocated() / (1024 ** 3), epoch)


        peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if device.type == "cuda" else 0.0)
        base_row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": current_lr,
            "train_loss": train_loss_epoch,
            "epoch_time_s": dt,
            "throughput_img_per_s": throughput,
            "peak_gpu_mem_gib": peak_mem,
            "val_loss": None, "val_psnr": None, "val_ssim": None,
            "best_metric_name": metric_name,
            "best_metric_value": best_metric if best_metric >= 0 else None,
            "is_best": False,
        }

        do_val = args.validate and (epoch % max(1, args.val_interval) == 0)
        if not do_val:
            print(f"[epoch {epoch}] train_loss={train_loss_epoch:.4f} "
                  f"time={dt:.1f}s throughput={throughput:.1f} img/s (validation skipped)")
            log_epoch_metrics(args, base_row)
            continue


        # ---- validation ----
        model.eval()
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in CUDAPrefetcher(val_loader, device):
                degraded = batch["degraded"]
                clean = batch["clean"]

                with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    pred = model(degraded)
                pred_f = pred.float()
                bs = degraded.size(0)
                val_loss_sum += criterion(pred_f, clean).item() * bs
                val_psnr_sum += psnr(pred_f, clean) * bs
                val_ssim_sum += ssim(pred_f, clean) * bs
                n_val += bs
        n_val = max(1, n_val)
        val_psnr = val_psnr_sum / n_val
        val_ssim = val_ssim_sum / n_val
        val_loss = val_loss_sum / n_val
        current = val_psnr if metric_name == "psnr" else val_ssim
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} val_psnr={val_psnr:.3f} dB "
              f"val_ssim={val_ssim:.4f} throughput={throughput:.1f} img/s time={dt:.1f}s")

        if writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/psnr", val_psnr, epoch)
            writer.add_scalar("val/ssim", val_ssim, epoch)

        is_best = False
        if current > best_metric:
            best_metric = current
            is_best = True
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "best_metric": metric_name,
                    "epoch": epoch,
                    "global_step": global_step,
                },
                out_path,
            )
            print(f"  -> new best {metric_name}={best_metric:.4f}, weights saved to {out_path}")
            if writer is not None:
                writer.add_scalar(f"val/best_{metric_name}", best_metric, epoch)

        # Always keep a resumable "last" checkpoint alongside best.
        last_path = out_path.with_name(out_path.stem + "_last" + out_path.suffix)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "best_metric": metric_name,
                "epoch": epoch,
                "global_step": global_step,
            },
            last_path,
        )

        row = dict(base_row)
        row.update({
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_metric_value": best_metric,
            "is_best": is_best,
        })
        log_epoch_metrics(args, row)


    if writer is not None:
        writer.close()

    if args.validate:
        print(f"Training done. Best val {metric_name} = {best_metric:.4f}. Weights: {out_path}")
    else:
        # No validation happened; save final weights so the run isn't wasted.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "epoch": args.epochs,
                "global_step": global_step,
            },
            out_path,
        )
        print(f"Training done (no validation). Final weights saved to {out_path}")


if __name__ == "__main__":
    main()
