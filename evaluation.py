"""Batched FP16 inference for semiconductor image restoration on H100.

Usage:
    python evaluation.py --input_dir path/to/degraded --output_dir path/to/out \
        --weights final_model_weights.pt --batch_size 32
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast

from model import NAFNet_SR


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_gray(path: Path, backend: str = "pil") -> np.ndarray:
    """Decode a grayscale image to float32 numpy in [0, 1] using the chosen backend."""
    if backend == "cv2":
        try:
            import cv2  # type: ignore
            try:
                cv2.setNumThreads(1)
            except Exception:
                pass
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                return arr.astype(np.float32) / 255.0
        except Exception:
            pass
    if backend == "torchvision":
        try:
            from torchvision.io import decode_image, read_file, ImageReadMode  # type: ignore
            img = decode_image(read_file(str(path)), mode=ImageReadMode.GRAY)
            return img.squeeze(0).numpy().astype(np.float32) / 255.0
        except Exception:
            pass
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0



def save_gray_uint8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = np.rint(arr).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--weights", type=str, default="final_model_weights.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_backend", choices=["pil", "torchvision", "cv2"],
                   default="pil", help="Image decoder backend for loading inputs.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for python/numpy/torch (CPU+CUDA).")
    p.add_argument("--deterministic", action="store_true",
                   help="Force deterministic algorithms and disable cuDNN autotune. "
                        "Note: FP16 autocast may still introduce tiny non-determinism.")
    return p.parse_args()


def set_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed python/numpy/torch and configure cuDNN for reproducible inference."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"[repro] use_deterministic_algorithms failed: {e}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



@torch.no_grad()
def run_batch(model: torch.nn.Module, batch_np: list[np.ndarray], device) -> list[np.ndarray]:
    """Run inference on a batch of images that share the same H, W."""
    x = torch.from_numpy(np.stack(batch_np, axis=0)).unsqueeze(1).to(device, non_blocking=True)
    with autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        y = model(x)
    y = y.float().clamp(0.0, 1.0).squeeze(1).cpu().numpy()
    return [y[i] for i in range(y.shape[0])]


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_reproducibility(args.seed, args.deterministic)
    if args.deterministic:
        print(f"[repro] deterministic inference (seed={args.seed}, cudnn.benchmark=False)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ---- load model ----
    ckpt = torch.load(args.weights, map_location=device)
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    width = saved_args.get("width", args.width)
    upscale = saved_args.get("upscale", args.upscale)
    model = NAFNet_SR(in_channels=1, out_channels=1, width=width, upscale=upscale).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    paths = list_images(in_dir)
    print(f"Found {len(paths)} images. Running batched FP16 inference on {device}.")

    # Group by (H, W) so we can build true batches without padding overhead.
    groups: dict[tuple[int, int], list[tuple[Path, np.ndarray]]] = {}
    for p in paths:
        arr = load_gray(p, args.image_backend)
        groups.setdefault(arr.shape, []).append((p, arr))

    t0 = time.time()
    n_done = 0
    for shape, items in groups.items():
        for i in range(0, len(items), args.batch_size):
            chunk = items[i : i + args.batch_size]
            imgs = [a for _, a in chunk]
            outs = run_batch(model, imgs, device)
            for (path, _), out_arr in zip(chunk, outs):
                rel = path.relative_to(in_dir)
                out_path = out_dir / rel
                # Preserve original filename & extension.
                save_gray_uint8(out_path, out_arr)
                n_done += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Processed {n_done} images in {dt:.2f}s ({n_done / max(dt, 1e-6):.2f} img/s).")


if __name__ == "__main__":
    main()
