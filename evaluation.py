"""
evaluation.py
Batch inference script for NAFNet-SR semiconductor image restoration.

Optimized for NVIDIA H100 throughput:
  - Batched processing (batch_size >= 16/32), never a single-image loop.
  - torch.amp.autocast('cuda') FP16 mixed-precision inference.
  - torch.no_grad() to disable autograd bookkeeping entirely.

Usage:
    python evaluation.py --input_dir path/to/test_images \
        --output_dir path/to/restored_images \
        --weights_path final_model_weights.pt
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import NAFNet_SR

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class InferenceDataset(Dataset):
    """Loads grayscale test images for batched inference, keeping track of
    each image's original size so padding added for batching can be
    cropped away after the model runs."""

    def __init__(self, input_dir: str):
        self.paths = sorted(
            str(p) for p in Path(input_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in input_dir: {input_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
        return tensor, os.path.basename(path), img.shape[0], img.shape[1]


def pad_collate(batch):
    """Pads all images in a batch (via edge replication) to the max H/W
    present in that batch, so variably-sized test images can still be
    processed together in a single batched forward pass."""
    tensors, names, heights, widths = zip(*batch)
    max_h = max(heights)
    max_w = max(widths)

    padded = []
    for t in tensors:
        _, h, w = t.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
        padded.append(t)

    batch_tensor = torch.stack(padded, dim=0)
    return batch_tensor, list(names), list(heights), list(widths)


def parse_args():
    parser = argparse.ArgumentParser(description="Batched FP16 inference for NAFNet-SR")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of degraded test images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored images")
    parser.add_argument("--weights_path", type=str, default="final_model_weights.pt", help="Path to trained model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size (>=16 recommended on H100)")
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: CUDA not available -- falling back to CPU, FP16 autocast will be disabled.")

    model = NAFNet_SR(
        in_channels=1,
        out_channels=1,
        width=args.width,
        num_blocks=args.num_blocks,
        upscale_factor=args.upscale_factor,
    ).to(device)

    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = InferenceDataset(args.input_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
    )

    scale = args.upscale_factor
    total_images = 0

    with torch.no_grad():
        for batch_tensor, names, heights, widths in loader:
            batch_tensor = batch_tensor.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(batch_tensor)

            # Cast back to fp32 before clamping/quantizing for numerically
            # stable, deterministic rounding to uint8.
            output = output.float()
            output = torch.clamp(output, 0.0, 1.0)
            output_uint8 = (output * 255.0).round().to(torch.uint8).cpu().numpy()

            for i, name in enumerate(names):
                out_h = heights[i] * scale
                out_w = widths[i] * scale
                restored = output_uint8[i, 0, :out_h, :out_w]

                out_path = os.path.join(args.output_dir, name)
                cv2.imwrite(out_path, restored)
                total_images += 1

    print(f"Restored {total_images} images. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
