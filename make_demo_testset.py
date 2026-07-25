"""
make_demo_testset.py
Builds a demo "unknown test set" by applying the same
SemiconductorDegradationPipeline used during training to a held-out set of
clean images that the model never saw during training or validation.

Produces:
  - data/test_input/  degraded LR images (what evaluation.py restores)
  - data/test_gt/      matching HR clean ground truth (for PSNR/SSIM scoring)

Usage:
    python make_demo_testset.py --clean_dir data/holdout_clean \
        --input_dir data/test_input --gt_dir data/test_gt --upscale_factor 4
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from degradation import SemiconductorDegradationPipeline

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a synthetic degraded demo test set")
    parser.add_argument("--clean_dir", type=str, default="data/holdout_clean")
    parser.add_argument("--input_dir", type=str, default="data/test_input")
    parser.add_argument("--gt_dir", type=str, default="data/test_gt")
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.gt_dir, exist_ok=True)

    pipeline = SemiconductorDegradationPipeline()

    paths = sorted(
        p for p in Path(args.clean_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {args.clean_dir}")

    scale = args.upscale_factor
    count = 0

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        h -= h % scale
        w -= w % scale
        img = img[:h, :w]

        clean_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        degraded_tensor = pipeline.apply_degradation(clean_tensor)

        degraded_np = degraded_tensor.squeeze(0).numpy()
        degraded_np = np.clip(degraded_np, 0.0, 1.0)
        degraded_uint8 = (degraded_np * 255.0).round().astype(np.uint8)

        out_name = path.stem + ".png"
        cv2.imwrite(os.path.join(args.input_dir, out_name), degraded_uint8)
        cv2.imwrite(os.path.join(args.gt_dir, out_name), img)
        count += 1

    print(f"Built {count} degraded test samples in {args.input_dir} (GT in {args.gt_dir})")


if __name__ == "__main__":
    main()
