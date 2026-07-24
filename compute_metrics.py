"""
compute_metrics.py
Quick PSNR/SSIM scoring of restored images against ground truth, plus a
naive bicubic-upsample baseline for comparison. One-off analysis script,
not part of the core 5-file deliverable.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((255.0 ** 2) / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Simple single-scale SSIM (Wang et al.), grayscale, global constants.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel)

    mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    mu_a_sq, mu_b_sq, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b

    sigma_a_sq = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu_a_sq
    sigma_b_sq = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu_b_sq
    sigma_ab = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / (
        (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)
    )
    return float(ssim_map.mean())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restored_dir", type=str, default="data/test_output")
    parser.add_argument("--gt_dir", type=str, default="data/test_gt")
    parser.add_argument("--degraded_dir", type=str, default="data/test_input")
    parser.add_argument("--upscale_factor", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    gt_paths = sorted(Path(args.gt_dir).glob("*.png"))

    model_psnrs, model_ssims = [], []
    baseline_psnrs, baseline_ssims = [], []

    for gt_path in gt_paths:
        name = gt_path.name
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        restored_path = Path(args.restored_dir) / name
        if not restored_path.exists():
            continue
        restored = cv2.imread(str(restored_path), cv2.IMREAD_GRAYSCALE)

        h, w = gt.shape
        rh, rw = restored.shape
        h, w = min(h, rh), min(w, rw)
        gt_c, restored_c = gt[:h, :w], restored[:h, :w]

        model_psnrs.append(psnr(restored_c, gt_c))
        model_ssims.append(ssim(restored_c, gt_c))

        degraded_path = Path(args.degraded_dir) / name
        degraded = cv2.imread(str(degraded_path), cv2.IMREAD_GRAYSCALE)
        baseline = cv2.resize(
            degraded, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        baseline_c = baseline[:h, :w]

        baseline_psnrs.append(psnr(baseline_c, gt_c))
        baseline_ssims.append(ssim(baseline_c, gt_c))

    n = len(model_psnrs)
    print(f"Evaluated on {n} test images\n")
    print(f"{'Method':<28}{'PSNR (dB)':<14}{'SSIM':<10}")
    print(f"{'Bicubic upsample (baseline)':<28}{np.mean(baseline_psnrs):<14.3f}{np.mean(baseline_ssims):<10.4f}")
    print(f"{'NAFNet-SR (ours)':<28}{np.mean(model_psnrs):<14.3f}{np.mean(model_ssims):<10.4f}")


if __name__ == "__main__":
    main()
