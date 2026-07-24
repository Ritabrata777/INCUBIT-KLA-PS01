"""
test_single_image.py
Quick ad-hoc test of the trained NAFNet-SR model on a single photo.

Two modes:

1. Default (your photo IS the degraded/low-quality image you want restored):
       python test_single_image.py --image my_photo.jpg

2. --degrade_first (your photo is a CLEAN image; this script synthetically
   degrades it first using degradation.py, then restores it, so you can see
   a full before/after/ground-truth comparison on any photo you have lying
   around, semiconductor or not):
       python test_single_image.py --image my_photo.jpg --degrade_first

Outputs (written to --output_dir, default "single_test_output/"):
  - degraded.png    (only with --degrade_first: the synthetically degraded input)
  - restored.png    the model's restored output
  - comparison.png  side-by-side visual: [input | restored | original if available]
"""

import argparse
import os

import cv2
import numpy as np
import torch

from degradation import SemiconductorDegradationPipeline
from model import NAFNet_SR


def parse_args():
    parser = argparse.ArgumentParser(description="Test NAFNet-SR on a single photo")
    parser.add_argument("--image", type=str, required=True, help="Path to the input photo")
    parser.add_argument("--weights_path", type=str, default="final_model_weights.pt")
    parser.add_argument("--output_dir", type=str, default="single_test_output")
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument(
        "--degrade_first",
        action="store_true",
        help="Treat --image as a CLEAN photo: synthetically degrade it first, "
             "then restore, so you can compare against the true original.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    scale = args.upscale_factor
    original_gt = None

    if args.degrade_first:
        # Crop to a multiple of the upscale factor, then synthetically degrade
        # it using the exact same pipeline used during training.
        h, w = img.shape
        h -= h % scale
        w -= w % scale
        original_gt = img[:h, :w]

        pipeline = SemiconductorDegradationPipeline()
        clean_tensor = torch.from_numpy(original_gt.astype(np.float32) / 255.0).unsqueeze(0)
        degraded_tensor, _ = pipeline.apply_degradation(clean_tensor, scale=scale)

        degraded_np = np.clip(degraded_tensor.squeeze(0).numpy(), 0.0, 1.0)
        input_img = (degraded_np * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, "degraded.png"), input_img)
        print(f"Synthetically degraded input saved to {args.output_dir}/degraded.png")
    else:
        # Treat the provided image as already degraded/low-quality.
        input_img = img

    # --- Load model ---
    model = NAFNet_SR(
        in_channels=1,
        out_channels=1,
        width=args.width,
        num_blocks=args.num_blocks,
        upscale_factor=scale,
    ).to(device)
    state_dict = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Run inference (single image = batch of 1) ---
    input_tensor = torch.from_numpy(input_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(input_tensor)

    output = output.float().clamp(0.0, 1.0)
    restored = (output.squeeze(0).squeeze(0).cpu().numpy() * 255.0).round().astype(np.uint8)

    restored_path = os.path.join(args.output_dir, "restored.png")
    cv2.imwrite(restored_path, restored)
    print(f"Restored output saved to {restored_path}")

    # --- Build a side-by-side comparison image ---
    out_h, out_w = restored.shape
    input_display = cv2.resize(input_img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    panels = [input_display, restored]
    labels = ["input (nearest-upsampled)", "restored"]
    if original_gt is not None:
        panels.append(original_gt)
        labels.append("original ground truth")

    comparison = np.hstack(panels)
    comparison_path = os.path.join(args.output_dir, "comparison.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"Side-by-side comparison ({' | '.join(labels)}) saved to {comparison_path}")


if __name__ == "__main__":
    main()
