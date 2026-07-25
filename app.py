"""
app.py — Gradio web UI for NAFNet-SR image restoration.

Tabs:
  1. Single Image   — upload a degraded (or clean) image, restore it, view comparison
  2. Batch Restore  — point at input/output dirs, run grouped FP16 inference
  3. Train          — kick off or resume a training run with all key flags
  4. Metrics        — PSNR/SSIM scoring against ground truth

Launch:
    python app.py
    python app.py --share        # public Gradio link
    python app.py --port 8080
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Inline PSNR / SSIM helpers (no subprocess needed for the metrics tab)
# ---------------------------------------------------------------------------

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((255.0 ** 2) / mse)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel)
    mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    mu_a_sq, mu_b_sq, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sigma_a_sq = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu_a_sq
    sigma_b_sq = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu_b_sq
    sigma_ab   = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / (
        (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)
    )
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Model loader — cached per (weights_path, width, upscale)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _load_model(weights_path: str, width: int, upscale: int):
    """Load NAFNet_SR from a checkpoint.

    Supports two checkpoint formats:
      - New (Books) format: dict with "model" and optional "args" keys.
        Uses U-Net encoder-decoder architecture (keys: encoders/middle/decoders).
      - Old (flat) format: plain state-dict with keys like "body.X.*".
        Uses flat NAFBlock body architecture; restored via OldNAFNet_SR shim.

    Width and upscale are read from saved args when available; UI sliders are
    used as fallbacks for plain state-dict files.
    """
    key = (weights_path, width, upscale)
    if key in _model_cache:
        return _model_cache[key]

    from model import NAFNet_SR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Unwrap checkpoint wrapper if present
    if isinstance(ckpt, dict) and ("model" in ckpt or "args" in ckpt):
        saved_args     = ckpt.get("args", {})
        resolved_width   = saved_args.get("width",   width)
        resolved_upscale = saved_args.get("upscale", upscale)
        state_dict = ckpt.get("model", ckpt)
    else:
        resolved_width, resolved_upscale = width, upscale
        state_dict = ckpt if not isinstance(ckpt, dict) else ckpt

    # Strip DDP / torch.compile prefixes
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

    # Detect architecture from key names
    keys = set(state_dict.keys())
    is_old_flat = any(k.startswith("body.") for k in keys)

    if is_old_flat:
        # Old flat architecture: body.0..N, body_tail_conv, upsample, final_conv
        # Infer num_blocks from the highest body index present
        num_blocks = max(
            int(k.split(".")[1]) for k in keys if k.startswith("body.")
        ) + 1
        model = _OldNAFNet_SR(
            in_channels=1,
            out_channels=1,
            width=resolved_width,
            num_blocks=num_blocks,
            upscale_factor=resolved_upscale,
        ).to(device)
    else:
        # New U-Net architecture
        model = NAFNet_SR(
            in_channels=1,
            out_channels=1,
            width=resolved_width,
            upscale=resolved_upscale,
        ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _model_cache[key] = (model, device)
    return model, device


# ---------------------------------------------------------------------------
# Shim: old flat NAFNet_SR architecture (for final_model_weights.pt trained
# before the Books upgrade). Kept here so the UI works with both checkpoints.
# ---------------------------------------------------------------------------

class _LayerNorm2d(torch.nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(c))
        self.bias   = torch.nn.Parameter(torch.zeros(c))
        self.eps = eps
    def forward(self, x):
        mu  = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x   = (x - mu) / (var + self.eps).sqrt()
        return x * self.weight[None,:,None,None] + self.bias[None,:,None,None]

class _SimpleGate(torch.nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1); return a * b

class _SCA(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x * self.conv(self.pool(x))

class _OldNAFBlock(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1  = _LayerNorm2d(c)
        self.conv1  = torch.nn.Conv2d(c, c*2, 1)
        self.dwconv = torch.nn.Conv2d(c*2, c*2, 3, 1, 1, groups=c*2)
        self.sg1    = _SimpleGate()
        self.sca    = _SCA(c)
        self.conv2  = torch.nn.Conv2d(c, c, 1)
        self.norm2  = _LayerNorm2d(c)
        self.conv3  = torch.nn.Conv2d(c, c*2, 1)
        self.sg2    = _SimpleGate()
        self.conv4  = torch.nn.Conv2d(c, c, 1)
        self.beta   = torch.nn.Parameter(torch.zeros(1,c,1,1))
        self.gamma  = torch.nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self, x):
        y = self.conv2(self.sca(self.sg1(self.dwconv(self.conv1(self.norm1(x))))))
        x = x + y * self.beta
        y = self.conv4(self.sg2(self.conv3(self.norm2(x))))
        return x + y * self.gamma

class _OldNAFNet_SR(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=32, num_blocks=8, upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.intro         = torch.nn.Conv2d(in_channels, width, 3, 1, 1)
        self.body          = torch.nn.Sequential(*[_OldNAFBlock(width) for _ in range(num_blocks)])
        self.body_tail_conv = torch.nn.Conv2d(width, width, 3, 1, 1)
        self.upsample      = torch.nn.Sequential(
            torch.nn.Conv2d(width, width * (upscale_factor**2), 3, 1, 1),
            torch.nn.PixelShuffle(upscale_factor),
        )
        self.final_conv    = torch.nn.Conv2d(width, out_channels, 3, 1, 1)
    def forward(self, x):
        inp = x
        x   = self.intro(x)
        x   = self.body(x)
        x   = self.body_tail_conv(x)
        x   = self.upsample(x)
        x   = self.final_conv(x)
        inp_up = torch.nn.functional.interpolate(
            inp, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False
        )
        return x + inp_up


def _run_inference(model, device, img_gray: np.ndarray) -> np.ndarray:
    """Single-image inference. Input/output: uint8 grayscale numpy array."""
    tensor = (
        torch.from_numpy(img_gray.astype(np.float32) / 255.0)
        .unsqueeze(0).unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(tensor)
    out = out.float().clamp(0.0, 1.0)
    return (out.squeeze().cpu().numpy() * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Tab 1 — Single Image Restore
# ---------------------------------------------------------------------------

def restore_single(
    input_image,
    degrade_first: bool,
    weights_path: str,
    upscale: int,
    width: int,
):
    if input_image is None:
        return None, None, "Upload an image first."

    weights_path = weights_path.strip()
    if not os.path.isfile(weights_path):
        return None, None, f"Weights file not found: {weights_path}"

    try:
        model, device = _load_model(weights_path, width, upscale)
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    # Gradio delivers RGB numpy; convert to grayscale uint8
    gray = cv2.cvtColor(input_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    original_gt = None

    if degrade_first:
        # New degradation API: apply_degradation(tensor) — no scale arg, returns tensor only
        from degradation import SemiconductorDegradationPipeline
        pipeline = SemiconductorDegradationPipeline()
        clean_t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
        degraded_t = pipeline.apply_degradation(clean_t)
        degraded_np = degraded_t.squeeze(0).numpy()
        # pipeline allows out-of-bounds by default — clamp for display only
        gray_deg = (np.clip(degraded_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        original_gt = gray.copy()
        gray = gray_deg

    try:
        restored = _run_inference(model, device, gray)
    except Exception as e:
        return None, None, f"Inference failed: {e}"

    # Side-by-side comparison panel
    out_h, out_w = restored.shape
    input_up = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    panels = [input_up, restored]
    labels = ["Input (nearest-up)", "Restored"]
    if original_gt is not None:
        gt_up = cv2.resize(original_gt, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        panels.append(gt_up)
        labels.append("Ground truth")

    comparison_rgb = cv2.cvtColor(np.hstack(panels), cv2.COLOR_GRAY2RGB)
    restored_rgb   = cv2.cvtColor(restored,           cv2.COLOR_GRAY2RGB)

    device_str = "GPU (CUDA)" if device.type == "cuda" else "CPU"
    status = (
        f"Done on {device_str}. "
        f"Output: {out_w}×{out_h} px. "
        f"Panel: {' | '.join(labels)}"
    )
    return restored_rgb, comparison_rgb, status


# ---------------------------------------------------------------------------
# Tab 2 — Batch Restore
# ---------------------------------------------------------------------------

def restore_batch(
    input_dir: str,
    output_dir: str,
    weights_path: str,
    batch_size: int,
    upscale: int,
    width: int,
    image_backend: str,
):
    input_dir   = input_dir.strip()
    output_dir  = output_dir.strip()
    weights_path = weights_path.strip()

    if not os.path.isdir(input_dir):
        return f"Input directory not found: {input_dir}"
    if not output_dir:
        return "Please specify an output directory."
    if not os.path.isfile(weights_path):
        return f"Weights file not found: {weights_path}"

    os.makedirs(output_dir, exist_ok=True)

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_paths = sorted(p for p in Path(input_dir).rglob("*") if p.suffix.lower() in EXTS)
    if not all_paths:
        return f"No images found in: {input_dir}"

    try:
        model, device = _load_model(weights_path, width, upscale)
    except Exception as e:
        return f"Failed to load model: {e}"

    # Load images and group by (H, W) — same strategy as advanced evaluation.py
    from PIL import Image as PILImage

    def _load_gray(path: Path) -> np.ndarray:
        if image_backend == "cv2":
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                return arr.astype(np.float32) / 255.0
        return np.asarray(PILImage.open(path).convert("L"), dtype=np.float32) / 255.0

    groups: dict[tuple, list] = {}
    logs = [f"Loading {len(all_paths)} images (backend={image_backend}) …"]
    for p in all_paths:
        try:
            arr = _load_gray(p)
            groups.setdefault(arr.shape, []).append((p, arr))
        except Exception as e:
            logs.append(f"  SKIP {p.name}: {e}")

    total = sum(len(v) for v in groups.values())
    logs.append(f"Grouped into {len(groups)} resolution bucket(s). Running on {device} …\n")

    n_done = 0
    for shape, items in groups.items():
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            imgs = [a for _, a in chunk]

            x = torch.from_numpy(np.stack(imgs)).unsqueeze(1).to(device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    y = model(x)
            y = y.float().clamp(0.0, 1.0).squeeze(1).cpu().numpy()

            for j, (src_path, _) in enumerate(chunk):
                rel = src_path.relative_to(input_dir)
                out_path = Path(output_dir) / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_arr = (np.clip(y[j], 0.0, 1.0) * 255.0).round().astype(np.uint8)
                PILImage.fromarray(out_arr, mode="L").save(out_path)
                n_done += 1

        logs.append(f"  {shape[0]}×{shape[1]}: {len(items)} image(s) done")

    logs.append(f"\nFinished. {n_done}/{total} images saved to: {output_dir}")
    return "\n".join(logs)


# ---------------------------------------------------------------------------
# Tab 3 — Training
# ---------------------------------------------------------------------------

def run_training(
    clean_dir: str,
    val_clean_dir: str,
    self_train: bool,
    epochs: int,
    batch_size: int,
    patch_size: int,
    upscale: int,
    width: int,
    lr: float,
    num_workers: int,
    grad_accum: int,
    output_path: str,
    resume_path: str,
    image_backend: str,
    cache_mode: str,
    tb_log_dir: str,
    metrics_csv: str,
):
    clean_dir   = clean_dir.strip()
    output_path = output_path.strip()

    if not os.path.isdir(clean_dir):
        return f"Clean directory not found: {clean_dir}"

    cmd = [
        sys.executable, "train.py",
        "--clean_dir", clean_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--patch_size", str(patch_size),
        "--upscale", str(upscale),
        "--width", str(width),
        "--lr", str(lr),
        "--num_workers", str(num_workers),
        "--grad_accum_steps", str(grad_accum),
        "--output", output_path,
        "--image_backend", image_backend,
        "--cache_mode", cache_mode,
        "--log_interval", "10",
    ]

    if self_train:
        cmd.append("--self_train")

    if val_clean_dir.strip():
        # Use val split from clean_dir (built into train.py via --val_split)
        pass  # train.py auto-splits; separate val dir not exposed in new API

    if resume_path.strip() and os.path.isfile(resume_path.strip()):
        cmd += ["--resume", resume_path.strip()]

    if tb_log_dir.strip():
        cmd += ["--tb_log_dir", tb_log_dir.strip()]

    if metrics_csv.strip():
        cmd += ["--metrics_csv", metrics_csv.strip()]

    cwd = str(Path(__file__).parent)
    yield f"Starting training …\nCommand: {' '.join(cmd)}\n\n"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            yield line
        proc.wait()
        if proc.returncode == 0:
            yield "\n\nTraining completed successfully."
        else:
            yield f"\n\nProcess exited with code {proc.returncode}."
    except Exception as e:
        yield f"\nError launching training process: {e}"


# ---------------------------------------------------------------------------
# Tab 4 — Metrics
# ---------------------------------------------------------------------------

def compute_metrics_ui(
    restored_dir: str,
    gt_dir: str,
    degraded_dir: str,
):
    restored_dir = restored_dir.strip()
    gt_dir       = gt_dir.strip()
    degraded_dir = degraded_dir.strip()

    for label, d in [("Restored", restored_dir), ("Ground truth", gt_dir)]:
        if not os.path.isdir(d):
            return f"{label} directory not found: {d}"

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    gt_paths = sorted(p for p in Path(gt_dir).iterdir() if p.suffix.lower() in EXTS)
    if not gt_paths:
        return f"No images found in ground-truth directory: {gt_dir}"

    model_psnrs, model_ssims     = [], []
    baseline_psnrs, baseline_ssims = [], []
    skipped = 0

    for gt_path in gt_paths:
        name = gt_path.name
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        rst_path = Path(restored_dir) / name
        if not rst_path.exists():
            skipped += 1
            continue
        rst = cv2.imread(str(rst_path), cv2.IMREAD_GRAYSCALE)
        if gt is None or rst is None:
            skipped += 1
            continue

        h = min(gt.shape[0], rst.shape[0])
        w = min(gt.shape[1], rst.shape[1])
        gt_c, rst_c = gt[:h, :w], rst[:h, :w]

        model_psnrs.append(_psnr(rst_c, gt_c))
        model_ssims.append(_ssim(rst_c, gt_c))

        if os.path.isdir(degraded_dir):
            deg_path = Path(degraded_dir) / name
            deg = cv2.imread(str(deg_path), cv2.IMREAD_GRAYSCALE)
            if deg is not None:
                baseline = cv2.resize(
                    deg, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC
                )
                bc = baseline[:h, :w]
                baseline_psnrs.append(_psnr(bc, gt_c))
                baseline_ssims.append(_ssim(bc, gt_c))

    if not model_psnrs:
        return "No matching restored/GT image pairs found."

    lines = [f"Evaluated on {len(model_psnrs)} image(s)  (skipped {skipped})\n"]
    lines.append(f"{'Method':<30} {'PSNR (dB)':>10} {'SSIM':>8}")
    lines.append("─" * 52)
    if baseline_psnrs:
        lines.append(
            f"{'Bicubic baseline':<30} {np.mean(baseline_psnrs):>10.3f}"
            f" {np.mean(baseline_ssims):>8.4f}"
        )
    lines.append(
        f"{'NAFNet-SR (ours)':<30} {np.mean(model_psnrs):>10.3f}"
        f" {np.mean(model_ssims):>8.4f}"
    )
    if baseline_psnrs:
        dp = np.mean(model_psnrs)   - np.mean(baseline_psnrs)
        ds = np.mean(model_ssims)   - np.mean(baseline_ssims)
        lines.append("─" * 52)
        lines.append(f"{'Improvement':<30} {dp:>+10.3f} {ds:>+8.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = str(Path(__file__).parent / "final_model_weights.pt")


def build_ui():
    with gr.Blocks(title="NAFNet-SR — Semiconductor Image Restoration", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
            # NAFNet-SR — Semiconductor Image Restoration
            Self-training PyTorch pipeline for denoising, deblurring, and super-resolution
            of semiconductor inspection images (KLA Hackathon — INCUBIT-KLA-PS01).
            """
        )

        # ── Shared model settings (collapsed by default) ───────────────────
        with gr.Accordion("Model settings", open=False):
            with gr.Row():
                weights_input  = gr.Textbox(value=DEFAULT_WEIGHTS, label="Weights path (.pt)", scale=3)
                upscale_input  = gr.Slider(1, 8,   value=4,  step=1, label="Upscale factor")
                width_input    = gr.Slider(8, 128,  value=32, step=8, label="Width")

        # ── Tab 1: Single image ────────────────────────────────────────────
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input    = gr.Image(label="Input image", type="numpy")
                    degrade_cb   = gr.Checkbox(
                        label="Degrade first (synthesize degradation from clean input)",
                        value=False,
                    )
                    run_btn      = gr.Button("Restore", variant="primary")
                    status_out   = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Column(scale=2):
                    restored_out   = gr.Image(label="Restored output")
                    comparison_out = gr.Image(label="Comparison  (input | restored | GT)")

            run_btn.click(
                fn=restore_single,
                inputs=[img_input, degrade_cb, weights_input, upscale_input, width_input],
                outputs=[restored_out, comparison_out, status_out],
            )

        # ── Tab 2: Batch restore ───────────────────────────────────────────
        with gr.Tab("Batch Restore"):
            with gr.Row():
                b_input_dir  = gr.Textbox(label="Input dir (degraded images)", value="data/test_input")
                b_output_dir = gr.Textbox(label="Output dir",                  value="data/test_output")
            with gr.Row():
                b_batch_size = gr.Slider(1, 128, value=32, step=1, label="Batch size")
                b_backend    = gr.Radio(["pil", "cv2", "torchvision"], value="pil", label="Image backend")

            batch_btn = gr.Button("Run batch inference", variant="primary")
            batch_log = gr.Textbox(label="Log", lines=14, interactive=False)

            batch_btn.click(
                fn=restore_batch,
                inputs=[
                    b_input_dir, b_output_dir,
                    weights_input, b_batch_size,
                    upscale_input, width_input,
                    b_backend,
                ],
                outputs=batch_log,
            )

        # ── Tab 3: Training ────────────────────────────────────────────────
        with gr.Tab("Train"):
            gr.Markdown(
                "Launches `train.py` as a subprocess and streams its output live. "
                "Keep this tab open while training."
            )
            with gr.Row():
                t_clean_dir  = gr.Textbox(label="Clean dir",          value="data/clean")
                t_val_dir    = gr.Textbox(label="Val clean dir (info only)", value="data/val_clean")
                t_output     = gr.Textbox(label="Output weights path", value="final_model_weights.pt")

            with gr.Row():
                t_self_train = gr.Checkbox(label="Self-train (on-the-fly degradation)", value=True)
                t_epochs     = gr.Slider(1, 500, value=100, step=1,  label="Epochs")
                t_batch_size = gr.Slider(1, 128, value=16,  step=1,  label="Batch size")
                t_patch_size = gr.Slider(32, 512, value=256, step=32, label="Patch size")

            with gr.Row():
                t_upscale    = gr.Slider(1, 8,   value=4,    step=1,    label="Upscale")
                t_width      = gr.Slider(8, 128, value=32,   step=8,    label="Width")
                t_lr         = gr.Number(value=2e-4, label="Learning rate")
                t_workers    = gr.Slider(0, 16,  value=4,    step=1,    label="Num workers")
                t_grad_accum = gr.Slider(1, 32,  value=1,    step=1,    label="Grad accum steps")

            with gr.Accordion("Advanced options", open=False):
                with gr.Row():
                    t_resume    = gr.Textbox(label="Resume checkpoint path (optional)", value="")
                    t_backend   = gr.Radio(["pil", "cv2", "torchvision"], value="pil", label="Image backend")
                    t_cache     = gr.Radio(["none", "memory", "disk"],    value="none", label="Cache mode")
                with gr.Row():
                    t_tb_dir    = gr.Textbox(label="TensorBoard log dir (optional)", value="")
                    t_csv       = gr.Textbox(label="Metrics CSV output (optional)",  value="")

            train_btn = gr.Button("Start training", variant="primary")
            train_log = gr.Textbox(label="Training log", lines=20, interactive=False)

            train_btn.click(
                fn=run_training,
                inputs=[
                    t_clean_dir, t_val_dir, t_self_train,
                    t_epochs, t_batch_size, t_patch_size,
                    t_upscale, t_width, t_lr,
                    t_workers, t_grad_accum,
                    t_output, t_resume,
                    t_backend, t_cache,
                    t_tb_dir, t_csv,
                ],
                outputs=train_log,
            )

        # ── Tab 4: Metrics ─────────────────────────────────────────────────
        with gr.Tab("Metrics"):
            with gr.Row():
                m_restored = gr.Textbox(label="Restored dir",  value="data/test_output")
                m_gt       = gr.Textbox(label="Ground-truth dir", value="data/test_gt")
                m_degraded = gr.Textbox(
                    label="Degraded dir (bicubic baseline — optional)",
                    value="data/test_input",
                )
            metrics_btn = gr.Button("Compute PSNR / SSIM", variant="primary")
            metrics_out = gr.Textbox(label="Results", lines=10, interactive=False)

            metrics_btn.click(
                fn=compute_metrics_ui,
                inputs=[m_restored, m_gt, m_degraded],
                outputs=metrics_out,
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",  action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port",   type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(share=args.share, server_port=args.port)
