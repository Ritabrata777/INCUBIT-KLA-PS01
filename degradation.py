"""Synthetic semiconductor degradation pipeline.

Applies physically-motivated degradations to clean grayscale image tensors:
    * Speckle (multiplicative) noise -- may push values outside [0, 1].
    * Gaussian additive noise.
    * Gaussian blur with variable kernel size & sigma.
    * Resolution downsampling (nearest / bilinear / bicubic) then upsample back.

All ops are implemented with torch so they run efficiently inside a DataLoader
worker (CPU) or on GPU if the input tensor lives there.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    gauss = torch.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    return kernel


def gaussian_blur(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """img: (C,H,W) or (N,C,H,W). Returns same shape."""
    squeeze = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    n, c, h, w = img.shape
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = _gaussian_kernel2d(kernel_size, sigma, img.device, img.dtype)
    kernel = kernel.expand(c, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(img, kernel, groups=c)
    return out.squeeze(0) if squeeze else out


@dataclass
class DegradationConfig:
    speckle_prob: float = 0.85
    speckle_std_range: Tuple[float, float] = (0.05, 0.35)

    gauss_noise_prob: float = 0.7
    gauss_noise_std_range: Tuple[float, float] = (0.005, 0.08)

    blur_prob: float = 0.75
    blur_kernel_choices: Sequence[int] = field(default_factory=lambda: (3, 5, 7, 9, 11))
    blur_sigma_range: Tuple[float, float] = (0.4, 2.8)

    downsample_prob: float = 0.7
    downsample_scales: Sequence[int] = field(default_factory=lambda: (2, 4))
    downsample_modes: Sequence[str] = field(
        default_factory=lambda: ("nearest", "bilinear", "bicubic")
    )

    # If True, output is NOT clipped, letting speckle produce out-of-bounds values
    # (this matches the "pixel > 1.0 or < 0.0" requirement).
    allow_out_of_bounds: bool = True


class SemiconductorDegradationPipeline:
    """Applies a random combination of realistic degradations to a clean image.

    Input:  clean_image_tensor of shape (C, H, W) or (N, C, H, W), float in [0, 1].
    Output: degraded tensor with the same spatial shape as the input.
    """

    def __init__(self, config: DegradationConfig | None = None):
        self.cfg = config or DegradationConfig()

    # ------------------------------------------------------------------ ops
    def _speckle(self, img: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.cfg.speckle_std_range)
        noise = torch.randn_like(img) * std
        return img + img * noise  # multiplicative -> can go outside [0, 1]

    def _gauss_noise(self, img: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.cfg.gauss_noise_std_range)
        return img + torch.randn_like(img) * std

    def _blur(self, img: torch.Tensor) -> torch.Tensor:
        k = random.choice(self.cfg.blur_kernel_choices)
        sigma = random.uniform(*self.cfg.blur_sigma_range)
        return gaussian_blur(img, k, sigma)

    def _downsample_up(self, img: torch.Tensor) -> torch.Tensor:
        scale = random.choice(self.cfg.downsample_scales)
        down_mode = random.choice(self.cfg.downsample_modes)
        up_mode = random.choice(("bilinear", "bicubic"))
        squeeze = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze = True
        _, _, h, w = img.shape
        dh, dw = max(1, h // scale), max(1, w // scale)
        down_kwargs = {} if down_mode == "nearest" else {"align_corners": False}
        low = F.interpolate(img, size=(dh, dw), mode=down_mode, **down_kwargs)
        up = F.interpolate(low, size=(h, w), mode=up_mode, align_corners=False)
        return up.squeeze(0) if squeeze else up

    # -------------------------------------------------------------- pipeline
    def apply_degradation(self, clean_image_tensor: torch.Tensor) -> torch.Tensor:
        img = clean_image_tensor.clone().float()

        # Randomise ordering slightly for diversity (blur/downsample first, noise last)
        if random.random() < self.cfg.blur_prob:
            img = self._blur(img)
        if random.random() < self.cfg.downsample_prob:
            img = self._downsample_up(img)
        if random.random() < self.cfg.speckle_prob:
            img = self._speckle(img)
        if random.random() < self.cfg.gauss_noise_prob:
            img = self._gauss_noise(img)

        if not self.cfg.allow_out_of_bounds:
            img = img.clamp(0.0, 1.0)
        return img

    __call__ = apply_degradation
