"""
degradation.py
Synthetic degradation pipeline for simulating semiconductor imaging
artifacts: speckle noise, Gaussian noise/blur, and resolution downsampling.

The pipeline is designed to plug directly into `dataset.py` so that clean
ground-truth images can be turned into realistic low-quality counterparts
on-the-fly, every epoch, with freshly randomized parameters -- meaning the
model effectively never sees the exact same degraded image twice.
"""

import random
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SemiconductorDegradationPipeline:
    """
    Applies a randomized combination of realistic semiconductor imaging
    degradations to a clean ground-truth image tensor, producing a
    synthetically degraded low-quality counterpart for self-training.

    All operations work on torch tensors shaped (C, H, W) with values
    nominally in [0, 1]. Speckle / Gaussian noise are intentionally left
    UNCLAMPED so pixel values may exceed 1.0 or drop below 0.0, mimicking
    real sensor/optical noise excursions rather than idealized bounded
    noise. Clamping (if desired) should only ever happen at final
    inference/display time, never inside the degradation pipeline itself.
    """

    def __init__(
        self,
        scale_factors=(2, 4),
        gaussian_blur_kernel_range=(3, 9),
        gaussian_blur_sigma_range=(0.2, 3.0),
        gaussian_noise_sigma_range=(0.0, 0.05),
        speckle_noise_sigma_range=(0.0, 0.15),
        downsample_modes=("nearest", "bicubic"),
        blur_prob=0.7,
        gaussian_noise_prob=0.7,
        speckle_noise_prob=0.6,
        downsample_prob=0.9,
    ):
        self.scale_factors = scale_factors
        self.gaussian_blur_kernel_range = gaussian_blur_kernel_range
        self.gaussian_blur_sigma_range = gaussian_blur_sigma_range
        self.gaussian_noise_sigma_range = gaussian_noise_sigma_range
        self.speckle_noise_sigma_range = speckle_noise_sigma_range
        self.downsample_modes = downsample_modes
        self.blur_prob = blur_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.speckle_noise_prob = speckle_noise_prob
        self.downsample_prob = downsample_prob

    # ------------------------------------------------------------------
    # Individual degradation operators
    # ------------------------------------------------------------------
    def _random_gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian blur with a random odd kernel size and sigma,
        simulating soft/hazy edges from optical defocus."""
        k_min, k_max = self.gaussian_blur_kernel_range
        kernel_size = random.randrange(k_min, k_max + 1, 2)  # must be odd
        sigma = random.uniform(*self.gaussian_blur_sigma_range)

        c, h, w = img.shape
        img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H, W, C)
        if c == 1:
            img_np = img_np[:, :, 0]  # OpenCV expects 2D for single channel

        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        if blurred.ndim == 2:
            blurred = blurred[:, :, None]

        return torch.from_numpy(blurred).permute(2, 0, 1).to(img.device, img.dtype)

    def _add_speckle_noise(self, img: torch.Tensor) -> torch.Tensor:
        """
        Multiplicative speckle noise: out = img + img * N(0, sigma^2).

        Deliberately left UNCLAMPED so pixel values may exceed the nominal
        [0, 1] ground-truth bounds, matching real speckle behavior observed
        in semiconductor SEM / optical inspection sensors.
        """
        sigma = random.uniform(*self.speckle_noise_sigma_range)
        noise = torch.randn_like(img) * sigma
        return img + img * noise

    def _add_gaussian_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian sensor/read noise, also left unclamped."""
        sigma = random.uniform(*self.gaussian_noise_sigma_range)
        noise = torch.randn_like(img) * sigma
        return img + noise

    def _random_downsample(self, img: torch.Tensor, scale: Optional[int] = None):
        """
        Downsamples the image spatially by `scale` (or a random factor
        from `self.scale_factors` if not provided) using a randomly chosen
        interpolation mode, simulating resolution loss from optical/sensor
        limitations.

        Returns:
            (low_res_tensor, scale_used)
        """
        if scale is None:
            scale = random.choice(self.scale_factors)
        mode = random.choice(self.downsample_modes)

        c, h, w = img.shape
        new_h, new_w = max(1, h // scale), max(1, w // scale)

        kwargs = {}
        if mode == "bicubic":
            kwargs["align_corners"] = False

        lr = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode=mode, **kwargs)
        return lr.squeeze(0), scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_degradation(self, clean_image_tensor: torch.Tensor, scale: Optional[int] = None):
        """
        Applies a random combination of degradations to a clean image tensor.

        Args:
            clean_image_tensor: (C, H, W) float tensor, values in [0, 1].
            scale: optional fixed downsampling factor. If provided, the
                image is ALWAYS downsampled by this factor (only the
                interpolation mode is randomized) -- this is what
                `dataset.py` uses so every training sample keeps a
                consistent LR/HR size ratio matching the model's
                PixelShuffle upscale factor. If None, downsampling is
                applied probabilistically with a randomly chosen factor.

        Returns:
            degraded: (C, H // scale, W // scale) float tensor, UNCLAMPED.
            scale_used: int, the downsampling factor actually applied
                (1 if no downsampling occurred).
        """
        img = clean_image_tensor.clone()

        if random.random() < self.blur_prob:
            img = self._random_gaussian_blur(img)

        if random.random() < self.speckle_noise_prob:
            img = self._add_speckle_noise(img)

        if random.random() < self.gaussian_noise_prob:
            img = self._add_gaussian_noise(img)

        scale_used = 1
        if scale is not None or random.random() < self.downsample_prob:
            img, scale_used = self._random_downsample(img, scale=scale)

        return img, scale_used

    def __call__(self, clean_image_tensor: torch.Tensor, scale: Optional[int] = None):
        return self.apply_degradation(clean_image_tensor, scale=scale)
