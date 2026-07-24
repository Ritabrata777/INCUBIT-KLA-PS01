"""
dataset.py
Dataset classes for semiconductor image restoration.

`SelfTrainingDataset` supports two modes:
  1. Static mode (`self_train=False`): loads pre-paired (clean, degraded)
     images from disk, matched by sorted filename order.
  2. Self-training mode (`self_train=True`, default): loads ONLY clean
     images and dynamically synthesizes a degraded counterpart on every
     `__getitem__` call using `degradation.SemiconductorDegradationPipeline`.
     Because degradation parameters (blur sigma/kernel, noise sigma,
     interpolation mode) are freshly resampled every call, the model never
     sees the exact same noisy image twice -- even across epochs.
"""

import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from degradation import SemiconductorDegradationPipeline

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(directory: str):
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        str(p) for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float32) / 255.0


class SelfTrainingDataset(Dataset):
    """
    Args:
        clean_dir: directory of clean ground-truth images.
        degraded_dir: directory of pre-degraded images, paired by sorted
            filename order with `clean_dir`. Required when
            `self_train=False`; ignored when `self_train=True`.
        self_train: if True, ignore `degraded_dir` and synthesize degraded
            images on-the-fly from `clean_dir` using the degradation
            pipeline. If False, load static pre-paired images instead.
        patch_size: side length (in HR pixels) of a random square crop
            taken from each clean image before degradation. Keeps training
            batchable/efficient. Set to None to use full images (only safe
            with batch_size=1 or pre-uniformly-sized images).
        upscale_factor: fixed SR scale factor used by the degradation
            pipeline and expected by the model's PixelShuffle head. Kept
            fixed across all samples so batches have consistent shapes.
        degradation_pipeline: optional pre-configured
            SemiconductorDegradationPipeline instance. If None, a default
            one is created internally.
    """

    def __init__(
        self,
        clean_dir: str,
        degraded_dir: Optional[str] = None,
        self_train: bool = True,
        patch_size: Optional[int] = 128,
        upscale_factor: int = 4,
        degradation_pipeline: Optional[SemiconductorDegradationPipeline] = None,
    ):
        super().__init__()
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.self_train = self_train
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

        self.clean_paths = _list_images(clean_dir)
        if len(self.clean_paths) == 0:
            raise RuntimeError(f"No images found in clean_dir: {clean_dir}")

        self.degraded_paths: List[str] = []
        if not self.self_train:
            if degraded_dir is None:
                raise ValueError("degraded_dir must be provided when self_train=False")
            self.degraded_paths = _list_images(degraded_dir)
            if len(self.degraded_paths) != len(self.clean_paths):
                raise ValueError(
                    "clean_dir and degraded_dir must contain the same number of "
                    f"images (got {len(self.clean_paths)} vs {len(self.degraded_paths)})"
                )

        self.pipeline = degradation_pipeline or SemiconductorDegradationPipeline()

    def __len__(self):
        return len(self.clean_paths)

    def _crop_size(self, h: int, w: int) -> int:
        """Chooses a crop size that fits within the image and is an exact
        multiple of `upscale_factor`, so LR/HR shapes always line up."""
        ps = min(self.patch_size, h, w)
        ps -= ps % self.upscale_factor
        return max(ps, self.upscale_factor)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        ps = self._crop_size(h, w)
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return img[top: top + ps, left: left + ps]

    def _paired_random_crop(self, clean_np: np.ndarray, degraded_np: np.ndarray):
        """Randomly crops matching regions from a clean/degraded pair,
        assuming `degraded_np` is exactly `clean_np` downsampled by an
        integer factor (i.e. already at the LR resolution)."""
        h, w = clean_np.shape
        dh, dw = degraded_np.shape
        scale_h = max(1, h // dh)
        scale_w = max(1, w // dw)

        ps = self._crop_size(h, w)
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        # Align the crop to the LR grid so it maps to whole LR pixels.
        top -= top % scale_h
        left -= left % scale_w
        ps_h = ps - (ps % scale_h)
        ps_w = ps - (ps % scale_w)
        ps_h = max(ps_h, scale_h)
        ps_w = max(ps_w, scale_w)

        clean_crop = clean_np[top: top + ps_h, left: left + ps_w]
        lr_top, lr_left = top // scale_h, left // scale_w
        lr_h, lr_w = ps_h // scale_h, ps_w // scale_w
        degraded_crop = degraded_np[lr_top: lr_top + lr_h, lr_left: lr_left + lr_w]
        return clean_crop, degraded_crop

    def __getitem__(self, idx):
        clean_np = _load_grayscale(self.clean_paths[idx])

        if self.self_train:
            if self.patch_size is not None:
                clean_np = self._random_crop(clean_np)
            clean_tensor = torch.from_numpy(clean_np).unsqueeze(0).float()
            degraded_tensor, scale_used = self.pipeline.apply_degradation(
                clean_tensor, scale=self.upscale_factor
            )
        else:
            assert self.degraded_paths, "degraded_paths must be populated in static mode"
            degraded_np = _load_grayscale(self.degraded_paths[idx])
            if self.patch_size is not None:
                clean_np, degraded_np = self._paired_random_crop(clean_np, degraded_np)
            clean_tensor = torch.from_numpy(clean_np).unsqueeze(0).float()
            degraded_tensor = torch.from_numpy(degraded_np).unsqueeze(0).float()
            scale_used = self.upscale_factor

        return {
            "degraded": degraded_tensor,
            "clean": clean_tensor,
            "scale": scale_used,
        }
