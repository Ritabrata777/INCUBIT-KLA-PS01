"""Self-training dataset for semiconductor image restoration.

Two modes:
    * static  -- pre-existing (degraded, clean) pairs from two parallel dirs.
    * self    -- clean images only; degradation is applied on-the-fly per
                 __getitem__, so the model never sees the same noisy sample twice.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from degradation import SemiconductorDegradationPipeline


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible, non-overlapping RNGs.

    PyTorch assigns each worker a unique ``torch.initial_seed()`` derived
    from the DataLoader's ``generator`` (which we reseed per epoch in
    ``train.py``). We propagate that per-worker, per-epoch seed to
    ``numpy`` and Python's ``random`` module so every degradation call
    inside ``__getitem__`` draws from a distinct, deterministic stream.
    """
    base = torch.initial_seed()
    seed32 = base % (2 ** 32)
    np.random.seed(seed32)
    random.seed(base)
    # Re-seed torch CPU RNG explicitly (defensive; PyTorch already does this).
    torch.manual_seed(base)


def _list_images(folder: str | os.PathLike) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


_TV_IO = None
_CV2 = None


def _get_tv_io():
    global _TV_IO
    if _TV_IO is None:
        from torchvision.io import decode_image, read_file, ImageReadMode  # type: ignore
        _TV_IO = (decode_image, read_file, ImageReadMode)
    return _TV_IO


def _get_cv2():
    global _CV2
    if _CV2 is None:
        import cv2  # type: ignore
        try:
            cv2.setNumThreads(1)  # avoid oversubscription with DataLoader workers
        except Exception:
            pass
        _CV2 = cv2
    return _CV2


def _load_gray_pil(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def _load_gray_torchvision(path: Path) -> torch.Tensor:
    """GPU-friendly decode using torchvision.io (libjpeg-turbo / libpng).

    Falls back to PIL for formats torchvision can't decode (e.g. .tif on
    many builds). Output is a float tensor in [0, 1], shape (1, H, W).
    """
    decode_image, read_file, ImageReadMode = _get_tv_io()
    try:
        data = read_file(str(path))
        img = decode_image(data, mode=ImageReadMode.GRAY)  # uint8 (1,H,W)
        return img.to(torch.float32).div_(255.0)
    except Exception:
        return _load_gray_pil(path)


def _load_gray_cv2(path: Path) -> torch.Tensor:
    """Fast decode via OpenCV (IMREAD_GRAYSCALE) → float32 tensor (1,H,W)."""
    cv2 = _get_cv2()
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return _load_gray_pil(path)
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32).div_(255.0)
    return t.unsqueeze(0)


_BACKENDS = {
    "pil": _load_gray_pil,
    "torchvision": _load_gray_torchvision,
    "cv2": _load_gray_cv2,
}


def _load_gray_tensor(path: Path, backend: str = "pil") -> torch.Tensor:
    loader = _BACKENDS.get(backend)
    if loader is None:
        raise ValueError(f"Unknown image_backend={backend!r}; choose from {list(_BACKENDS)}")
    return loader(path)



class SelfTrainingDataset(Dataset):
    """Semiconductor restoration dataset.

    Args:
        clean_dir: directory with clean ground-truth images (required).
        degraded_dir: optional directory with paired degraded inputs. If given
            AND self_train is False, static pairs are used (filenames must match).
        self_train: if True, ignore degraded_dir and synthesize a fresh degraded
            input from the clean image on every __getitem__ call.
        patch_size: if set, a random crop of this size is drawn each epoch.
        augment: enable random flips / 90-deg rotations.
        pipeline: optional pre-built SemiconductorDegradationPipeline instance.
        cache_mode: ``"none"`` (default), ``"memory"``, or ``"disk"``. When
            enabled, decoded clean (and, in static mode, degraded) tensors are
            cached to skip repeated PNG/JPEG decoding on subsequent epochs.
            ``"memory"`` keeps CPU tensors in a per-worker dict (fastest, uses
            RAM proportional to dataset size). ``"disk"`` stores ``.pt`` files
            under ``cache_dir`` so decode cost is paid once across runs.
        cache_dir: directory for disk cache. Defaults to
            ``<clean_dir>/.decoded_cache`` when ``cache_mode="disk"``.
    """

    def __init__(
        self,
        clean_dir: str,
        degraded_dir: Optional[str] = None,
        self_train: bool = True,
        patch_size: Optional[int] = 256,
        augment: bool = True,
        pipeline: Optional[SemiconductorDegradationPipeline] = None,
        image_backend: str = "pil",
        cache_mode: str = "none",
        cache_dir: Optional[str] = None,
    ):
        if image_backend not in _BACKENDS:
            raise ValueError(
                f"image_backend={image_backend!r} not in {list(_BACKENDS)}"
            )
        if cache_mode not in ("none", "memory", "disk"):
            raise ValueError(
                f"cache_mode={cache_mode!r} not in ('none','memory','disk')"
            )
        self.image_backend = image_backend
        self.clean_paths: Sequence[Path] = _list_images(clean_dir)

        if len(self.clean_paths) == 0:
            raise ValueError(f"No images found in clean_dir={clean_dir}")

        self.self_train = self_train
        self.patch_size = patch_size
        self.augment = augment

        if self_train:
            self.degraded_paths = None
            self.pipeline = pipeline or SemiconductorDegradationPipeline()
        else:
            if degraded_dir is None:
                raise ValueError("degraded_dir required when self_train=False")
            deg = {p.name: p for p in _list_images(degraded_dir)}
            paired_clean, paired_deg = [], []
            for cp in self.clean_paths:
                if cp.name in deg:
                    paired_clean.append(cp)
                    paired_deg.append(deg[cp.name])
            if not paired_clean:
                raise ValueError("No matching (clean, degraded) filename pairs found.")
            self.clean_paths = paired_clean
            self.degraded_paths = paired_deg
            self.pipeline = None

        # --- decoded-image cache ------------------------------------------------
        self.cache_mode = cache_mode
        self._mem_cache: dict[str, torch.Tensor] = {}
        if cache_mode == "disk":
            self.cache_dir = Path(cache_dir) if cache_dir else Path(clean_dir) / ".decoded_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def __len__(self) -> int:
        return len(self.clean_paths)

    # ------------------------------------------------------------------
    def _cache_key(self, path: Path) -> str:
        # Include backend + mtime so a re-encode invalidates stale entries.
        try:
            mtime = int(path.stat().st_mtime)
        except OSError:
            mtime = 0
        return f"{self.image_backend}:{mtime}:{path.resolve()}"

    def _disk_path(self, path: Path) -> Path:
        import hashlib
        h = hashlib.sha1(self._cache_key(path).encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{h}.pt"  # type: ignore[union-attr]

    def _load_cached(self, path: Path) -> torch.Tensor:
        if self.cache_mode == "none":
            return _load_gray_tensor(path, self.image_backend)

        key = self._cache_key(path)
        if self.cache_mode == "memory":
            t = self._mem_cache.get(key)
            if t is None:
                t = _load_gray_tensor(path, self.image_backend).contiguous()
                self._mem_cache[key] = t
            return t

        # disk
        dp = self._disk_path(path)
        if dp.exists():
            try:
                return torch.load(dp, map_location="cpu", weights_only=True)
            except Exception:
                pass  # fall through and re-decode
        t = _load_gray_tensor(path, self.image_backend).contiguous()
        try:
            torch.save(t, dp)
        except Exception:
            pass
        return t


    # ------------------------------------------------------------------
    def _random_crop(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.patch_size is None:
            return tensors
        _, h, w = tensors[0].shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            tensors = tuple(
                torch.nn.functional.pad(t.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
                for t in tensors
            )
            _, h, w = tensors[0].shape
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return tuple(t[:, top : top + ps, left : left + ps] for t in tensors)

    def _augment(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not self.augment:
            return tensors
        if random.random() < 0.5:
            tensors = tuple(torch.flip(t, dims=[-1]) for t in tensors)
        if random.random() < 0.5:
            tensors = tuple(torch.flip(t, dims=[-2]) for t in tensors)
        k = random.randint(0, 3)
        if k:
            tensors = tuple(torch.rot90(t, k, dims=[-2, -1]) for t in tensors)
        return tensors

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        # Draw a fresh per-sample seed from torch's worker RNG (which
        # PyTorch reseeds per epoch, even with persistent_workers=True).
        # This keeps `random` / `numpy.random` streams non-overlapping
        # across workers and non-repeating across epochs, while remaining
        # deterministic given a fixed DataLoader generator seed.
        s = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        random.seed(s)
        np.random.seed(s)

        clean = self._load_cached(self.clean_paths[idx])

        if self.self_train:
            # Crop first for speed, then degrade the patch.
            (clean_patch,) = self._random_crop(clean)
            clean_patch, = self._augment(clean_patch)
            degraded = self.pipeline.apply_degradation(clean_patch)
        else:
            degraded = self._load_cached(self.degraded_paths[idx])

            clean_patch, degraded = self._random_crop(clean, degraded)
            clean_patch, degraded = self._augment(clean_patch, degraded)

        return {"degraded": degraded.float(), "clean": clean_patch.float()}
