"""Decode+resize precompute cache for HF-backed image DataModules.

Each image DataModule (afhq, celebahq, mnist, cifar10) now sources its data from
a Hugging Face dataset (``load_dataset``). Decoding + resizing PIL images on every
``__getitem__`` is the throughput bottleneck, so ``prepare_data`` runs this once:
every image in an HF split is decoded, resized to ``image_size``, and written to a
flat ``uint8`` ``.npy`` memmap (``<split>.images.npy`` as ``(N, C, H, W)`` plus
``<split>.labels.npy``). ``setup`` then memory-maps that cache and applies only
cheap tensor transforms (dtype scale, normalize, on-the-fly augmentation).

Caches are keyed by ``image_size`` in the directory name, so different resolutions
never collide, and the build is idempotent (skips when the files already exist).
This restores the old ``.chimera_cache`` fast path, now derived from the HF source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def build_npy_cache(
    ds,
    cache_dir: str | Path,
    split: str,
    image_size: int,
    *,
    image_col: str = "image",
    label_col: str = "label",
    channels: int = 3,
    class_names: Optional[list[str]] = None,
) -> tuple[Path, Path]:
    """Decode+resize an HF dataset split into uint8 ``(N, C, H, W)`` npy memmaps.

    Idempotent: returns immediately if the cache files already exist. Streams via
    an ``open_memmap`` write so the full split never sits in RAM.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_path = cache_dir / f"{split}.images.npy"
    lbl_path = cache_dir / f"{split}.labels.npy"
    if img_path.exists() and lbl_path.exists():
        return img_path, lbl_path
    if class_names is not None:
        (cache_dir / "classes.json").write_text(json.dumps(class_names))

    n = len(ds)
    mode = "RGB" if channels == 3 else "L"
    # write to a temp name so an interrupted build never leaves a half cache that
    # the existence check would treat as complete.
    tmp_img = cache_dir / f"{split}.images.npy.tmp"
    images = np.lib.format.open_memmap(
        tmp_img, mode="w+", dtype=np.uint8, shape=(n, channels, image_size, image_size)
    )
    labels = np.empty(n, dtype=np.int64)
    for i, ex in enumerate(ds):
        pil = ex[image_col]
        if pil.mode != mode:
            pil = pil.convert(mode)
        if pil.size != (image_size, image_size):
            pil = pil.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.uint8)
        arr = arr[:, :, None] if arr.ndim == 2 else arr  # HW -> HW1
        images[i] = arr.transpose(2, 0, 1)  # HWC -> CHW
        labels[i] = int(ex[label_col])
        if (i + 1) % 5000 == 0:
            print(f"  [{split}] cached {i + 1}/{n}")
    images.flush()
    del images
    tmp_img.replace(img_path)
    np.save(lbl_path, labels)
    print(f"  [{split}] done: {n} imgs -> {img_path}")
    return img_path, lbl_path


def load_classes(cache_dir: str | Path) -> Optional[list[str]]:
    p = Path(cache_dir) / "classes.json"
    return json.loads(p.read_text()) if p.exists() else None


class CachedImageDataset(Dataset):
    """Serves ``(image, label)`` from the uint8 npy memmap, applying ``transform``.

    ``transform`` operates on a ``uint8`` CHW tensor (torchvision transforms accept
    tensors); use ``ConvertImageDtype`` in it to scale to float ``[0, 1]``.
    """

    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # .copy() -> a writable, contiguous array (the memmap is read-only, which
        # torch.from_numpy warns about and some transforms mutate in place).
        img = torch.from_numpy(self.images[idx].copy())  # uint8 CHW
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])
