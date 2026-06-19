"""Vision dataset helpers: cache a torchvision dataset as preprocessed tensors."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import QMNIST
from torchvision.transforms import v2
from tqdm import tqdm


def precompute(dataset, batch_size: int = 1024, num_workers: int = 4):
    """Materialize a dataset into stacked ``(x, y)`` tensors via a DataLoader."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    xs, ys = [], []
    for x, y in tqdm(loader, desc="Precomputing"):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def get_qmnist_blob(path, train: bool, *, data_dir, dtype, size: int = 32):
    """Load QMNIST as a cached ``{"x", "y"}`` tensor blob, downloading on first use.

    Images are resized to ``size`` x ``size`` and cast to ``dtype`` (scaled to
    ``[0, 1]``). The blob is written atomically to ``path`` and reused thereafter.
    """
    path = Path(path)
    if path.exists():
        return torch.load(path)
    preprocess = v2.Compose(
        [v2.ToImage(), v2.ToDtype(dtype, True), v2.Resize((size, size))]
    )
    dataset = QMNIST(root=data_dir, train=train, download=True, transform=preprocess)
    x, y = precompute(dataset)
    blob = {"x": x, "y": y}
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.rename(path)
    return blob
