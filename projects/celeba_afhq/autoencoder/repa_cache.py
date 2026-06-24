"""Precompute + serve frozen DINOv2 REPA targets, so DINOv2 leaves the training loop.

REPA aligns the AE latent to DINOv2 patch features of the *input* image, and the gradient
never flows into DINOv2 -- it's pure frozen inference. This AE training applies NO augmentation
(``build_datamodule`` passes no ``gpu_transform``; each image is materialized once into a fixed
uint8 store), so every image's DINOv2 target is deterministic and can be computed once offline.

``precompute`` runs DINOv2 over each split in ConcatDataset order and writes an
``(N, dino_dim, g, g)`` fp16 memmap; :class:`RepaConcatDataModule` then wraps the concatenated
dataset so each sample carries its cached target, and the batch becomes ``(images, labels,
repa_targets)``. ``LitAutoEncoder._repa`` consumes the target instead of running DINOv2 --
removing both its ~35% of GPU kernel time and the eager-launch gap it caused.

The cache is keyed by (image_size, repa_model, repa_dino_size): anything that changes the
target invalidates it. fp16 (not bf16 -- numpy has no bf16) is plenty: the target is
L2-normalized in the loss, so the extra fp16 mantissa is irrelevant.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from chimera.data import ConcatImageDataModule
from chimera.data.base import to_bf16_scaled
from chimera.models import DINOV2_HIDDEN_SIZE, Dinov2Features


def repa_dir(data_dir: str, image_size: int, repa_model: str, repa_dino_size: int) -> str:
    """Cache dir for one (size, model, dino_size) target set, next to the uint8 stores."""
    tag = f"repa-{repa_model.replace('/', '_')}-img{image_size}-dino{repa_dino_size}"
    return os.path.join(data_dir, ".chimera_cache", tag)


def repa_paths(data_dir: str, image_size: int, repa_model: str, repa_dino_size: int) -> dict:
    d = repa_dir(data_dir, image_size, repa_model, repa_dino_size)
    return {s: os.path.join(d, f"{s}.npy") for s in ("train", "test")}


def cache_exists(data_dir: str, image_size: int, repa_model: str, repa_dino_size: int) -> bool:
    return all(os.path.exists(p) for p in repa_paths(data_dir, image_size, repa_model, repa_dino_size).values())


class RepaFeatureDataset(Dataset):
    """Wrap a dataset so each item also yields its precomputed REPA target (fp16 memmap row
    aligned to this dataset's index order). Returns ``(img, label, target)``."""

    def __init__(self, base: Dataset, feats: np.ndarray):
        assert len(base) == len(feats), f"dataset/feature length mismatch: {len(base)} vs {len(feats)}"
        self.base = base
        self.feats = feats

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        img, label = self.base[i]
        # Copy the memmap row into an owned, writable array (tiny -- ~192KB -- and done in the
        # worker): torch.from_numpy on a read-only memmap view warns about non-writable tensors.
        feat = torch.from_numpy(np.array(self.feats[i]))  # fp16 (dino_dim, g, g) -> tensor
        return img, label, feat


def collate_repa(samples):
    """Stack ``(img, label, target)`` samples; cast images uint8->bf16/255, keep targets fp16."""
    xs = to_bf16_scaled(torch.stack([s[0] for s in samples]))
    ys = torch.stack([s[1] for s in samples])
    fs = torch.stack([s[2] for s in samples])
    return xs, ys, fs


class RepaConcatDataModule(ConcatImageDataModule):
    """ConcatImageDataModule that also serves precomputed REPA targets.

    Same concat/label logic as the parent (so it's a real LightningDataModule the Trainer
    drives unchanged); after ``setup`` it wraps each split in :class:`RepaFeatureDataset` and
    swaps in :func:`collate_repa`, so loaders yield ``(images, labels, repa_targets)``."""

    def __init__(self, datamodules, *, paths: dict, in_memory: bool = True, **kwargs):
        super().__init__(datamodules, in_memory=in_memory, **kwargs)
        self._repa_paths = paths
        # Targets are large (~8GB) -> memory-map them regardless of the image-store in_memory
        # choice; the dataloader has ample headroom to absorb the per-batch reads.
        self._feat_mmap_mode = "r"
        self._collate = collate_repa

    def setup(self, stage: str):
        super().setup(stage)
        if getattr(self, "train_set", None) is not None and not isinstance(
            self.train_set, RepaFeatureDataset
        ):
            self.train_set = RepaFeatureDataset(
                self.train_set, np.load(self._repa_paths["train"], mmap_mode=self._feat_mmap_mode)
            )
        if getattr(self, "test_set", None) is not None and not isinstance(
            self.test_set, RepaFeatureDataset
        ):
            self.test_set = RepaFeatureDataset(
                self.test_set, np.load(self._repa_paths["test"], mmap_mode=self._feat_mmap_mode)
            )


@torch.no_grad()
def precompute(
    *,
    data_dir: str,
    image_size: int,
    repa_model: str,
    repa_dino_size: int,
    batch_size: int = 128,
    num_workers: int = 7,
    device: str = "cuda",
    build_datamodule,
    force: bool = False,
) -> dict:
    """Run frozen DINOv2 over every split (in ConcatDataset order) and write fp16 target memmaps.

    ``build_datamodule`` is the training script's own factory, so the dataset/order is identical
    to training. Each split's targets are written atomically to ``<repa_dir>/<split>.npy`` shaped
    ``(N, dino_dim, g, g)``; existing files are reused unless ``force``."""
    paths = repa_paths(data_dir, image_size, repa_model, repa_dino_size)
    d = repa_dir(data_dir, image_size, repa_model, repa_dino_size)
    os.makedirs(d, exist_ok=True)
    dino_dim = DINOV2_HIDDEN_SIZE[repa_model]
    g = repa_dino_size // 14  # DINOv2 patch grid per side

    dm = build_datamodule(
        data_dir=data_dir, image_size=image_size, batch_size=batch_size,
        num_workers=num_workers, in_memory=True,
    )
    dm.prepare_data()
    dm.setup("fit")  # loads train_set + test_set

    dino = Dinov2Features(repa_model, image_size=repa_dino_size).to(device, torch.bfloat16).eval()

    for split, dataset in (("train", dm.train_set), ("test", dm.test_set)):
        path = paths[split]
        if os.path.exists(path) and not force:
            print(f"[repa-cache] {split}: exists, skipping ({path})")
            continue
        n = len(dataset)
        tmp = path + ".tmp"
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, dino_dim, g, g))
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=dm._collate, pin_memory=True,
        )
        i = 0
        for images, _ in loader:
            x = images.to(device, non_blocking=True).bfloat16()
            target = dino.as_grid(dino(x))  # (b, dino_dim, g, g) bf16
            b = target.shape[0]
            out[i : i + b] = target.float().cpu().numpy().astype(np.float16)
            i += b
            if i % (batch_size * 20) == 0:
                print(f"[repa-cache] {split}: {i}/{n}")
        out.flush()
        del out
        os.replace(tmp, path)
        print(f"[repa-cache] {split}: wrote {n} targets -> {path}  ({os.path.getsize(path)/1e9:.2f} GB)")
    return paths


def main() -> None:
    import argparse

    from train import build_datamodule  # the training script's own factory

    p = argparse.ArgumentParser(description="Precompute DINOv2 REPA targets for AE training.")
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--repa-model", default="facebook/dinov2-small")
    p.add_argument("--repa-dino-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=7)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    precompute(
        data_dir=args.data_dir, image_size=args.image_size, repa_model=args.repa_model,
        repa_dino_size=args.repa_dino_size, batch_size=args.batch_size,
        num_workers=args.num_workers, build_datamodule=build_datamodule, force=args.force,
    )


if __name__ == "__main__":
    main()
