"""
MNIST DataModule for PyTorch Lightning.

Source of truth is the canonical Hugging Face dataset ``ylecun/mnist`` (splits
``train`` / ``test``; features ``image`` (grayscale PIL) / ``label`` 0-9).
``prepare_data`` decodes+resizes every digit once into a uint8 npy memmap cache
(see :mod:`chimera.data._image_cache`); ``setup`` memory-maps it. The HF ``test``
split is served as this module's ``test``/``predict`` set; the ``fit`` train/val
split is carved out of the HF ``train`` split (seeded).

Usage:
    dm = MNISTDataModule(data_dir="/mnt/ai/data", batch_size=128)
    trainer.fit(model, datamodule=dm)
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import lightning as pl

from ._image_cache import CachedImageDataset, build_npy_cache

HF_REPO = "ylecun/mnist"


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.image_size = image_size
        # native digits are 28x28; resize only if asked (e.g. 32 for the AE).
        self._size = image_size or 28

        # Cached uint8 (1,H,W) -> float [0, 1]; matches the old ToTensor step.
        self.transform = transforms.ConvertImageDtype(torch.float)

        self.mnist_train: Optional[Dataset] = None
        self.mnist_val: Optional[Dataset] = None
        self.mnist_test: Optional[Dataset] = None
        self.mnist_predict: Optional[Dataset] = None

    @property
    def _cache_dir(self) -> Path:
        return self.data_dir / "_imgcache" / f"mnist-{self._size}"

    def prepare_data(self):
        c = self._cache_dir
        if (c / "train.images.npy").exists() and (c / "test.images.npy").exists():
            return
        os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
        from datasets import load_dataset

        dsd = load_dataset(HF_REPO)
        kw = dict(image_col="image", label_col="label", channels=1)
        build_npy_cache(dsd["train"], c, "train", self._size, **kw)
        build_npy_cache(dsd["test"], c, "test", self._size, **kw)

    def _cached(self, split: str) -> CachedImageDataset:
        c = self._cache_dir
        return CachedImageDataset(
            c / f"{split}.images.npy", c / f"{split}.labels.npy",
            transform=self.transform,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_train = self._cached("train")
            n_val = int(len(full_train) * self.val_split)
            n_train = len(full_train) - n_val
            self.mnist_train, self.mnist_val = random_split(
                full_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.mnist_test = self._cached("test")

        if stage == "predict":
            self.mnist_predict = self._cached("test")

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = MNISTDataModule(data_dir="/mnt/ai/data")
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"train batch: x={x.shape}, y={y.shape}, dtype={x.dtype}, "
          f"range=[{x.min():.2f},{x.max():.2f}]")
