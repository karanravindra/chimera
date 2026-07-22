"""
CelebA-HQ DataModule for PyTorch Lightning.

CelebA-HQ, as packaged by StarGAN-v2, is a high-resolution face dataset with two
classes (``female``, ``male``). The source of truth is the Hugging Face dataset
``karanravindra/celeba-hq`` (splits ``train`` / ``val``; features ``id`` /
``image`` / ``class``). ``prepare_data`` decodes+resizes every image once into a
uint8 npy memmap cache (see :mod:`chimera.data._image_cache`); ``setup``
memory-maps it and serves ``(image, label)`` batches with cheap tensor transforms.

The HF ``val`` split is served as this module's ``test``/``predict`` set; the
``fit`` train/val split is carved out of the HF ``train`` split (seeded).

Usage:
    dm = CelebAHQDataModule(data_dir="/mnt/ai/data", batch_size=32, image_size=256)
    trainer.fit(model, datamodule=dm)
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import lightning as pl

from ._image_cache import CachedImageDataset, build_npy_cache, load_classes

HF_REPO = "karanravindra/celeba-hq"


class CelebAHQDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        image_size: int = 256,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        # Cached uint8 CHW -> float [0, 1] (ConvertImageDtype) -> [-1, 1] (Normalize).
        # The cache is already at image_size, so no Resize is needed.
        self.transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.classes: list[str] = []

        self.celeba_train: Optional[Dataset] = None
        self.celeba_val: Optional[Dataset] = None
        self.celeba_test: Optional[Dataset] = None
        self.celeba_predict: Optional[Dataset] = None

    @property
    def _cache_dir(self) -> Path:
        return self.data_dir / "_imgcache" / f"celeba_hq-{self.image_size}"

    def prepare_data(self):
        c = self._cache_dir
        if (c / "train.images.npy").exists() and (c / "test.images.npy").exists():
            return
        os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
        from datasets import load_dataset

        dsd = load_dataset(HF_REPO)
        names = dsd["train"].features["class"].names
        kw = dict(image_col="image", label_col="class", channels=3, class_names=names)
        build_npy_cache(dsd["train"], c, "train", self.image_size, **kw)
        build_npy_cache(dsd["val"], c, "test", self.image_size, **kw)

    def _cached(self, split: str) -> CachedImageDataset:
        c = self._cache_dir
        return CachedImageDataset(
            c / f"{split}.images.npy",
            c / f"{split}.labels.npy",
            transform=self.transform,
        )

    def setup(self, stage: Optional[str] = None):
        self.classes = load_classes(self._cache_dir) or []
        if stage == "fit" or stage is None:
            full_train = self._cached("train")
            n_val = int(len(full_train) * self.val_split)
            n_train = len(full_train) - n_val
            self.celeba_train, self.celeba_val = random_split(
                full_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.celeba_test = self._cached("test")

        if stage == "predict":
            self.celeba_predict = self._cached("test")

    def train_dataloader(self):
        return DataLoader(
            self.celeba_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.celeba_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeba_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.celeba_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = CelebAHQDataModule(data_dir="/mnt/ai/data")
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"classes={dm.classes}")
    print(
        f"train batch: x={x.shape}, y={y.shape}, dtype={x.dtype}, "
        f"range=[{x.min():.2f},{x.max():.2f}]"
    )
