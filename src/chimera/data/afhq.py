"""
AFHQ (Animal Faces-HQ) DataModule for PyTorch Lightning.

AFHQ is the StarGAN-v2 animal-faces dataset: 512x512 images across three classes
(``cat``, ``dog``, ``wild``). The source of truth is the Hugging Face dataset
``karanravindra/afhq`` (splits ``train`` / ``val``; features ``id`` / ``image`` /
``class``). ``prepare_data`` decodes+resizes every image once into a uint8 npy
memmap cache (see :mod:`chimera.data._image_cache`); ``setup`` memory-maps it and
serves ``(image, label)`` batches, applying only cheap tensor transforms.

The HF ``val`` split is served as this module's ``test``/``predict`` set; the
``fit`` train/val split is carved out of the HF ``train`` split (seeded), matching
the previous behavior when only a train folder was available.

Usage:
    dm = AFHQDataModule(data_dir="/mnt/ai/data", batch_size=32, image_size=256)
    trainer.fit(model, datamodule=dm)
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

import lightning as pl

from ._image_cache import CachedImageDataset, build_npy_cache, load_classes

HF_REPO = "karanravindra/afhq"


class AFHQDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        image_size: int = 256,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        normalize: bool = True,
        augment: bool = False,
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
        self.normalize = normalize
        self.augment = augment

        # Transforms operate on the cached uint8 CHW tensor. ConvertImageDtype
        # scales uint8 -> float [0, 1] (the old ToTensor step); Normalize then
        # maps to [-1, 1] (the usual GAN/diffusion range). normalize=False keeps
        # [0, 1] to match a sigmoid-output decoder + PSNR/SSIM with data_range=1.0.
        norm = (
            [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            if normalize
            else []
        )
        to_float = transforms.ConvertImageDtype(torch.float)
        # ``self.transform`` is the deterministic eval view (val/test/predict).
        # The cache is already at image_size, so no Resize is needed here.
        self.transform = transforms.Compose([to_float, *norm])
        # ``train_transform`` adds train-only augmentation (never applied to
        # val/test, so reconstruction metrics stay comparable):
        #   - RandomResizedCrop: mild scale/aspect jitter, kept gentle
        #     (scale >=0.8, near-square ratio) so faces aren't cropped out.
        #   - RandomHorizontalFlip: near-free same-difficulty extra data.
        # Both run on the uint8 tensor before the float conversion.
        if augment:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1),
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                to_float,
                *norm,
            ])
        else:
            self.train_transform = self.transform
        self.classes: list[str] = []

        self.afhq_train: Optional[Dataset] = None
        self.afhq_val: Optional[Dataset] = None
        self.afhq_test: Optional[Dataset] = None
        self.afhq_predict: Optional[Dataset] = None

    @property
    def _cache_dir(self) -> Path:
        return self.data_dir / "_imgcache" / f"afhq-{self.image_size}"

    def prepare_data(self):
        # decode+resize the HF dataset into the npy cache, once, on one process.
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

    def _cached(self, split: str, transform) -> CachedImageDataset:
        c = self._cache_dir
        return CachedImageDataset(
            c / f"{split}.images.npy", c / f"{split}.labels.npy", transform=transform
        )

    def setup(self, stage: Optional[str] = None):
        self.classes = load_classes(self._cache_dir) or []
        if stage == "fit" or stage is None:
            # Two views of the same cached train split: the train subset gets
            # augmentation, the val subset stays deterministic. Split the indices
            # (seeded) so augmentation never leaks into validation.
            train_full = self._cached("train", self.train_transform)
            eval_full = self._cached("train", self.transform)
            n_val = int(len(train_full) * self.val_split)
            n_train = len(train_full) - n_val
            train_idx, val_idx = random_split(
                range(len(train_full)),
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )
            self.afhq_train = Subset(train_full, list(train_idx))
            self.afhq_val = Subset(eval_full, list(val_idx))

        if stage == "test" or stage is None:
            self.afhq_test = self._cached("test", self.transform)

        if stage == "predict":
            self.afhq_predict = self._cached("test", self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.afhq_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.afhq_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.afhq_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.afhq_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = AFHQDataModule(data_dir="/mnt/ai/data")
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"classes={dm.classes}")
    print(f"train batch: x={x.shape}, y={y.shape}, dtype={x.dtype}, "
          f"range=[{x.min():.2f},{x.max():.2f}]")
