"""
CelebA-HQ DataModule for PyTorch Lightning.

CelebA-HQ, as packaged by StarGAN-v2, is a high-resolution face dataset split into
``train`` and ``val`` with two classes (``female``, ``male``). ``prepare_data``
downloads and extracts the official zip from the StarGAN-v2 mirror into
``data_dir/celeba_hq/{train,val}/<class>/``, and the module serves ``(image, label)``
batches via a torchvision ``ImageFolder``.

Usage:
    dm = CelebAHQDataModule(data_dir="./data", batch_size=32, image_size=256)
    trainer.fit(model, datamodule=dm)
"""

import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

import lightning as pl

CELEBA_HQ_URL = "https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=1"


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

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.classes: list[str] = []

        self.celeba_train: Optional[Dataset] = None
        self.celeba_val: Optional[Dataset] = None
        self.celeba_test: Optional[Dataset] = None
        self.celeba_predict: Optional[Dataset] = None

    @property
    def _dir(self) -> Path:
        return self.data_dir / "celeba_hq"

    def prepare_data(self):
        # download + unzip only, called once on a single process
        if self._dir.exists():
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir / "celeba_hq.zip"
        if not zip_path.exists():
            print(f"Downloading CelebA-HQ from {CELEBA_HQ_URL} ...")
            req = urllib.request.Request(
                CELEBA_HQ_URL, headers={"User-Agent": "chimera"}
            )
            with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_train = ImageFolder(self._dir / "train", transform=self.transform)
            self.classes = full_train.classes
            n_val = int(len(full_train) * self.val_split)
            n_train = len(full_train) - n_val
            self.celeba_train, self.celeba_val = random_split(
                full_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.celeba_test = ImageFolder(self._dir / "val", transform=self.transform)
            self.classes = self.celeba_test.classes

        if stage == "predict":
            self.celeba_predict = ImageFolder(
                self._dir / "val", transform=self.transform
            )

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
    dm = CelebAHQDataModule()
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"classes={dm.classes}")
    print(f"train batch: x={x.shape}, y={y.shape}")
