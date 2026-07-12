"""
AFHQ (Animal Faces-HQ) DataModule for PyTorch Lightning.

AFHQ is the StarGAN-v2 animal-faces dataset: 512x512 images across three classes
(``cat``, ``dog``, ``wild``), pre-split into ``train`` and ``val``. ``prepare_data``
downloads and extracts the official zip from the StarGAN-v2 mirror into
``data_dir/afhq/{train,val}/<class>/``, and the module serves ``(image, label)``
batches via a torchvision ``ImageFolder``.

Usage:
    dm = AFHQDataModule(data_dir="./data", batch_size=32, image_size=256)
    trainer.fit(model, datamodule=dm)
"""

import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

import lightning as pl

AFHQ_URL = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"


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

        # normalize=True maps pixels to [-1, 1] (the usual GAN/diffusion range);
        # normalize=False keeps them in [0, 1] to match a sigmoid-output decoder
        # and PSNR/SSIM with data_range=1.0 (the autoencoder-project convention).
        norm = (
            [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            if normalize
            else []
        )
        # ``self.transform`` is the deterministic eval view (val/test/predict).
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), *norm]
        )
        # ``train_transform`` adds train-only augmentation, applied ONLY to the
        # train split (see setup) and never to val/test, so reconstruction
        # metrics stay comparable. Two ops:
        #   - RandomResizedCrop: mild scale/aspect jitter (a random crop rescaled
        #     back to image_size), kept gentle (scale >=0.8, near-square ratio)
        #     so faces aren't cropped out or distorted. Replaces the plain Resize.
        #   - RandomHorizontalFlip: near-free same-difficulty extra data.
        if augment:
            geom = [
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            geom = [transforms.Resize((image_size, image_size))]
        self.train_transform = transforms.Compose([*geom, transforms.ToTensor(), *norm])
        self.classes: list[str] = []

        self.afhq_train: Optional[Dataset] = None
        self.afhq_val: Optional[Dataset] = None
        self.afhq_test: Optional[Dataset] = None
        self.afhq_predict: Optional[Dataset] = None

    @property
    def _dir(self) -> Path:
        return self.data_dir / "afhq"

    def prepare_data(self):
        # download + unzip only, called once on a single process
        if self._dir.exists():
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir / "afhq.zip"
        if not zip_path.exists():
            print(f"Downloading AFHQ from {AFHQ_URL} ...")
            req = urllib.request.Request(AFHQ_URL, headers={"User-Agent": "chimera"})
            with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.data_dir)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Two views of the same folder: the train split gets augmentation,
            # the val split stays deterministic. random_split gives Subsets that
            # share their parent's transform, so we split the *indices* (seeded)
            # and wrap each half in a Subset of the appropriately-transformed
            # ImageFolder -- otherwise augmentation would leak into validation.
            train_full = ImageFolder(self._dir / "train", transform=self.train_transform)
            eval_full = ImageFolder(self._dir / "train", transform=self.transform)
            self.classes = train_full.classes
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
            self.afhq_test = ImageFolder(self._dir / "val", transform=self.transform)
            self.classes = self.afhq_test.classes

        if stage == "predict":
            self.afhq_predict = ImageFolder(self._dir / "val", transform=self.transform)

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
    dm = AFHQDataModule()
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"classes={dm.classes}")
    print(f"train batch: x={x.shape}, y={y.shape}")
