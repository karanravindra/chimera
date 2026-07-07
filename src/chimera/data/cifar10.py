"""
CIFAR-10 DataModule for PyTorch Lightning.

Usage:
    dm = CIFAR10DataModule(data_dir="./data", batch_size=128)
    trainer.fit(model, datamodule=dm)
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

import lightning as pl


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.cifar_train: Optional[Dataset] = None
        self.cifar_val: Optional[Dataset] = None
        self.cifar_test: Optional[Dataset] = None
        self.cifar_predict: Optional[Dataset] = None

    def prepare_data(self):
        # download only, called once on a single process
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            n_val = int(len(full_train) * self.val_split)
            n_train = len(full_train) - n_val
            self.cifar_train, self.cifar_val = random_split(
                full_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.cifar_predict = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = CIFAR10DataModule()
    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    print(f"train batch: x={x.shape}, y={y.shape}")
