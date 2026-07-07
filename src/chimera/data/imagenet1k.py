"""
ImageNet-1k DataModule for PyTorch Lightning.

Wraps the gated ``ILSVRC/imagenet-1k`` Hugging Face dataset (ILSVRC 2012:
1,281,167 train / 50,000 validation images, 1,000 classes, ~155GB) as a map-style
torch dataset yielding ``(image, label)`` batches.

ACCESS: this dataset is gated. Before use you must (1) accept the terms on
https://huggingface.co/datasets/ILSVRC/imagenet-1k and (2) be authenticated,
either via ``huggingface-cli login`` or by exporting ``HF_TOKEN``. No credentials
are embedded here.

Usage:
    dm = ImageNet1kDataModule(data_dir="./data", batch_size=64, image_size=224)
    trainer.fit(model, datamodule=dm)
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import lightning as pl

HF_REPO = "ILSVRC/imagenet-1k"
NUM_CLASSES = 1000


class _HFImageDataset(Dataset):
    """Adapts a Hugging Face image split into a torch ``(image, label)`` dataset."""

    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        image = row["image"].convert("RGB")
        return self.transform(image), row["label"]


class ImageNet1kDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        image_size: int = 224,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = NUM_CLASSES

        resize = int(image_size * 256 / 224)
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.imagenet_train: Optional[Dataset] = None
        self.imagenet_val: Optional[Dataset] = None

    def _load_split(self, split: str):
        from datasets import load_dataset

        return load_dataset(
            HF_REPO,
            split=split,
            cache_dir=str(self.data_dir / "hf_cache"),
        )

    def prepare_data(self):
        # download + cache only, called once on a single process
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_split("train")
        self._load_split("validation")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.imagenet_train = _HFImageDataset(
                self._load_split("train"), self.train_transform
            )
            self.imagenet_val = _HFImageDataset(
                self._load_split("validation"), self.eval_transform
            )

        if stage == "test" or stage is None:
            self.imagenet_val = _HFImageDataset(
                self._load_split("validation"), self.eval_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.imagenet_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.imagenet_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = ImageNet1kDataModule()
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"num_classes={dm.num_classes}")
    print(f"train batch: x={x.shape}, y={y.shape}")
