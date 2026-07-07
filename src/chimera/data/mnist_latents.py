"""Precomputed-latent DataModule for MNIST.

Encodes MNIST once with a trained autoencoder, standardizes the latents to zero
mean / unit variance per channel, and caches them to a ``.pt`` file. Subsequent
runs load the cache and skip encoding entirely. Serves ``(latent, label)`` pairs
for training a latent generative model (e.g. a rectified-flow MM-DiT).

Mirrors the cache idiom of ``fineweb_edu`` (config-named cache + skip-if-present)
and the split/dataloader shape of ``MNISTDataModule``.
"""

from pathlib import Path
from typing import Optional

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from ._text import load_cached_ids, save_cached_ids


class _LatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, labels: torch.Tensor):
        self.latents = latents
        self.labels = labels

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


class MNISTLatentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        autoencoder=None,
        data_dir: str = "./data",
        batch_size: int = 128,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        encode_batch_size: int = 512,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
        latent_channels: Optional[int] = None,
    ):
        super().__init__()
        # The autoencoder is a live nn.Module, not a hyperparameter.
        self.save_hyperparameters(ignore=["autoencoder"])

        self.autoencoder = autoencoder
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.encode_batch_size = encode_batch_size
        self.device = device
        self.image_size = image_size

        tfms = []
        if image_size is not None:
            tfms.append(transforms.Resize((image_size, image_size)))
        tfms.append(transforms.ToTensor())
        self.transform = transforms.Compose(tfms)

        # Cache filename must be identical whether or not the AE is passed (so a
        # later cache-only reload with autoencoder=None resolves the same path).
        if latent_channels is None:
            latent_channels = getattr(autoencoder, "latent_dim", "na")
        size_tag = f"_s{image_size}" if image_size is not None else ""
        cache_root = Path(cache_dir) if cache_dir else self.data_dir / "mnist_latents"
        self.cache_path = (
            cache_root / f"latents_c{latent_channels}{size_tag}_seed{seed}.pt"
        )

        # Populated in setup().
        self.latent_mean: Optional[torch.Tensor] = None
        self.latent_std: Optional[torch.Tensor] = None
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    @torch.inference_mode()
    def _encode_split(self, train: bool, device) -> tuple[torch.Tensor, torch.Tensor]:
        ds = MNIST(self.data_dir, train=train, transform=self.transform)
        loader = DataLoader(
            ds,
            batch_size=self.encode_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        param_dtype = next(self.autoencoder.parameters()).dtype
        zs, ys = [], []
        for x, y in loader:
            z = self.autoencoder.encode(x.to(device=device, dtype=param_dtype))
            zs.append(z.float().cpu())
            ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

        if load_cached_ids(self.cache_path) is not None:
            return

        if self.autoencoder is None:
            raise ValueError(
                "No cached latents found and no autoencoder provided to encode them."
            )

        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.autoencoder.eval().to(device)

        train_z, train_y = self._encode_split(train=True, device=device)
        test_z, test_y = self._encode_split(train=False, device=device)

        # Per-channel standardization stats from the training latents.
        mean = train_z.mean(dim=(0, 2, 3), keepdim=True)
        std = train_z.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

        save_cached_ids(
            self.cache_path,
            {
                "train_latents": (train_z - mean) / std,
                "train_labels": train_y,
                "test_latents": (test_z - mean) / std,
                "test_labels": test_y,
                "mean": mean,
                "std": std,
            },
        )

    def setup(self, stage: Optional[str] = None):
        cache = load_cached_ids(self.cache_path)
        if cache is None:
            raise RuntimeError(
                f"Latent cache missing at {self.cache_path}; call prepare_data() first."
            )
        self.latent_mean = cache["mean"]
        self.latent_std = cache["std"]

        if stage in ("fit", None):
            full = _LatentDataset(cache["train_latents"], cache["train_labels"])
            n_val = int(len(full) * self.val_split)
            n_train = len(full) - n_val
            self.train_set, self.val_set = random_split(
                full,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage in ("test", None):
            self.test_set = _LatentDataset(cache["test_latents"], cache["test_labels"])

    def _loader(self, ds, shuffle, drop_last=False):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_set, shuffle=False)
