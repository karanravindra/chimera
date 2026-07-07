"""
FineWeb-Edu BPE DataModule for PyTorch Lightning.

FineWeb-Edu (``HuggingFaceFW/fineweb-edu``) is a large, high-quality English web
corpus for language-model pretraining. This module downloads a sample config
(default ``sample-10BT``, ~28GB) via the Hugging Face ``datasets`` library, trains
a custom byte-level BPE tokenizer (:class:`chimera.tokenizers.BPETokenizer`), and
serves a bounded token stream as non-overlapping next-token ``(input, target)`` chunks.

Because the full 10BT sample is ~10B tokens, ``max_train_tokens`` caps how many
tokens are materialised into memory (set ``None`` to use every row — expect very
high memory use).

Usage:
    dm = FineWebEduDataModule(data_dir="./data", batch_size=32, seq_len=1024)
    trainer.fit(model, datamodule=dm)
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as pl

from chimera.tokenizers import BPETokenizer

from ._text import TokenDataset

HF_REPO = "HuggingFaceFW/fineweb-edu"


class FineWebEduDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        name: str = "sample-10BT",
        batch_size: int = 32,
        seq_len: int = 1024,
        val_split: float = 0.01,
        vocab_size: int = 32000,
        tokenizer_backend: str = "scratch",
        max_train_tokens: Optional[int] = 10_000_000,
        tokenizer_train_chars: int = 5_000_000,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.val_split = val_split
        self.vocab_size = vocab_size
        self.tokenizer_backend = tokenizer_backend
        self.max_train_tokens = max_train_tokens
        self.tokenizer_train_chars = tokenizer_train_chars
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer: Optional[BPETokenizer] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def _dir(self) -> Path:
        return self.data_dir / "fineweb-edu"

    @property
    def _tokenizer_path(self) -> Path:
        return self._dir / f"tokenizer_{self.name}_{self.tokenizer_backend}.json"

    def _load_dataset(self):
        from datasets import load_dataset

        return load_dataset(
            HF_REPO,
            name=self.name,
            split="train",
            cache_dir=str(self.data_dir / "hf_cache"),
        )

    def prepare_data(self):
        # download + cache only, called once on a single process
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load_dataset()

    def _load_or_train_tokenizer(self, ds) -> BPETokenizer:
        if self._tokenizer_path.exists():
            return BPETokenizer.load(
                self._tokenizer_path, backend=self.tokenizer_backend
            )

        # concatenate rows until we have enough characters to train on
        parts: list[str] = []
        total = 0
        for row in ds:
            text = row["text"]
            parts.append(text)
            total += len(text)
            if total >= self.tokenizer_train_chars:
                break

        tok = BPETokenizer(backend=self.tokenizer_backend)
        tok.train("\n".join(parts), vocab_size=self.vocab_size)
        tok.save(self._tokenizer_path)
        return tok

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        ds = self._load_dataset()
        self.tokenizer = self._load_or_train_tokenizer(ds)
        self.vocab_size = self.tokenizer.vocab_size

        ids: list[int] = []
        for row in ds:
            ids.extend(self.tokenizer.encode(row["text"]))
            if self.max_train_tokens is not None and len(ids) >= self.max_train_tokens:
                ids = ids[: self.max_train_tokens]
                break

        data = torch.tensor(ids, dtype=torch.long)
        n_val = int(len(data) * self.val_split)
        n_train = len(data) - n_val
        self.train_dataset = TokenDataset(data[:n_train], self.seq_len)
        self.val_dataset = TokenDataset(data[n_train:], self.seq_len)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = FineWebEduDataModule()
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}")
    print(f"train batch: x={x.shape}, y={y.shape}")
