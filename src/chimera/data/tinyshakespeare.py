"""
TinyShakespeare BPE DataModule for PyTorch Lightning.

TinyShakespeare is the ~1MB concatenation of Shakespeare's works used by
Karpathy's char-rnn. Here it is tokenized with a custom byte-level BPE tokenizer
(:class:`chimera.tokenizers.BPETokenizer`, ``scratch`` or ``hf`` backend) and
served as non-overlapping next-token ``(input, target)`` chunks.

Usage:
    dm = TinyShakespeareDataModule(data_dir="./data", batch_size=64, seq_len=256)
    trainer.fit(model, datamodule=dm)
"""

import urllib.request
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as pl

from chimera.tokenizers import BPETokenizer

from ._text import (
    TokenDataset,
    iter_text_chunks,
    load_cached_ids,
    save_cached_ids,
    tokenize_with_progress,
)

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "refs/heads/master/data/tinyshakespeare/input.txt"
)


class TinyShakespeareDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        seq_len: int = 256,
        val_split: float = 0.1,
        vocab_size: int = 1024,
        tokenizer_backend: str = "scratch",
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.val_split = val_split
        self.vocab_size = vocab_size
        self.tokenizer_backend = tokenizer_backend
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer: Optional[BPETokenizer] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def _dir(self) -> Path:
        return self.data_dir / "tinyshakespeare"

    @property
    def _text_path(self) -> Path:
        return self._dir / "input.txt"

    @property
    def _tokenizer_path(self) -> Path:
        return self._dir / f"tokenizer_{self.tokenizer_backend}.json"

    @property
    def _ids_path(self) -> Path:
        """Path of the cached flat token stream for this exact configuration."""
        hp = self.hparams
        return self._dir / f"ids_{hp.tokenizer_backend}_v{hp.vocab_size}.pt"

    def prepare_data(self):
        # download only, called once on a single process
        if self._text_path.exists():
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading tinyshakespeare from {TINYSHAKESPEARE_URL} ...")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, self._text_path)

    def _load_or_train_tokenizer(self, text: str) -> BPETokenizer:
        if self._tokenizer_path.exists():
            return BPETokenizer.load(
                self._tokenizer_path, backend=self.tokenizer_backend
            )
        tok = BPETokenizer(backend=self.tokenizer_backend)
        tok.train(text, vocab_size=self.vocab_size)
        tok.save(self._tokenizer_path)
        return tok

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        text = self._text_path.read_text(encoding="utf-8")
        self.tokenizer = self._load_or_train_tokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size

        data = load_cached_ids(self._ids_path)
        if data is None:
            ids = tokenize_with_progress(
                self.tokenizer,
                iter_text_chunks(text),
                desc="Tokenizing tinyshakespeare",
                unit="chunk",
            )
            data = torch.tensor(ids, dtype=torch.long)
            save_cached_ids(self._ids_path, data)

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
    dm = TinyShakespeareDataModule()
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}")
    print(f"train batch: x={x.shape}, y={y.shape}")
