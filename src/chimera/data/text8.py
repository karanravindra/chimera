"""
text8 DataModule for PyTorch Lightning.

text8 is the first 100M characters of a cleaned English Wikipedia dump,
lowercased to 27 symbols (``a``-``z`` and space). It is conventionally split
into 90M/5M/5M characters for train/val/test.

The Kaggle mirror (https://www.kaggle.com/datasets/yorkyong/text8-zip) ships the
same ``text8.zip``. Kaggle needs API credentials, so ``prepare_data`` downloads
the identical file from its canonical mirror instead. To use the Kaggle copy,
drop ``text8`` (or ``text8.zip``) into ``data_dir`` and the download is skipped.

Tokenization is character-level by default (``tokenizer_backend="char"``, the
classic 27-symbol vocabulary). Set ``tokenizer_backend`` to ``"scratch"`` or
``"hf"`` to instead tokenize with a custom byte-level BPE
(:class:`chimera.tokenizers.BPETokenizer`), matching the other text DataModules.

Usage:
    dm = Text8DataModule(data_dir="./data", batch_size=64, seq_len=256)
    trainer.fit(model, datamodule=dm)
"""

import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as pl

from chimera.tokenizers import BPETokenizer

from ._text import TokenDataset

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


class Text8DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        seq_len: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_size: int = 90_000_000,
        val_size: int = 5_000_000,
        tokenizer_backend: str = "char",
        vocab_size: int = 8192,
        pretrained_id: str = "LiquidAI/LFM2.5-230M",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.val_size = val_size
        self.tokenizer_backend = tokenizer_backend
        self.vocab_size = vocab_size
        self.pretrained_id = pretrained_id

        # char-level lookup, populated in setup() when backend == "char"
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []
        # BPE tokenizer, populated in setup() when backend in {"scratch", "hf"}
        self.tokenizer: Optional[BPETokenizer] = None

        self.text8_train: Optional[Dataset] = None
        self.text8_val: Optional[Dataset] = None
        self.text8_test: Optional[Dataset] = None

    @property
    def _text8_path(self) -> Path:
        return self.data_dir / "text8"

    @property
    def _tokenizer_path(self) -> Path:
        return self.data_dir / f"text8_tokenizer_{self.tokenizer_backend}.json"

    def prepare_data(self):
        # download + unzip only, called once on a single process
        if self._text8_path.exists():
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir / "text8.zip"
        if not zip_path.exists():
            print(f"Downloading text8 from {TEXT8_URL} ...")
            urllib.request.urlretrieve(TEXT8_URL, zip_path)

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.data_dir)

    def _encode_char(self, raw: bytes) -> torch.Tensor:
        # text8 is pure ASCII, so byte values map directly to characters.
        arr = np.frombuffer(raw, dtype=np.uint8)
        chars = sorted(set(arr.tolist()))
        self.itos = [chr(b) for b in chars]
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

        lut = np.zeros(256, dtype=np.uint8)
        for i, b in enumerate(chars):
            lut[b] = i
        return torch.from_numpy(lut[arr].copy())

    def _encode_bpe(self, raw: bytes) -> torch.Tensor:
        text = raw.decode("utf-8", errors="replace")
        if self._tokenizer_path.exists():
            self.tokenizer = BPETokenizer.load(
                self._tokenizer_path, backend=self.tokenizer_backend
            )
        elif self.tokenizer_backend == "pretrained":
            # fixed tokenizer from the Hub (e.g. LiquidAI/LFM2.5-230M); no training
            self.tokenizer = BPETokenizer.from_pretrained(self.pretrained_id)
            self.tokenizer.save(self._tokenizer_path)
        else:
            self.tokenizer = BPETokenizer(backend=self.tokenizer_backend)
            # train on the train split only to avoid leaking val/test statistics
            self.tokenizer.train(text[: self.train_size], vocab_size=self.vocab_size)
            self.tokenizer.save(self._tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size
        return torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

    def setup(self, stage: Optional[str] = None):
        if self.text8_train is not None:
            return

        raw = self._text8_path.read_bytes()
        char_len = len(raw)  # text8 is ASCII, so byte count == character count
        if self.tokenizer_backend == "char":
            data = self._encode_char(raw)
        else:
            data = self._encode_bpe(raw)

        # train_size / val_size are expressed in characters; scale them to the
        # actual token stream so subword tokenizers (which emit far fewer tokens
        # than characters) still split proportionally instead of leaving val/test
        # empty. For the char backend len(data) == char_len, so this is exact.
        n = len(data)
        train_end = min(n, int(n * self.train_size / char_len))
        val_end = min(n, train_end + int(n * self.val_size / char_len))
        self.text8_train = TokenDataset(data[:train_end], self.seq_len)
        self.text8_val = TokenDataset(data[train_end:val_end], self.seq_len)
        self.text8_test = TokenDataset(data[val_end:], self.seq_len)

    def decode(self, ids) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)

    def train_dataloader(self):
        return DataLoader(
            self.text8_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.text8_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.text8_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = Text8DataModule()
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}")
    print(f"train batch: x={x.shape}, y={y.shape}")
