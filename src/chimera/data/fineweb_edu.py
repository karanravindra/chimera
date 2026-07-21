"""
FineWeb-Edu BPE DataModule for PyTorch Lightning.

FineWeb-Edu (``HuggingFaceFW/fineweb-edu``) is a large, high-quality English web
corpus for language-model pretraining. This module downloads a sample config
(default ``sample-10BT``, ~28GB) via the Hugging Face ``datasets`` library,
tokenizes it (a custom byte-level BPE via :class:`chimera.tokenizers.BPETokenizer`,
or a fixed pretrained tokenizer when ``tokenizer_backend="pretrained"``), and
serves a bounded token stream as non-overlapping next-token ``(input, target)`` chunks.

Documents are concatenated into a single stream with an end-of-document token
(``eos_token``, default ``<|endoftext|>``) appended after each one so the model
learns document boundaries rather than attending across unrelated pages. This
requires a tokenizer that defines the token — i.e. ``hf`` or ``pretrained``
backends; the from-scratch byte-level BPE has none, so ``add_eos`` is a no-op there.

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

from ._text import (
    TokenDataset,
    load_cached_ids,
    save_cached_ids,
    tokenize_with_progress,
)

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
        pretrained_id: str = "LiquidAI/LFM2.5-230M",
        add_eos: bool = True,
        eos_token: str = "<|endoftext|>",
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
        self.pretrained_id = pretrained_id
        self.add_eos = add_eos
        self.eos_token = eos_token
        self.max_train_tokens = max_train_tokens
        self.tokenizer_train_chars = tokenizer_train_chars
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer: Optional[BPETokenizer] = None
        # id of the document-separator token, resolved in setup()
        self.eos_id: Optional[int] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def _dir(self) -> Path:
        return self.data_dir / "fineweb-edu"

    @property
    def _tokenizer_path(self) -> Path:
        return self._dir / f"tokenizer_{self.name}_{self.tokenizer_backend}.json"

    @property
    def _ids_path(self) -> Path:
        """Path of the cached flat token stream for this exact configuration."""
        hp = self.hparams
        if hp.tokenizer_backend == "pretrained":
            tok_tag = "pretrained_" + hp.pretrained_id.replace("/", "_")
        else:
            tok_tag = f"{hp.tokenizer_backend}_v{hp.vocab_size}"
        eos = "eos" if hp.add_eos else "noeos"
        cap = "all" if hp.max_train_tokens is None else str(hp.max_train_tokens)
        return self._dir / f"ids_{hp.name}_{tok_tag}_{eos}_{cap}.pt"

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
        # if the tokenized id stream is already cached we never touch the raw
        # dataset again, so skip the (large) download entirely.
        if self._ids_path.exists():
            return
        self._load_dataset()

    def _load_or_train_tokenizer(self) -> BPETokenizer:
        if self._tokenizer_path.exists():
            return BPETokenizer.load(
                self._tokenizer_path, backend=self.tokenizer_backend
            )

        if self.tokenizer_backend == "pretrained":
            # fixed tokenizer from the Hub (e.g. LiquidAI/LFM2.5-230M); no training
            tok = BPETokenizer.from_pretrained(self.pretrained_id)
            tok.save(self._tokenizer_path)
            return tok

        # concatenate rows until we have enough characters to train on
        ds = self._load_dataset()
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

    def _resolve_eos_id(self) -> Optional[int]:
        """Token id used to separate documents, or None if unavailable.

        Only the fast (``hf`` / ``pretrained``) backends carry named special
        tokens; a from-scratch byte-level BPE has no EOS, so ``add_eos`` is a
        no-op there.
        """
        if not self.add_eos:
            return None
        tok = self.tokenizer._tok if self.tokenizer is not None else None
        if tok is None:
            return None
        return tok.token_to_id(self.eos_token)

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        self.tokenizer = self._load_or_train_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_id = self._resolve_eos_id()
        if self.add_eos and self.eos_id is None:
            print(
                f"add_eos=True but tokenizer has no {self.eos_token!r} token "
                f"(backend={self.tokenizer_backend!r}); documents will be "
                "concatenated without a separator."
            )

        # Concatenate documents into one token stream, appending the EOS token
        # after each document so the model learns document boundaries instead of
        # bleeding context across unrelated web pages. The stream is cached to
        # disk so subsequent runs skip both the download and the tokenization.
        data = load_cached_ids(self._ids_path)
        if data is None:
            ds = self._load_dataset()

            # Batched column reads are faster than per-row access on the Arrow
            # dataset; tokenize_with_progress then batches these for encode_batch
            # and stops early once max_train_tokens is reached.
            def texts():
                for batch in ds.iter(batch_size=1024):
                    yield from batch["text"]

            ids = tokenize_with_progress(
                self.tokenizer,
                texts(),
                desc="Tokenizing fineweb-edu",
                total=len(ds),
                unit="doc",
                eos_id=self.eos_id,
                max_tokens=self.max_train_tokens,
            )
            data = ids.long()  # tokenize_with_progress returns an int16 tensor
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
    dm = FineWebEduDataModule()
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}")
    print(f"train batch: x={x.shape}, y={y.shape}")
