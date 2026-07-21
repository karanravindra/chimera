"""
ContextMixDataModule — blend a "broad short" pool and a "long coherent" pool for
context-length extension (the 2k / 4k / 8k stages).

Long-range ability only comes from documents that are *actually* long: document
masking resets attention and RoPE positions at every EOS, so packing unrelated
short documents into a long tensor trains nothing at long range. This module
therefore keeps two pools and mixes them BY TOKENS:

- **short pool** — a normal :class:`ConcatTextDataModule` over the broad sources
  (FineWeb / Cosmopedia / stories / QA), served as packed
  :class:`~chimera.data._text.TokenDataset` windows (may cross doc boundaries;
  that's fine — this pool is for short-context retention).
- **long pool** — a :class:`ConcatTextDataModule` over long-document sources
  (Wikipedia, Stack Exchange, a long-FineWeb slice), served as
  :class:`~chimera.data._text.WindowSampledDataset` windows: a random contiguous
  slice of a single long document, so every position is mutually visible.

Both pools produce items of exactly ``ctx`` tokens, so **token share == item
share** and the per-stage mix (e.g. 35% short / 65% long at 2k) is just a
:class:`~torch.utils.data.WeightedRandomSampler` weight over the concatenation.

Both pools MUST pin the same frozen tokenizer (``tokenizer_path``) — asserted —
so the long and short ids caches key on one vocabulary fingerprint and stay
comparable to the 512-token base.

Usage:
    dm = ContextMixDataModule(short_pool, long_pool, ctx=2048,
                              short_share=0.35, long_share=0.65, batch_size=32)
    dm.prepare_data(); dm.setup("fit")
"""

from typing import Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

import lightning as pl

from ._text import TokenDataset, WindowSampledDataset, window_worker_init_fn
from .concat_text import ConcatTextDataModule


class ContextMixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        short_pool: ConcatTextDataModule,
        long_pool: ConcatTextDataModule,
        ctx: int,
        short_share: float,
        long_share: float,
        batch_size: int,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        min_doc_len: Optional[int] = None,
        max_windows_per_doc: int = 4,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        assert short_pool.tokenizer_path is not None, (
            "short_pool must pin a frozen tokenizer (tokenizer_path) for context stages"
        )
        assert long_pool.tokenizer_path is not None, (
            "long_pool must pin a frozen tokenizer (tokenizer_path) for context stages"
        )
        assert short_pool.tokenizer_path == long_pool.tokenizer_path, (
            "short and long pools must pin the SAME tokenizer so the two ids caches "
            f"share one vocab ({short_pool.tokenizer_path} != {long_pool.tokenizer_path})"
        )
        assert short_pool.seq_len == long_pool.seq_len == ctx, (
            f"both pools must be built at seq_len == ctx ({ctx}); got "
            f"short={short_pool.seq_len} long={long_pool.seq_len}"
        )
        total = short_share + long_share
        assert total > 0, "shares must be positive"
        self.short_pool = short_pool
        self.long_pool = long_pool
        self.ctx = ctx
        # normalize so callers can pass either fractions or raw weights
        self.short_share = short_share / total
        self.long_share = long_share / total
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else short_pool.num_workers
        )
        self.pin_memory = (
            pin_memory if pin_memory is not None else short_pool.pin_memory
        )
        self.min_doc_len = min_doc_len
        self.max_windows_per_doc = max_windows_per_doc
        self.num_samples = num_samples
        self.seed = seed

        self.tokenizer = None
        self.vocab_size: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None
        self.short_dataset: Optional[TokenDataset] = None
        self.long_dataset: Optional[WindowSampledDataset] = None
        self.train_dataset: Optional[ConcatDataset] = None

    def prepare_data(self):
        self.short_pool.prepare_data()
        self.long_pool.prepare_data()

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return
        self.short_pool.setup(stage)
        self.long_pool.setup(stage)

        # both pools pin the same vocab, so these agree
        self.tokenizer = self.short_pool.tokenizer
        self.vocab_size = self.short_pool.vocab_size
        self.eos_id = self.short_pool.eos_id
        self.bos_id = self.short_pool.bos_id
        assert self.eos_id is not None, (
            "context stages need an EOS to recover document boundaries; the pinned "
            "tokenizer must define it (hf backend with the reserved specials)"
        )

        # short pool: packed windows, already built at ctx by ConcatTextDataModule
        self.short_dataset = self.short_pool.train_dataset
        # long pool: window-sample single documents from the concatenated stream
        self.long_dataset = WindowSampledDataset(
            self.long_pool.train_dataset.data,
            seq_len=self.ctx,
            eos_id=self.eos_id,
            bos_id=self.bos_id,
            min_doc_len=self.min_doc_len,
            max_windows_per_doc=self.max_windows_per_doc,
            seed=self.seed,
        )
        assert len(self.long_dataset) > 0, (
            "long pool has no documents >= min_doc_len; check the long sources / "
            f"min_doc_len ({self.min_doc_len}) at ctx {self.ctx}"
        )
        self.train_dataset = ConcatDataset([self.short_dataset, self.long_dataset])

    def set_epoch(self, epoch: int) -> None:
        """Resample long-window offsets for a new epoch."""
        if self.long_dataset is not None:
            self.long_dataset.set_epoch(epoch)

    def _sampler(self) -> WeightedRandomSampler:
        n_short, n_long = len(self.short_dataset), len(self.long_dataset)
        # Per-item weight so P(pick from a pool) == that pool's target token share
        # (every item is exactly ctx tokens, so token share == item share).
        w = [self.short_share / max(1, n_short)] * n_short + [
            self.long_share / max(1, n_long)
        ] * n_long
        num_samples = self.num_samples or (n_short + n_long)
        return WeightedRandomSampler(w, num_samples=num_samples, replacement=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self._sampler(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            worker_init_fn=window_worker_init_fn,
        )

    def val_dataloader(self):
        # Mixture val = the short pool's overall val stream (short-context yardstick);
        # length-banded long-context val is scored separately (see bpb_banded.py).
        return self.short_pool.val_dataloader()

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)
