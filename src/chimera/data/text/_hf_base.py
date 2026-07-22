"""
Shared base for the Hugging Face corpus DataModules.

Both the pretrain text loader (:class:`chimera.data.hf_text.HFTextDataModule`)
and the SFT loader (:class:`chimera.data.chat_sft.ChatSFTDataModule`) do the same
thing in outline: download an HF dataset, tokenize each split into a flat on-disk
cache keyed on a tokenizer fingerprint + a source fingerprint, and serve
non-overlapping ``(input, target)`` chunks. This base holds everything that is
identical between them — dataset loading, split carving, fingerprinting, cache
round-trip, and the ``prepare_data``/``setup``/dataloader scaffolding — and leaves
the parts that genuinely differ to a handful of hooks:

- :meth:`_prepare_tokenizer` / :meth:`_setup_tokenizer` — obtain the tokenizer and
  resolve ``vocab_size`` / ``eos_id`` / ``bos_id`` (trained-or-loaded vs.
  fixed-from-disk).
- :meth:`_tokenizer_fingerprint` — the content hash the ids caches are bound to.
- :meth:`_tokenize_split` — turn a loaded split into the cache payload (a flat ids
  tensor vs. a stacked ``[ids, labels]`` tensor).
- :meth:`_cache_path` — the per-split cache filename scheme (``ids_v2_*`` vs.
  ``sft_v2_*``).
- :meth:`_make_datasets` — wrap the cached payloads as the served Datasets.
- :meth:`_fingerprint_extra` / :meth:`_renderer_methods` — the subclass-specific
  fields folded into the source fingerprint.

This module imports ``lightning`` and therefore must never be imported at
``chimera.data`` package import time — the lazy ``__getattr__`` registry in
``__init__`` keeps that contract.
"""

import hashlib
import inspect
import json
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, Dataset

import lightning as pl

from .datasets import load_cached_ids, save_cached_ids


class _HFCorpusBase(pl.LightningDataModule):
    HF_REPO: str
    DIR_NAME: str
    TRAIN_SPLIT: str = "train"
    VAL_SPLIT: str = "validation"
    UNIT: str = "doc"

    # HF dataset config passed as ``name=`` (e.g. FineWeb-Edu's "sample-10BT").
    CONFIG_NAME: Optional[str] = None
    # Repo-relative parquet file(s) to build from, passed as ``data_files=``.
    # Bounds the download to a few shards instead of pulling a whole config.
    DATA_FILES = None
    # Fraction of the train split carved off its head for validation, for
    # corpora with no native validation split. None => use VAL_SPLIT verbatim.
    VAL_FROM_TRAIN: Optional[float] = None

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        seq_len: int = 512,
        max_train_tokens: Optional[int] = 10_000_000,
        max_val_tokens: Optional[int] = 1_000_000,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_train_tokens = max_train_tokens
        self.max_val_tokens = max_val_tokens
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = None
        self.vocab_size: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    # -- identity / paths ----------------------------------------------------
    @property
    def name(self) -> str:
        """Short per-source key (mixture labels, per-source metrics)."""
        return self.DIR_NAME

    @property
    def _dir(self) -> Path:
        return self.data_dir / self.DIR_NAME

    # -- HF loading + split carving ------------------------------------------
    def _hf_split(self, split: str) -> str:
        """Translate a logical split name to the HF split spec to load.

        Logical names (:attr:`TRAIN_SPLIT` / :attr:`VAL_SPLIT`) are what the ids
        caches and progress labels are keyed on. When :attr:`VAL_FROM_TRAIN` is
        set the corpus has no native validation split, so validation is carved
        off the head of the train split and train takes the remainder — while
        the cache/label names stay clean.
        """
        if self.VAL_FROM_TRAIN is None:
            return split
        pct = max(1, round(self.VAL_FROM_TRAIN * 100))
        if split == self.VAL_SPLIT:
            return f"{self.TRAIN_SPLIT}[:{pct}%]"
        if split == self.TRAIN_SPLIT:
            return f"{self.TRAIN_SPLIT}[{pct}%:]"
        return split

    def _load_dataset(self, split: str):
        from datasets import load_dataset

        kwargs = {"cache_dir": str(self.data_dir / "hf_cache")}
        if self.DATA_FILES is not None:
            kwargs["data_files"] = self.DATA_FILES
            # a shard subset can't match the repo's declared full-split sizes
            kwargs["verification_mode"] = "no_checks"
        if self.CONFIG_NAME is not None:
            kwargs["name"] = self.CONFIG_NAME
        return load_dataset(self.HF_REPO, split=self._hf_split(split), **kwargs)

    # -- fingerprints --------------------------------------------------------
    def _dataset_fingerprint(self) -> str:
        """Hash every source/rendering option that can change token ids."""
        payload = {
            "schema": 2,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "repo": getattr(self, "HF_REPO", None),
            "config": self.CONFIG_NAME,
            "data_files": self.DATA_FILES,
            "train_split": self._hf_split(self.TRAIN_SPLIT),
            "val_split": self._hf_split(self.VAL_SPLIT),
            "renderer": self._renderer_fingerprint(),
            **self._fingerprint_extra(),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.blake2b(encoded, digest_size=6).hexdigest()

    def _fingerprint_extra(self) -> dict:
        """Subclass-specific fields folded into the source fingerprint."""
        return {}

    def _renderer_fingerprint(self) -> str:
        """Hash subclass text-rendering hooks so code changes invalidate caches."""
        parts = []
        for method in self._renderer_methods():
            try:
                parts.append(inspect.getsource(method))
            except (OSError, TypeError):
                parts.append(f"{method.__module__}.{method.__qualname__}")
        return hashlib.blake2b("\n".join(parts).encode(), digest_size=6).hexdigest()

    def _renderer_methods(self):
        """Callables whose source defines this module's rendering."""
        return ()

    # -- cache build / round-trip --------------------------------------------
    def prepare_data(self):
        # Lightning runs prepare_data on one process. All downloading, tokenizer
        # work, tokenization, and cache writes therefore belong here — not setup,
        # which runs independently on every DDP rank.
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prepare_tokenizer()
        self._build_split(self.TRAIN_SPLIT, self.max_train_tokens)
        self._build_split(self.VAL_SPLIT, self.max_val_tokens)

    def _build_split(self, split: str, max_tokens: Optional[int]):
        """Return the cached (or freshly tokenized) payload for ``split``."""
        path = self._cache_path(split, max_tokens)
        cached = load_cached_ids(path)
        if cached is not None:
            return cached

        ds = self._load_dataset(split)
        data = self._tokenize_split(ds, split, max_tokens)
        save_cached_ids(
            path,
            data,
            metadata={
                "source": self._dataset_fingerprint(),
                "tokenizer": self._tokenizer_fingerprint(),
                "split": split,
                "max_tokens": max_tokens,
            },
        )
        return data

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return
        self._setup_tokenizer()
        train = load_cached_ids(
            self._cache_path(self.TRAIN_SPLIT, self.max_train_tokens)
        )
        val = load_cached_ids(self._cache_path(self.VAL_SPLIT, self.max_val_tokens))
        if train is None or val is None:
            raise RuntimeError(
                "token caches are missing; run prepare_data() once before setup()"
            )
        self.train_dataset, self.val_dataset = self._make_datasets(train, val)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    # -- dataloaders ---------------------------------------------------------
    def _dl(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._dl(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_dataset, shuffle=False)

    # -- hooks subclasses implement ------------------------------------------
    def _prepare_tokenizer(self):
        """Obtain the tokenizer and set vocab_size / eos_id / bos_id (build side)."""
        raise NotImplementedError

    def _setup_tokenizer(self):
        """Ensure the tokenizer is loaded and ids resolved (read side)."""
        raise NotImplementedError

    def _tokenizer_fingerprint(self) -> str:
        """Content hash of the tokenizer in use; ids caches bind to it."""
        raise NotImplementedError

    def _cache_path(self, split: str, max_tokens: Optional[int]) -> Path:
        """Per-split cache filename for this config."""
        raise NotImplementedError

    def _tokenize_split(self, ds, split: str, max_tokens: Optional[int]):
        """Turn a loaded split into the payload persisted by :meth:`_build_split`."""
        raise NotImplementedError

    def _make_datasets(self, train_payload, val_payload):
        """Wrap cached payloads as ``(train_dataset, val_dataset)``."""
        raise NotImplementedError
