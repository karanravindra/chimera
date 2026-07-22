"""Unified, mixture-capable Lightning DataModule for compiled text views."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from chimera.data.cache import content_hash
from chimera.data.manifest import CatalogLock
from chimera.data.source import DataFiles
from chimera.tokenizers import BPETokenizer

from .artifacts import (
    DocumentWindowArtifactDataset,
    PackedArtifactDataset,
    ShardedTokenStore,
)
from .catalog import LOCK_PATH, LocalTextView, get_source, get_view, load_rows
from .chat_template import BOS, EOS, SPECIAL_TOKENS
from .compiler import (
    artifact_directory,
    build_descriptor,
    compile_view,
    tokenizer_fingerprint,
)


@dataclass(frozen=True)
class TokenizerSpec:
    mode: str = "train"
    path: Optional[Path] = None
    backend: str = "hf"
    vocab_size: int = 8192
    pretrained_id: str = "LiquidAI/LFM2.5-230M"
    sample_chars: int = 5_000_000

    @classmethod
    def pinned(cls, path: str | Path, *, backend: str = "hf"):
        return cls(mode="pinned", path=Path(path), backend=backend)

    @classmethod
    def pretrained(cls, identifier: str, *, path: str | Path | None = None):
        return cls(
            mode="pretrained",
            path=Path(path) if path else None,
            backend="pretrained",
            pretrained_id=identifier,
        )


@dataclass(frozen=True)
class Packed:
    key: str = "packed"


@dataclass(frozen=True)
class DocumentWindow:
    min_doc_len: Optional[int] = None
    max_windows_per_doc: int = 4
    seed: int = 0
    key: str = "document-window"


@dataclass(frozen=True)
class MixtureSource:
    view: str | LocalTextView
    weight: Optional[float] = None
    max_train_tokens: Optional[int] = 10_000_000
    max_val_tokens: Optional[int] = 1_000_000
    data_files: DataFiles = None
    sampling: Packed | DocumentWindow = field(default_factory=Packed)
    name: Optional[str] = None


@dataclass(frozen=True)
class TextMixtureSpec:
    sources: Sequence[MixtureSource]
    tokenizer: TokenizerSpec
    add_eos: bool = True
    add_bos: bool = False
    eos_token: str = EOS
    bos_token: str = BOS
    shard_tokens: int = 20_000_000
    num_samples: Optional[int] = None


def _tokenizer_path(spec: TextMixtureSpec, data_dir: Path) -> Path:
    tok = spec.tokenizer
    if tok.mode == "pinned":
        if tok.path is None:
            raise ValueError("a pinned tokenizer requires path")
        return tok.path
    if tok.path is not None:
        return tok.path
    locks = CatalogLock.load(LOCK_PATH)
    sources = []
    for mixture_source in spec.sources:
        view = get_view(mixture_source.view)
        if view.exclude_from_tokenizer:
            continue
        if isinstance(view, LocalTextView):
            continue
        source = get_source(view.source)
        lock = locks.require(source.key, source.source.repo)
        sources.append((view.key, lock.revision, mixture_source.data_files))
    key = content_hash(
        {
            "sources": sources,
            "mode": tok.mode,
            "backend": tok.backend,
            "vocab_size": tok.vocab_size,
            "pretrained_id": tok.pretrained_id,
            "sample_chars": tok.sample_chars,
        }
    )
    return data_dir / "text" / "tokenizers" / f"tokenizer-{key}.json"


def _iter_tokenizer_documents(spec: TextMixtureSpec, data_dir: Path):
    eligible = [
        source
        for source in spec.sources
        if not get_view(source.view).exclude_from_tokenizer
    ]
    if not eligible:
        raise ValueError("no mixture source is eligible for tokenizer training")
    per_source = max(1, spec.tokenizer.sample_chars // len(eligible))
    for mixture_source in eligible:
        view = get_view(mixture_source.view)
        rows = load_rows(
            view,
            "train",
            data_dir=data_dir,
            data_files=mixture_source.data_files,
            streaming=True,
        )
        consumed = 0
        for example in view.adapter.iter_examples(rows):
            text = example.text
            if not text:
                continue
            yield text
            consumed += len(text)
            if consumed >= per_source:
                break


def _build_tokenizer(spec: TextMixtureSpec, data_dir: Path):
    path = _tokenizer_path(spec, data_dir)
    tok_spec = spec.tokenizer
    if path.exists():
        backend = "hf" if tok_spec.backend == "pretrained" else tok_spec.backend
        return BPETokenizer.load(path, backend=backend), path
    path.parent.mkdir(parents=True, exist_ok=True)
    if tok_spec.mode == "pinned":
        raise RuntimeError(f"pinned tokenizer does not exist: {path}")
    if tok_spec.mode == "pretrained":
        tokenizer = BPETokenizer.from_pretrained(tok_spec.pretrained_id)
    elif tok_spec.mode == "train":
        tokenizer = BPETokenizer(backend=tok_spec.backend)
        tokenizer.train(
            _iter_tokenizer_documents(spec, data_dir),
            vocab_size=tok_spec.vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )
    else:
        raise ValueError(f"unknown tokenizer mode {tok_spec.mode!r}")
    tokenizer.save(path)
    return tokenizer, path


def _load_tokenizer(spec: TextMixtureSpec, data_dir: Path):
    path = _tokenizer_path(spec, data_dir)
    if not path.exists():
        raise RuntimeError("tokenizer is missing; run prepare_data() before setup()")
    backend = "hf" if spec.tokenizer.backend == "pretrained" else spec.tokenizer.backend
    return BPETokenizer.load(path, backend=backend), path


def _special_id(tokenizer, enabled: bool, token: str) -> Optional[int]:
    if not enabled:
        return None
    token_id = tokenizer._tok.token_to_id(token) if tokenizer._tok is not None else None
    if token_id is None:
        raise ValueError(f"enabled document token {token!r} is missing from tokenizer")
    return token_id


def text_worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return

    def descendants(dataset):
        children = getattr(dataset, "datasets", None)
        if children is None:
            yield dataset
        else:
            for child in children:
                yield from descendants(child)

    for dataset in descendants(info.dataset):
        if isinstance(dataset, DocumentWindowArtifactDataset):
            dataset._gen = torch.Generator().manual_seed(
                (dataset.seed * 1_000_003 + dataset._epoch) * 1_000_003 + worker_id + 1
            )


class TextDataModule(pl.LightningDataModule):
    """Compile and sample one or more locked text views with one tokenizer."""

    def __init__(
        self,
        mixture: TextMixtureSpec,
        *,
        data_dir: str = "./data",
        batch_size: int = 64,
        seq_len: int = 512,
        num_workers: int = 4,
        pin_memory: bool = True,
        verify_artifacts: bool = False,
    ):
        super().__init__()
        if not mixture.sources:
            raise ValueError("a text mixture needs at least one source")
        self.mixture = mixture
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verify_artifacts = verify_artifacts
        self.tokenizer = None
        self.tokenizer_path: Optional[Path] = None
        self.vocab_size: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_datasets = {}
        self.val_datasets = {}
        self.source_train_tokens = {}
        self.source_val_tokens = {}
        self.source_names = self._source_names()
        self._train_sources = []

    def _source_names(self):
        names = []
        for source in self.mixture.sources:
            view = get_view(source.view)
            base = source.name or (
                view.key if isinstance(view, LocalTextView) else view.source
            )
            name, suffix = base, 2
            while name in names:
                name, suffix = f"{base}-{suffix}", suffix + 1
            names.append(name)
        return names

    def _resolve_specials(self):
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_id = _special_id(
            self.tokenizer, self.mixture.add_eos, self.mixture.eos_token
        )
        self.bos_id = _special_id(
            self.tokenizer, self.mixture.add_bos, self.mixture.bos_token
        )
        if any(
            get_view(source.view).objective == "assistant-only"
            for source in self.mixture.sources
        ):
            if self.eos_id is None or self.bos_id is None:
                raise ValueError(
                    "assistant-only views require tokenizer BOS and EOS tokens"
                )

    def prepare_data(self):
        self.tokenizer, self.tokenizer_path = _build_tokenizer(
            self.mixture, self.data_dir
        )
        self._resolve_specials()
        fingerprint = tokenizer_fingerprint(self.tokenizer_path)
        for source in self.mixture.sources:
            compile_view(
                source.view,
                "train",
                tokenizer=self.tokenizer,
                tokenizer_hash=fingerprint,
                data_dir=self.data_dir,
                max_tokens=source.max_train_tokens,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                data_files=source.data_files,
                shard_tokens=self.mixture.shard_tokens,
            )
            compile_view(
                source.view,
                "validation",
                tokenizer=self.tokenizer,
                tokenizer_hash=fingerprint,
                data_dir=self.data_dir,
                max_tokens=source.max_val_tokens,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                data_files=source.data_files,
                shard_tokens=self.mixture.shard_tokens,
            )

    def _artifact_path(self, source: MixtureSource, split: str, fingerprint: str):
        max_tokens = (
            source.max_train_tokens if split == "train" else source.max_val_tokens
        )
        descriptor = build_descriptor(
            source.view,
            split,
            tokenizer_hash=fingerprint,
            max_tokens=max_tokens,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            data_files=source.data_files,
            shard_tokens=self.mixture.shard_tokens,
        )
        return artifact_directory(self.data_dir, descriptor)

    def setup(self, stage: Optional[str] = None):
        del stage
        if self.train_dataset is not None:
            return
        self.tokenizer, self.tokenizer_path = _load_tokenizer(
            self.mixture, self.data_dir
        )
        self._resolve_specials()
        fingerprint = tokenizer_fingerprint(self.tokenizer_path)
        train_datasets, val_datasets = [], []
        for index, (name, source) in enumerate(
            zip(self.source_names, self.mixture.sources)
        ):
            train_store = ShardedTokenStore(
                self._artifact_path(source, "train", fingerprint),
                verify=self.verify_artifacts,
            )
            val_store = ShardedTokenStore(
                self._artifact_path(source, "validation", fingerprint),
                verify=self.verify_artifacts,
            )
            if isinstance(source.sampling, DocumentWindow):
                train_dataset = DocumentWindowArtifactDataset(
                    train_store,
                    self.seq_len,
                    min_doc_len=source.sampling.min_doc_len,
                    max_windows_per_doc=source.sampling.max_windows_per_doc,
                    seed=source.sampling.seed + index,
                )
            else:
                train_dataset = PackedArtifactDataset(train_store, self.seq_len)
            val_dataset = PackedArtifactDataset(val_store, self.seq_len)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            self.train_datasets[name] = train_dataset
            self.val_datasets[name] = val_dataset
            self.source_train_tokens[name] = len(train_store)
            self.source_val_tokens[name] = len(val_store)
        self._train_sources = train_datasets
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    def set_epoch(self, epoch: int) -> None:
        for dataset in self._train_sources:
            set_epoch = getattr(dataset, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(epoch)

    def _sampler(self):
        if not any(source.weight is not None for source in self.mixture.sources):
            return None
        lengths = [len(dataset) for dataset in self._train_sources]
        raw = [
            source.weight if source.weight is not None else float(length)
            for source, length in zip(self.mixture.sources, lengths)
        ]
        if any(weight < 0 for weight in raw) or sum(raw) <= 0:
            raise ValueError("mixture weights must be non-negative with a positive sum")
        weights = []
        for weight, length in zip(raw, lengths):
            if length == 0 and weight > 0:
                raise ValueError(
                    "a positively weighted source produced no training items"
                )
            weights.extend([weight / max(1, length)] * length)
        num_samples = self.mixture.num_samples or sum(lengths)
        return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

    def train_dataloader(self):
        sampler = self._sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            worker_init_fn=text_worker_init_fn,
        )

    def _val_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return self._val_loader(self.val_dataset)

    def val_dataloaders_by_source(self):
        return {
            name: self._val_loader(dataset)
            for name, dataset in self.val_datasets.items()
        }

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)
