"""Version-3 sharded, mmap-friendly token artifacts."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from chimera.data.cache import atomic_json_save, atomic_torch_save, file_hash


ARTIFACT_VERSION = 3
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ArtifactShard:
    file: str
    tokens: int
    documents: int
    checksum: str


@dataclass(frozen=True)
class TextArtifactManifest:
    build: dict
    dtype: str
    stores_labels: bool
    tokens: int
    documents: int
    shards: tuple[ArtifactShard, ...]
    version: int = ARTIFACT_VERSION

    @classmethod
    def load(cls, path: Path, *, verify: bool = False):
        payload = json.loads(path.read_text())
        if payload.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"unsupported text artifact version in {path}")
        manifest = cls(
            build=payload["build"],
            dtype=payload["dtype"],
            stores_labels=payload["stores_labels"],
            tokens=payload["tokens"],
            documents=payload["documents"],
            shards=tuple(ArtifactShard(**shard) for shard in payload["shards"]),
            version=payload["version"],
        )
        if verify:
            for shard in manifest.shards:
                shard_path = path.parent / shard.file
                if not shard_path.exists():
                    raise RuntimeError(f"missing artifact shard {shard_path}")
                if file_hash(shard_path) != shard.checksum:
                    raise RuntimeError(f"checksum mismatch for {shard_path}")
        return manifest

    def save(self, path: Path) -> None:
        atomic_json_save(
            {
                "version": self.version,
                "build": self.build,
                "dtype": self.dtype,
                "stores_labels": self.stores_labels,
                "tokens": self.tokens,
                "documents": self.documents,
                "shards": [shard.__dict__ for shard in self.shards],
            },
            path,
        )


class TextArtifactWriter:
    """Write document-aligned shards and publish the manifest last."""

    def __init__(
        self,
        directory: Path,
        *,
        build: dict,
        dtype: torch.dtype,
        stores_labels: bool,
        shard_tokens: int = 20_000_000,
    ):
        self.directory = directory
        self.build = build
        self.dtype = dtype
        self.stores_labels = stores_labels
        self.shard_tokens = shard_tokens
        self._ids: list[int] = []
        self._labels: list[int] = []
        self._offsets: list[tuple[int, int]] = []
        self._shards: list[ArtifactShard] = []
        self._tokens = 0
        self._documents = 0

    @property
    def tokens(self) -> int:
        return self._tokens + len(self._ids)

    def add_document(self, ids: list[int], labels: Optional[list[int]] = None) -> None:
        if not ids:
            return
        if self.stores_labels:
            if labels is None or len(ids) != len(labels):
                raise ValueError("labeled artifacts require one label per token")
        elif labels is not None:
            raise ValueError("unlabeled artifacts cannot accept labels")
        if self._ids and len(self._ids) + len(ids) > self.shard_tokens:
            self._flush()
        start = len(self._ids)
        self._ids.extend(ids)
        if labels is not None:
            self._labels.extend(labels)
        self._offsets.append((start, len(self._ids)))
        self._documents += 1
        if len(self._ids) >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self._ids:
            return
        index = len(self._shards)
        path = self.directory / f"shard-{index:05d}.pt"
        payload = {
            "version": ARTIFACT_VERSION,
            "ids": torch.tensor(self._ids, dtype=self.dtype),
            "document_offsets": torch.tensor(self._offsets, dtype=torch.int64),
        }
        if self.stores_labels:
            payload["labels"] = torch.tensor(self._labels, dtype=torch.int32)
        atomic_torch_save(payload, path)
        self._shards.append(
            ArtifactShard(
                file=path.name,
                tokens=len(self._ids),
                documents=len(self._offsets),
                checksum=file_hash(path),
            )
        )
        self._tokens += len(self._ids)
        self._ids.clear()
        self._labels.clear()
        self._offsets.clear()

    def finish(self) -> TextArtifactManifest:
        self._flush()
        manifest = TextArtifactManifest(
            build=self.build,
            dtype=str(self.dtype).removeprefix("torch."),
            stores_labels=self.stores_labels,
            tokens=self._tokens,
            documents=self._documents,
            shards=tuple(self._shards),
        )
        manifest.save(self.directory / MANIFEST_NAME)
        return manifest


class ShardedTokenStore:
    """Read a v3 artifact as one virtual token stream without corpus copies."""

    def __init__(self, directory: Path, *, verify: bool = False):
        self.directory = directory
        self.manifest = TextArtifactManifest.load(
            directory / MANIFEST_NAME, verify=verify
        )
        self._payloads = [
            torch.load(
                directory / shard.file,
                mmap=True,
                map_location="cpu",
                weights_only=True,
            )
            for shard in self.manifest.shards
        ]
        self._starts: list[int] = []
        total = 0
        for shard in self.manifest.shards:
            self._starts.append(total)
            total += shard.tokens

    def __len__(self) -> int:
        return self.manifest.tokens

    def slice(self, start: int, end: int, field: str = "ids") -> torch.Tensor:
        if start < 0 or end < start or end > len(self):
            raise IndexError(
                f"invalid token slice [{start}:{end}] for {len(self)} tokens"
            )
        if start == end:
            dtype = torch.int32 if field == "labels" else torch.int64
            return torch.empty(0, dtype=dtype)
        pieces = []
        cursor = start
        while cursor < end:
            shard_index = bisect.bisect_right(self._starts, cursor) - 1
            local = cursor - self._starts[shard_index]
            tensor = self._payloads[shard_index][field]
            take = min(end - cursor, len(tensor) - local)
            pieces.append(tensor[local : local + take])
            cursor += take
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    @property
    def document_spans(self) -> tuple[tuple[int, int], ...]:
        spans = []
        for base, payload in zip(self._starts, self._payloads):
            spans.extend(
                (base + int(start), base + int(end))
                for start, end in payload["document_offsets"].tolist()
            )
        return tuple(spans)


class PackedArtifactDataset(Dataset):
    def __init__(self, store: ShardedTokenStore, seq_len: int):
        self.store = store
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.store) - 1) // self.seq_len)

    def __getitem__(self, index: int):
        start = index * self.seq_len
        ids = self.store.slice(start, start + self.seq_len + 1, "ids").long()
        x = ids[:-1]
        if self.store.manifest.stores_labels:
            y = self.store.slice(start + 1, start + self.seq_len + 1, "labels").long()
        else:
            y = ids[1:]
        return x, y


class DocumentWindowArtifactDataset(Dataset):
    """Random windows sampled from explicit document spans."""

    def __init__(
        self,
        store: ShardedTokenStore,
        seq_len: int,
        *,
        min_doc_len: int | None = None,
        max_windows_per_doc: int = 4,
        seed: int = 0,
    ):
        self.store = store
        self.seq_len = seq_len
        self.min_doc_len = max(seq_len + 1, min_doc_len or 0)
        self.max_windows_per_doc = max_windows_per_doc
        self.seed = seed
        self._epoch = 0
        self._gen = torch.Generator().manual_seed(seed)
        self._spans = []
        self._cum = []
        total = 0
        for start, end in store.document_spans:
            length = end - start
            if length < self.min_doc_len:
                continue
            self._spans.append((start, end))
            total += max(1, min(max_windows_per_doc, length // (seq_len + 1)))
            self._cum.append(total)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._gen = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    def __getitem__(self, index: int):
        doc = bisect.bisect_right(self._cum, index)
        start, end = self._spans[doc]
        high = end - self.seq_len - 1
        offset = (
            start
            if high <= start
            else int(torch.randint(start, high + 1, (1,), generator=self._gen).item())
        )
        ids = self.store.slice(offset, offset + self.seq_len + 1, "ids").long()
        x = ids[:-1]
        if self.store.manifest.stores_labels:
            y = self.store.slice(offset + 1, offset + self.seq_len + 1, "labels").long()
        else:
            y = ids[1:]
        return x, y
