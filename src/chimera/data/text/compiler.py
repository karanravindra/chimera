"""Compile locked text views into version-3 token artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import re
from typing import Optional

import torch
from tqdm.auto import tqdm

from chimera.data.cache import content_hash
from chimera.data.manifest import CatalogLock
from chimera.data.source import DataFiles

from .artifacts import MANIFEST_NAME, TextArtifactManifest, TextArtifactWriter
from .catalog import LOCK_PATH, LocalTextView, get_source, get_view, load_rows
from .objectives import OBJECTIVES


def tokenizer_fingerprint(path: Path) -> str:
    return content_hash(path.read_bytes(), digest_size=6)


def _adapter_payload(adapter) -> object:
    return asdict(adapter) if is_dataclass(adapter) else repr(adapter)


def _validate_selection(view_key: str | LocalTextView, data_files: DataFiles) -> None:
    view = get_view(view_key)
    if isinstance(view, LocalTextView):
        if data_files is not None:
            raise ValueError("data_files cannot override a local text view")
        return
    if data_files is None or isinstance(data_files, str):
        return
    source = get_source(view.source)
    lock = CatalogLock.load(LOCK_PATH).require(source.key, source.source.repo)
    allowed = set(lock.files)
    if allowed:
        unknown = sorted(set(data_files) - allowed)
        if unknown:
            raise ValueError(
                f"source selection for {source.key!r} contains unlocked files: {unknown}"
            )


def build_descriptor(
    view_key: str | LocalTextView,
    logical_split: str,
    *,
    tokenizer_hash: str,
    max_tokens: Optional[int],
    bos_id: Optional[int],
    eos_id: Optional[int],
    data_files: DataFiles,
    shard_tokens: int,
) -> dict:
    view = get_view(view_key)
    if isinstance(view, LocalTextView):
        files = tuple(sorted(view.directory.glob(view.glob)))
        if not files:
            raise FileNotFoundError(
                f"no files matching {view.glob!r} in {view.directory}"
            )
        source_payload = {
            "source": "local",
            "directory": str(view.directory.resolve()),
            "files": [
                {
                    "name": path.name,
                    "content": content_hash(path.read_bytes()),
                }
                for path in files
            ],
            "repeat": view.repeat,
            "split": logical_split,
        }
    else:
        source = get_source(view.source)
        lock = CatalogLock.load(LOCK_PATH).require(source.key, source.source.repo)
        selection = source.source.data_files if data_files is None else data_files
        source_payload = {
            "source": source.key,
            "repo": source.source.repo,
            "revision": lock.revision,
            "data_files": selection,
            "split": (view.splits or source.splits).resolve(logical_split),
        }
    return {
        "schema": 3,
        "view": view.key,
        **source_payload,
        "adapter": {"key": view.adapter.key, "config": _adapter_payload(view.adapter)},
        "objective": view.objective,
        "tokenizer": tokenizer_hash,
        "max_tokens": max_tokens,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "shard_tokens": shard_tokens,
    }


def artifact_directory(data_dir: Path, descriptor: dict) -> Path:
    view = descriptor["view"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", view):
        raise ValueError(f"unsafe text view key {view!r}")
    key = content_hash(descriptor, digest_size=12)
    return data_dir / "text" / "artifacts" / "v3" / view / key


def compile_view(
    view_key: str | LocalTextView,
    logical_split: str,
    *,
    tokenizer,
    tokenizer_hash: str,
    data_dir: Path,
    max_tokens: Optional[int],
    bos_id: Optional[int],
    eos_id: Optional[int],
    data_files: DataFiles = None,
    shard_tokens: int = 20_000_000,
) -> Path:
    """Build a view if needed and return its content-addressed directory."""
    _validate_selection(view_key, data_files)
    descriptor = build_descriptor(
        view_key,
        logical_split,
        tokenizer_hash=tokenizer_hash,
        max_tokens=max_tokens,
        bos_id=bos_id,
        eos_id=eos_id,
        data_files=data_files,
        shard_tokens=shard_tokens,
    )
    directory = artifact_directory(data_dir, descriptor)
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.exists():
        TextArtifactManifest.load(manifest_path)
        return directory

    view = get_view(view_key)
    objective = OBJECTIVES[view.objective]
    dtype = (
        torch.int16
        if tokenizer.vocab_size - 1 <= torch.iinfo(torch.int16).max
        else torch.int32
    )
    writer = TextArtifactWriter(
        directory,
        build=descriptor,
        dtype=dtype,
        stores_labels=objective.stores_labels,
        shard_tokens=shard_tokens,
    )
    if max_tokens is not None and max_tokens <= 0:
        writer.finish()
        return directory
    rows = load_rows(
        view,
        logical_split,
        data_dir=data_dir,
        data_files=data_files,
    )
    examples = view.adapter.iter_examples(rows)
    progress = tqdm(
        examples, desc=f"Compiling {view.key} [{logical_split}]", unit=view.unit
    )
    for example in progress:
        ids = [bos_id] if bos_id is not None else []
        labels = [-100] if objective.stores_labels and bos_id is not None else []
        for segment in example.segments:
            piece = tokenizer.encode(segment.text, add_special_tokens=False)
            ids.extend(piece)
            if objective.stores_labels:
                labels.extend(
                    piece if objective.supervises(segment) else [-100] * len(piece)
                )
        if eos_id is not None:
            ids.append(eos_id)
            if objective.stores_labels:
                labels.append(-100)

        if max_tokens is not None and writer.tokens + len(ids) > max_tokens:
            remaining = max_tokens - writer.tokens
            if writer.tokens == 0 and remaining > 0:
                prefix = [bos_id] if bos_id is not None else []
                suffix = [eos_id] if eos_id is not None else []
                room = max(0, remaining - len(prefix) - len(suffix))
                body_start = len(prefix)
                body_end = max(body_start, len(ids) - len(suffix))
                ids = prefix + ids[body_start:body_end][:room] + suffix
                ids = ids[:remaining]
                if eos_id is not None and ids:
                    ids[-1] = eos_id
                if objective.stores_labels:
                    labels = labels[:remaining]
                    if len(labels) < len(ids):
                        labels.extend([-100] * (len(ids) - len(labels)))
                    if labels:
                        labels[-1] = -100
                writer.add_document(ids, labels if objective.stores_labels else None)
            break

        writer.add_document(ids, labels if objective.stores_labels else None)
        progress.set_postfix(tokens=writer.tokens)
        if max_tokens is not None and writer.tokens >= max_tokens:
            break
    progress.close()
    writer.finish()
    return directory
