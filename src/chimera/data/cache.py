"""Atomic writes and content fingerprints shared by data modalities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def content_hash(value: Any, *, digest_size: int = 12) -> str:
    if isinstance(value, bytes):
        encoded = value
    else:
        encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.blake2b(encoded, digest_size=digest_size).hexdigest()


def file_hash(path: Path, *, digest_size: int = 12) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
