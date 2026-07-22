"""Committed source lockfile support."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping


@dataclass(frozen=True)
class SourceLock:
    key: str
    repo: str
    revision: str
    gated: bool
    license: object
    files: tuple[str, ...] = ()
    schema: str = ""


class CatalogLock:
    def __init__(self, entries: Mapping[str, SourceLock], version: int = 1):
        self.entries = dict(entries)
        self.version = version

    @classmethod
    def load(cls, path: Path) -> "CatalogLock":
        payload = json.loads(path.read_text())
        if payload.get("version") != 1:
            raise ValueError(f"unsupported catalog lock version in {path}")
        entries = {
            key: SourceLock(
                key=key,
                **{
                    **entry,
                    "files": tuple(entry.get("files", ())),
                    "license": (
                        tuple(entry["license"])
                        if isinstance(entry.get("license"), list)
                        else entry.get("license")
                    ),
                },
            )
            for key, entry in payload["sources"].items()
        }
        return cls(entries, version=payload["version"])

    def require(self, key: str, repo: str) -> SourceLock:
        try:
            entry = self.entries[key]
        except KeyError as error:
            raise RuntimeError(
                f"source {key!r} is not locked; run `chimera-data text lock {key}`"
            ) from error
        if entry.repo != repo:
            raise RuntimeError(
                f"source lock mismatch for {key!r}: {entry.repo!r} != {repo!r}"
            )
        if re.fullmatch(r"[0-9a-f]{40}", entry.revision) is None:
            raise RuntimeError(f"source {key!r} is not pinned to a commit SHA")
        return entry
