"""Canonical intermediate representation for all text training examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TextSegment:
    text: str
    kind: str = "content"


@dataclass(frozen=True)
class TextExample:
    segments: tuple[TextSegment, ...]
    document_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)
