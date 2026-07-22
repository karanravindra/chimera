"""Loss policies applied after source-specific rendering."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import TextSegment


@dataclass(frozen=True)
class NextTokenObjective:
    key: str = "next-token"
    stores_labels: bool = False

    def supervises(self, segment: TextSegment) -> bool:
        return True


@dataclass(frozen=True)
class AssistantOnlyObjective:
    key: str = "assistant-only"
    stores_labels: bool = True

    def supervises(self, segment: TextSegment) -> bool:
        return segment.kind == "assistant"


OBJECTIVES = {
    "next-token": NextTokenObjective(),
    "assistant-only": AssistantOnlyObjective(),
}
