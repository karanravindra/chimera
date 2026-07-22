"""Reusable logical-to-physical dataset split policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SplitPolicy(Protocol):
    def resolve(self, logical_split: str) -> str: ...


@dataclass(frozen=True)
class NativeSplits:
    train: str = "train"
    validation: str = "validation"

    def resolve(self, logical_split: str) -> str:
        if logical_split == "train":
            return self.train
        if logical_split == "validation":
            return self.validation
        raise ValueError(f"unknown logical split {logical_split!r}")


@dataclass(frozen=True)
class CarvedValidation:
    """Preserve Chimera's ordered head-validation/tail-training convention."""

    source_split: str = "train"
    fraction: float = 0.01

    def resolve(self, logical_split: str) -> str:
        if not 0 < self.fraction < 1:
            raise ValueError("validation fraction must be between zero and one")
        pct = max(1, round(self.fraction * 100))
        if logical_split == "validation":
            return f"{self.source_split}[:{pct}%]"
        if logical_split == "train":
            return f"{self.source_split}[{pct}%:]"
        raise ValueError(f"unknown logical split {logical_split!r}")
