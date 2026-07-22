"""Modality-neutral source declarations and lazy source loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence, Union


DataFiles = Optional[Union[str, Sequence[str]]]


@dataclass(frozen=True)
class HFSource:
    """A Hugging Face dataset repository independent of any training view."""

    repo: str
    config: Optional[str] = None
    data_files: DataFiles = None
    gated: bool = False

    def select(self, *, data_files: DataFiles = None, config: Optional[str] = None):
        return replace(
            self,
            data_files=self.data_files if data_files is None else data_files,
            config=self.config if config is None else config,
        )

    def load(
        self,
        *,
        split: str,
        revision: str,
        cache_dir: Union[str, Path],
        streaming: bool = False,
    ):
        from datasets import load_dataset

        kwargs = {
            "cache_dir": str(cache_dir),
            "revision": revision,
            "streaming": streaming,
        }
        if self.config is not None:
            kwargs["name"] = self.config
        if self.data_files is not None:
            kwargs["data_files"] = self.data_files
            kwargs["verification_mode"] = "no_checks"
        return load_dataset(self.repo, split=split, **kwargs)


@dataclass(frozen=True)
class LocalSource:
    """A deterministic set of local documents."""

    directory: Path
    glob: str = "*.md"
    repeat: int = 1

    @property
    def files(self) -> tuple[Path, ...]:
        files = tuple(sorted(self.directory.glob(self.glob)))
        if not files:
            raise FileNotFoundError(
                f"no files matching {self.glob!r} in {self.directory}"
            )
        return files
