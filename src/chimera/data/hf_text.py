"""Compatibility forwarding module; prefer :mod:`chimera.data.text`."""

from .text.hf_text import HFTextDataModule

__all__ = ["HFTextDataModule"]
