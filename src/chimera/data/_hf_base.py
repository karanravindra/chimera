"""Compatibility forwarding module; prefer :mod:`chimera.data.text`."""

from .text._hf_base import _HFCorpusBase

__all__ = ["_HFCorpusBase"]
