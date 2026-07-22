"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.squad import SQuADTextDataModule

__all__ = ["SQuADTextDataModule"]
