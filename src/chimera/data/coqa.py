"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.coqa import CoQADataModule

__all__ = ["CoQADataModule"]
