"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.wikipedia import WikipediaDataModule

__all__ = ["WikipediaDataModule"]
