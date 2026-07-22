"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.gooaq import GooAQDataModule

__all__ = ["GooAQDataModule"]
