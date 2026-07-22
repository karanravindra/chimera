"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.tiny_textbooks import TinyTextbooksDataModule

__all__ = ["TinyTextbooksDataModule"]
