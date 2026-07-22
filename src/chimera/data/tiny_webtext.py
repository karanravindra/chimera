"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.tiny_webtext import TinyWebTextDataModule

__all__ = ["TinyWebTextDataModule"]
