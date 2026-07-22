"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.coqa_chat import CoQAChatDataModule

__all__ = ["CoQAChatDataModule"]
