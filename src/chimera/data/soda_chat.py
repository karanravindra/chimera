"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.soda_chat import SODAChatDataModule

__all__ = ["SODAChatDataModule"]
