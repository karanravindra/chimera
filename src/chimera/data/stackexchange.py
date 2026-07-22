"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.stackexchange import StackExchangeDataModule

__all__ = ["StackExchangeDataModule"]
