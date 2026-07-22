"""Compatibility forwarding module; prefer :mod:`chimera.data.text`."""

from .text.chat_sft import ChatSFTDataModule

__all__ = ["ChatSFTDataModule"]
