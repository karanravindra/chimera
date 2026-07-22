"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.gooaq_chat import GooAQChatDataModule

__all__ = ["GooAQChatDataModule"]
