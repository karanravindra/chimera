"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.squad_chat import SQuADChatDataModule

__all__ = ["SQuADChatDataModule"]
