"""Compatibility forwarder; prefer :mod:`chimera.data.text`."""

from .text.local_documents import LocalDocumentsDataModule

__all__ = ["LocalDocumentsDataModule"]
