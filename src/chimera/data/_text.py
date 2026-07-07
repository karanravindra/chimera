"""Shared helpers for character/token-level text DataModules."""

import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Wraps an encoded 1D tensor as non-overlapping ``(input, target)`` chunks.

    For a chunk starting at position ``i`` the target is the input shifted by one
    token, i.e. the model predicts the next token at every step. Works with any
    1D tensor of token ids (character ids, BPE ids, ...).
    """

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len].long()
        y = self.data[i + 1 : i + 1 + self.seq_len].long()
        return x, y
