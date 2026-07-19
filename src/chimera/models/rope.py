"""Rotary position embedding helpers shared by GQA and MLA attention."""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings, computed on the fly so positions are unbounded."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, offset: int, seq_len: int, device, pos_ids: torch.Tensor | None = None):
        # pos_ids overrides the contiguous offset..offset+seq_len positions with explicit
        # per-token positions, e.g. (B, T) ids that restart at 0 on each packed-document
        # boundary. cos/sin then come back with matching leading dims.
        if pos_ids is None:
            pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        else:
            pos = pos_ids.to(torch.float32)
        freqs = pos[..., None] * self.inv_freq  # (..., T, head_dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (..., T, head_dim)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim), or (B, T, head_dim) when built
    # from per-row position ids.
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(1)  # (B, 1, T, head_dim)
        sin = sin.unsqueeze(1)
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)
