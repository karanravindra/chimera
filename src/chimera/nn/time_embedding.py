import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a ``(B,)`` time tensor in [0, 1] -> ``(B, dim)``."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad an odd dim
        emb = F.pad(emb, (0, 1))
    return emb


class SinusoidalTimeEmbedding(nn.Module):
    """Module wrapper around :func:`timestep_embedding`.

    Maps a ``(B,)`` (or ``(B, 1)``) time tensor in [0, 1] to ``(B, dim)``
    sinusoidal features. A parameter-free sibling for future learned/Fourier
    time embeddings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t.reshape(-1), self.dim)
