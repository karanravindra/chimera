import torch
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation: ``x * (1 + scale) + shift`` with broadcast over tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZero(nn.Module):
    """adaLN-Zero conditioning projection: ``SiLU -> Linear(dim, num_params * dim)``.

    Produces ``num_params`` per-channel modulation vectors (shift/scale/gate) from
    a conditioning embedding; the caller ``.chunk(num_params, dim=-1)``s the output.
    The linear layer is zero-initialized so a freshly built block starts as the
    identity (zero shift, zero scale offset, zero gate).
    """

    def __init__(self, dim: int, num_params: int):
        super().__init__()
        self.num_params = num_params
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim, num_params * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(cond))
