import torch
from torch import nn

from chimera.nn.adaln import AdaLNZero, modulate
from chimera.nn.mlp import Mlp


class DiTBlock(nn.Module):
    """A DiT transformer block with adaLN-Zero conditioning.

    The conditioning vector produces per-block shift/scale/gate for both the
    self-attention and MLP sublayers; gates are zero-initialized (via
    :class:`AdaLNZero`) so each block starts as the identity.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_dim, int(hidden_dim * mlp_ratio))
        self.adaln = AdaLNZero(hidden_dim, 6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln(
            cond
        ).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x
