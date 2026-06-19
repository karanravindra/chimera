from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        batch_first: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} must be divisible by num_heads {num_heads}"
        )
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.batch_first = batch_first
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * num_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        # QK-norm is per-head, so normalize over head_dim, not dim
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        kv = self.kv(x).reshape(B, N, 2, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k, v = kv.unbind(dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out
