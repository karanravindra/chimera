"""LLM model definition."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.utils.checkpoint import checkpoint


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to ``x`` ([B, H, T, head_dim]).

    ``cos``/``sin`` are ``[B, 1, T, head_dim]`` and broadcast over the head
    dimension, so this works unchanged for q and k even under GQA (different
    head counts). cos/sin are computed in float32; cast to x's dtype here so the
    rotation runs in the activation dtype (bf16 under autocast) without
    promoting x.
    """
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE frequencies and turns position ids into cos/sin.

    Replaces the learned absolute ``pos_embedding`` table: position information
    is injected directly into q and k inside attention rather than added to the
    token embeddings. ``inv_freq`` is a derived buffer (non-persistent — kept out
    of the state dict).
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``positions`` is ``[T]`` (plain causal) or ``[B, T]`` (packed docs).

        Returns ``cos``/``sin`` each shaped ``[B, 1, T, head_dim]`` (B=1 for the
        plain path, broadcasting over the batch).
        """
        if positions.dim() == 1:
            positions = positions[None, :]  # [1, T]
        # [B, T, head_dim/2] -> duplicated to [B, T, head_dim].
        freqs = positions[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]  # each [B, 1, T, head_dim]


class MLP(nn.Module):
    def __init__(self, dim: int, m: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * m, bias=False)
        self.fc2 = nn.Linear(dim * m, dim, bias=False)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, block_mask=None):
        batch_size, seq_len, dim = x.size()

        qkv = self.qkv_proj(x).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, num_heads, seq_len, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotary position embedding applied after QK-norm, to q and k only.
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if block_mask is None:
            # Fast path: plain causal attention (flash kernel).
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        else:
            out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_multiplier: int = 4,
    ):
        super().__init__()
        self.attention = Attention(dim, num_heads)
        self.mlp = MLP(dim, mlp_multiplier)

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x, cos, sin, block_mask=None):
        x = x + self.attention(self.norm1(x), cos, sin, block_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        mlp_multiplier: int,
        doc_sep_token: int | None = None,
        grad_checkpoint: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        # When set, attention is masked per packed document (split on this
        # separator token) so no block attends across a document boundary.
        # None keeps plain causal attention (backward compatible).
        self.doc_sep_token = doc_sep_token
        # Recompute each block's activations in the backward pass instead of
        # storing them: trades ~extra compute for much lower peak memory, letting
        # a larger batch fit. Off by default; toggle the attribute to enable.
        self.grad_checkpoint = grad_checkpoint
        # Kept as a guard for the forward-length check. RoPE has no table limit
        # (unlike the old learned pos_embedding), so this is a soft sanity bound.
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, dim)
        # Position information now comes from RoPE inside attention rather than
        # a learned absolute position-embedding table added to the tokens.
        self.rotary = RotaryEmbedding(dim // num_heads, theta=rope_theta)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, mlp_multiplier)
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.RMSNorm(dim)

        # GPT-2 initialization.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            # Scale the residual projections by 1/sqrt(2 * num_layers).
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.051 / math.sqrt(2 * num_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x, return_hidden: bool = False, block_mask: BlockMask | None = None
    ):
        if x.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds max_seq_len={self.max_seq_len}"
            )
        # Build the intra-document BlockMask and per-document positions from the
        # token ids before x is reassigned to embeddings. None mask -> the fast
        # plain-causal path in Attention, with plain absolute positions. A caller
        # may pass a prebuilt block_mask (e.g. constructed eagerly outside a
        # torch.compile region); otherwise it is built here. The same positions
        # feed RoPE, so rotary phase resets per document exactly as the mask does.
        if self.doc_sep_token is not None:
            if block_mask is None:
                block_mask = self.document_block_mask(x)
            positions = self._document_positions(x)  # [B, T], resets each document
        else:
            block_mask = None
            positions = torch.arange(x.size(1), device=x.device)  # [T]
        cos, sin = self.rotary(positions)
        x = self.token_embedding(x)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = checkpoint(layer, x, cos, sin, block_mask, use_reentrant=False)
            else:
                x = layer(x, cos, sin, block_mask)
        x = self.norm_final(x)
        if return_hidden:
            return x
        return F.linear(x, self.token_embedding.weight)

    def document_block_mask(self, tokens: torch.Tensor) -> BlockMask:
        """FlexAttention ``BlockMask`` forbidding any cross-document attention.

        Documents are packed with ``doc_sep_token`` (the eot separator inserted at
        packing time) marking each boundary. A query attends an earlier key only
        when both lie in the same document. The separator belongs to the document
        it terminates, so the document index advances on the token *after* each
        separator (an exclusive cumsum). The diagonal is always valid, so no query
        is fully masked. This expresses the same pattern as a dense ``causal &
        same_doc`` mask, but as a sparse BlockMask that FlexAttention can run on a
        fused kernel — skipping the all-masked off-diagonal blocks entirely.
        """
        batch_size, seq_len = tokens.shape
        is_sep = (tokens == self.doc_sep_token).to(torch.int32)
        # Exclusive cumsum: doc_id[b, i] = number of separators strictly before i.
        doc_id = torch.cumsum(is_sep, dim=1) - is_sep  # [B, T]

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc_id[b, q_idx] == doc_id[b, kv_idx])

        block = 128  # flex's default sparse block size
        if seq_len % block:
            # Fallback (eval/generation lengths): the generic builder. Built in
            # eager OUTSIDE the model's torch.compile region and passed into
            # forward — creating it inside the compiled graph trips an Inductor
            # FlexibleLayout assertion on the inference forward.
            return create_block_mask(
                mask_mod,
                B=batch_size,
                H=None,  # same mask for every head
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=tokens.device,
                _compile=True,
            )

        # Training path: build the BlockMask directly from block-boundary doc
        # ids with fixed-shape tensor ops. create_block_mask runs aten::nonzero,
        # whose unknown output size forces a full device sync — profiled at
        # ~60 ms of CPU stall per training step, draining the GPU pipeline
        # right before every forward. doc_id is non-decreasing, so for k-block
        # j strictly below q-block i every doc in j <= every doc in i: they
        # share a document iff the q-block's first doc equals the k-block's
        # last. A block pair is FULL (mask_mod skipped by the kernel) when both
        # blocks lie entirely inside that one document; the diagonal is always
        # PARTIAL (causal boundary runs through it).
        num_blocks = seq_len // block
        start = doc_id[:, ::block]  # [B, N] doc at each block's first token
        end = doc_id[:, block - 1 :: block]  # [B, N] doc at each block's last
        q_start, q_end = start[:, :, None], end[:, :, None]
        k_start, k_end = start[:, None, :], end[:, None, :]
        idx = torch.arange(num_blocks, device=tokens.device)
        below = idx[:, None] > idx[None, :]
        nonempty_below = below & (q_start == k_end)
        full = nonempty_below & (q_start == q_end) & (k_start == k_end)
        partial = (nonempty_below & ~full) | (idx[:, None] == idx[None, :])

        def to_kv(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            num = mask.sum(dim=-1, dtype=torch.int32)[:, None]  # [B, 1, N]
            order = torch.argsort(~mask, dim=-1, stable=True)  # True blocks first
            return num, order.to(torch.int32)[:, None]  # [B, 1, N, N]

        partial_num, partial_idx = to_kv(partial)
        full_num, full_idx = to_kv(full)
        return BlockMask.from_kv_blocks(
            kv_num_blocks=partial_num,
            kv_indices=partial_idx,
            full_kv_num_blocks=full_num,
            full_kv_indices=full_idx,
            BLOCK_SIZE=block,
            mask_mod=mask_mod,
            seq_lengths=(seq_len, seq_len),
        )

    def _document_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Position ids ``[B, T]`` that reset to 0 at the start of each document.

        A document starts at index 0 and at every token immediately following a
        ``doc_sep_token`` (the separator belongs to the document it ends, matching
        ``document_block_mask``). The within-document offset is the absolute index minus
        the index of that document's first token, computed as a running maximum of
        the document-start indices. These feed RoPE, so rotary phase is relative to
        each document's start.
        """
        batch_size, seq_len = tokens.shape
        idx = torch.arange(seq_len, device=tokens.device)
        is_sep = tokens == self.doc_sep_token
        # A new document starts at index 0 and right after each separator.
        ones = torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device)
        doc_start = torch.cat([ones, is_sep[:, :-1]], dim=1)
        starts = torch.where(doc_start, idx, torch.zeros_like(idx))
        start_index = torch.cummax(starts, dim=1).values
        return idx - start_index


def llm_xs(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    doc_sep_token: int | None = None,
) -> LLM:
    """LLM XS (~10M, not an official size): 6 layers, dim 384, 6 heads."""
    return LLM(
        vocab_size=vocab_size,
        dim=384,
        num_heads=6,
        num_layers=6,
        mlp_multiplier=4,
        max_seq_len=max_seq_len,
        doc_sep_token=doc_sep_token,
    )


def llm(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    doc_sep_token: int | None = None,
) -> LLM:
    """LLM (124M): 12 layers, dim 768, 12 heads."""
    return LLM(
        vocab_size=vocab_size,
        dim=768,
        num_heads=12,
        num_layers=12,
        mlp_multiplier=4,
        max_seq_len=max_seq_len,
        doc_sep_token=doc_sep_token,
    )


def llm_medium(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    doc_sep_token: int | None = None,
) -> LLM:
    """LLM Medium (355M): 24 layers, dim 1024, 16 heads."""
    return LLM(
        vocab_size=vocab_size,
        dim=1024,
        num_heads=16,
        num_layers=24,
        mlp_multiplier=4,
        max_seq_len=max_seq_len,
        doc_sep_token=doc_sep_token,
    )


def llm_large(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    doc_sep_token: int | None = None,
) -> LLM:
    """LLM Large (774M): 36 layers, dim 1280, 20 heads."""
    return LLM(
        vocab_size=vocab_size,
        dim=1280,
        num_heads=20,
        num_layers=36,
        mlp_multiplier=4,
        max_seq_len=max_seq_len,
        doc_sep_token=doc_sep_token,
    )


def llm_xl(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    doc_sep_token: int | None = None,
) -> LLM:
    """LLM XL (1558M): 48 layers, dim 1600, 25 heads."""
    return LLM(
        vocab_size=vocab_size,
        dim=1600,
        num_heads=25,
        num_layers=48,
        mlp_multiplier=4,
        max_seq_len=max_seq_len,
        doc_sep_token=doc_sep_token,
    )
