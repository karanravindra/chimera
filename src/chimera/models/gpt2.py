"""GPT-2 model definition."""

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


class MLP(nn.Module):
    def __init__(self, dim: int, m: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * m)
        self.fc2 = nn.Linear(dim * m, dim)
        # GPT-2 uses the tanh ("gelu_new") approximation, not the exact erf GELU.
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, block_mask=None):
        batch_size, seq_len, dim = x.size()
        qkv = self.qkv_proj(x).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if block_mask is None:
            # Fast path: plain causal attention (flash kernel).
            dropout_p = self.attn_dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )
        else:
            # block_mask encodes causality + per-document boundaries as a sparse
            # BlockMask. FlexAttention runs a fused flash-style kernel that skips
            # fully-masked blocks, instead of materializing a dense [B,1,T,T] mask
            # and falling off SDPA's flash path. (No attention dropout here — doc
            # packing is used for from-scratch pretraining where dropout=0.)
            out = flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(dim, num_heads, dropout)
        self.mlp = MLP(dim, mlp_multiplier, dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, block_mask=None):
        x = x + self.attention(self.norm1(x), block_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
        doc_sep_token: int | None = None,
        grad_checkpoint: bool = False,
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
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        # GPT-2 has a final LayerNorm before the (tied) output projection.
        self.norm_final = nn.LayerNorm(dim)

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
        if x.size(1) > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds "
                f"max_seq_len={self.pos_embedding.num_embeddings}"
            )
        # Build the intra-document BlockMask and per-document positions from the
        # token ids before x is reassigned to embeddings. None mask -> the fast
        # plain-causal path in Attention, with plain absolute positions. A caller
        # may pass a prebuilt block_mask (e.g. constructed eagerly outside a
        # torch.compile region); otherwise it is built here.
        if self.doc_sep_token is not None:
            if block_mask is None:
                block_mask = self.document_block_mask(x)
            positions = self._document_positions(x)  # [B, T], resets each document
        else:
            block_mask = None
            positions = torch.arange(x.size(1), device=x.device)  # [T]
        x = self.embed_dropout(self.token_embedding(x) + self.pos_embedding(positions))
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = checkpoint(layer, x, block_mask, use_reentrant=False)
            else:
                x = layer(x, block_mask)
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

        # _compile=True builds the block mask via a compiled, block-granular path
        # (no dense [B,1,T,T] materialization). Build this in eager OUTSIDE the
        # model's torch.compile region and pass it into forward — creating it
        # inside the compiled graph trips an Inductor FlexibleLayout assertion on
        # the inference forward.
        return create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,  # same mask for every head
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=tokens.device,
            _compile=True,
        )

    def _document_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Position ids ``[B, T]`` that reset to 0 at the start of each document.

        A document starts at index 0 and at every token immediately following a
        ``doc_sep_token`` (the separator belongs to the document it ends, matching
        ``document_block_mask``). The within-document offset is the absolute index minus
        the index of that document's first token, computed as a running maximum of
        the document-start indices. Within-document positions never exceed the
        absolute index, so they stay inside the positional-embedding table.
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


# ---------------------------------------------------------------------------
# Variants. Standard GPT-2 sizes (vocab 50257, context 1024, head_dim 64).
# The original GPT-2 used dropout=0.1; the default here is 0.0, which is the
# usual choice for from-scratch pretraining. Pass dropout=0.1 for the original.
# ---------------------------------------------------------------------------


def gpt2_xs(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    dropout: float = 0.0,
    doc_sep_token: int | None = None,
) -> GPT:
    """GPT-2 XS (~10M, not an official size): 6 layers, dim 384, 6 heads."""
    return GPT(
        vocab_size=vocab_size,
        dim=384,
        num_heads=6,
        num_layers=6,
        max_seq_len=max_seq_len,
        dropout=dropout,
        doc_sep_token=doc_sep_token,
    )


def gpt2(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    dropout: float = 0.0,
    doc_sep_token: int | None = None,
) -> GPT:
    """GPT-2 (124M): 12 layers, dim 768, 12 heads."""
    return GPT(
        vocab_size=vocab_size,
        dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=max_seq_len,
        dropout=dropout,
        doc_sep_token=doc_sep_token,
    )


def gpt2_medium(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    dropout: float = 0.0,
    doc_sep_token: int | None = None,
) -> GPT:
    """GPT-2 Medium (355M): 24 layers, dim 1024, 16 heads."""
    return GPT(
        vocab_size=vocab_size,
        dim=1024,
        num_heads=16,
        num_layers=24,
        max_seq_len=max_seq_len,
        dropout=dropout,
        doc_sep_token=doc_sep_token,
    )


def gpt2_large(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    dropout: float = 0.0,
    doc_sep_token: int | None = None,
) -> GPT:
    """GPT-2 Large (774M): 36 layers, dim 1280, 20 heads."""
    return GPT(
        vocab_size=vocab_size,
        dim=1280,
        num_heads=20,
        num_layers=36,
        max_seq_len=max_seq_len,
        dropout=dropout,
        doc_sep_token=doc_sep_token,
    )


def gpt2_xl(
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
    dropout: float = 0.0,
    doc_sep_token: int | None = None,
) -> GPT:
    """GPT-2 XL (1558M): 48 layers, dim 1600, 25 heads."""
    return GPT(
        vocab_size=vocab_size,
        dim=1600,
        num_heads=25,
        num_layers=48,
        max_seq_len=max_seq_len,
        dropout=dropout,
        doc_sep_token=doc_sep_token,
    )
