"""FlexAttention plumbing shared by LM projects.

Compile-aware ``flex_attention`` wrapper plus causal + document block-mask
construction with per-document RoPE position ids (for training on packed
documents), and dense reference masks for visualization/tests.
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Eager flex is unusably slow, so the eager/CPU path goes through a compiled wrapper.
# Inside torch.compile(model) we call the RAW kernel instead: nesting an already-compiled
# callable in the outer graph is redundant at best (graph-break-prone at worst) and stops
# the outer compile from fusing the QKV/RoPE epilogue into flex's prologue.
_flex_compiled = torch.compile(flex_attention)


def flex_attn(q, k, v, block_mask):
    if torch.compiler.is_compiling():  # constant-folded away by Dynamo
        return flex_attention(q, k, v, block_mask=block_mask)
    return _flex_compiled(q, k, v, block_mask=block_mask)


# Mask construction is on the critical path every step; compiled it's one fused kernel
# instead of many small eager ones.
create_block_mask_c = torch.compile(create_block_mask)


def doc_ids_and_pos(x: torch.Tensor, eos_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token document ids and within-document positions for packed inputs.

    Documents are delimited by ``eos_id``; the EOS token is the last token of the
    document it closes, and positions restart at 0 on the token after each EOS.

    Args:
        x: packed token ids, shape (B, N).

    Returns:
        (doc_ids, pos_ids), both (B, N) int tensors.
    """
    B, N = x.shape
    is_eos = x == eos_id
    shifted_eos = torch.zeros_like(is_eos)
    shifted_eos[:, 1:] = is_eos[:, :-1]
    doc_ids = torch.cumsum(shifted_eos, dim=-1)  # (B, N)

    # Position within the current doc: index minus the index of the doc's first token
    # (cummax picks up the most recent doc-start index; index 0 is doc 0's start).
    idx = torch.arange(N, device=x.device).expand(B, N)
    pos_ids = idx - torch.cummax(idx * shifted_eos.long(), dim=-1).values  # (B, N)
    return doc_ids, pos_ids


def build_block_mask_and_pos(x: torch.Tensor, eos_id: int):
    """Causal + document BlockMask for FlexAttention, plus per-document RoPE position
    ids, both derived from one pass over the EOS positions. A query attends only to
    earlier keys in the SAME document (documents delimited by eos_id; the EOS token is
    the last token of the doc it closes), and positions restart at 0 on the token after
    each EOS — so packed docs are trained exactly as if each sat alone at positions
    0..len-1. Rebuilt per batch; the block-sparsity also skips fully-masked blocks, so
    packed short docs run faster."""
    B, N = x.shape
    doc_ids, pos_ids = doc_ids_and_pos(x, eos_id)

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    return create_block_mask_c(mask_mod, B, None, N, N, device=x.device), pos_ids


# --- Dense reference masks -------------------------------------------------------
# O(N^2) materialized masks. Not for training (use build_block_mask_and_pos); they
# exist as the readable spec of the masking rule, for notebook visualization and for
# tests that check the block-mask path against them.


def make_causal_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    """Make a causal attention mask for a sequence of length `seq_len`."""
    return torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    ).view(1, 1, seq_len, seq_len)


def make_document_mask(x: torch.Tensor, eos_id: int = 0) -> torch.Tensor:
    """Make a document attention mask from packed input tokens.

    Tokens may only attend to other tokens within the same document, where
    documents are delimited by `eos_id`. The EOS token itself is treated as
    the last token of the document it closes.

    Args:
        x: token ids, shape (seq_len,) or (batch, seq_len).
        eos_id: token id marking the end of a document.

    Returns:
        bool mask of shape (batch, 1, seq_len, seq_len), True = allowed.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (1, seq_len)
    batch, seq_len = x.shape
    doc_ids, _ = doc_ids_and_pos(x, eos_id)
    mask = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)  # (batch, seq_len, seq_len)
    return mask.view(batch, 1, seq_len, seq_len)
