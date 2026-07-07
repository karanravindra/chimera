import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings, computed on the fly so positions are unbounded."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, offset: int, seq_len: int, device):
        # positions [offset, offset + seq_len); float32 for precision regardless of AMP
        pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.float())  # (T, head_dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    """Causal attention with grouped-query heads, RoPE, QK-norm, and an optional KV cache."""

    def __init__(self, dim: int, n_head: int, n_kv_head: int):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head

        # Q keeps n_head heads; K and V share only n_kv_head heads (the GQA fix).
        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        # QK-norm: per-head RMSNorm over the head dimension before attention.
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, past_kv=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # QK-norm, then rotary embeddings on the fresh q/k for this step.
        q = apply_rotary(self.q_norm(q), cos, sin)
        k = apply_rotary(self.k_norm(k), cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)

        if past_kv is None:
            # Prefill / training: square causal mask.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        else:
            # Incremental decode: query positions [past_len, past_len+T) attend to all
            # keys up to and including their own absolute position.
            Tk = k.size(2)
            past_len = Tk - T
            q_pos = torch.arange(past_len, Tk, device=x.device)
            k_pos = torch.arange(Tk, device=x.device)
            attn_mask = k_pos[None, :] <= q_pos[:, None]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=True
            )

        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.proj(out), present


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, n_kv_head: int, mlp_hidden_dim: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_head, n_kv_head)
        self.ln2 = nn.RMSNorm(dim)
        self.mlp = MLP(dim, mlp_hidden_dim)

    def forward(self, x, cos, sin, past_kv=None):
        attn_out, present = self.attn(self.ln1(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    """A decoder-only Transformer (GPT-style) for character-level language modeling.

    Uses grouped-query attention, rotary position embeddings, QK-norm, and a KV
    cache during generation.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_embd: int = 384,
        n_head: int = 6,
        n_kv_head: int = 2,
        n_layer: int = 6,
        tie_embedding: bool = False,
    ):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=n_embd,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    mlp_hidden_dim=4 * n_embd,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.RMSNorm(n_embd)

        if tie_embedding:
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight
        else:
            self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def _forward(self, idx, past_kvs=None):
        B, T = idx.shape
        if past_kvs is None:
            past_len = 0
            past_kvs = [None] * len(self.blocks)
        else:
            past_len = past_kvs[0][0].size(2)

        x = self.tok_emb(idx)
        cos, sin = self.rope(past_len, T, idx.device)

        presents = []
        for block, past in zip(self.blocks, past_kvs):
            x, present = block(x, cos, sin, past)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, presents

    def forward(self, idx):
        assert idx.size(1) <= self.block_size, (
            "Cannot forward, model block size is exhausted."
        )
        logits, _ = self._forward(idx, past_kvs=None)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        self.eval()
        # Prefill the cache with the (cropped) prompt in a single pass.
        idx_cond = idx[:, -self.block_size :]
        logits, past_kvs = self._forward(idx_cond, past_kvs=None)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            # Feed only the new token; RoPE/attention use the cached keys/values.
            logits, past_kvs = self._forward(nxt, past_kvs=past_kvs)
        return idx
