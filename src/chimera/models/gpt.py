import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act = ReLU2()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings, computed on the fly so positions are unbounded."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, offset: int, seq_len: int, device):
        pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)  # (T, head_dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
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
        self.kv_proj = nn.Linear(dim, n_kv_head * self.head_dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # QK-norm: per-head RMSNorm over the head dimension before attention.
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, past_kv=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = (
            self.kv_proj(x)
            .view(B, T, self.n_kv_head, self.head_dim * 2)
            .transpose(1, 2)
        )
        k, v = kv.chunk(2, dim=-1)

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
        mup_base_width: int = 256,
        mup_base_std: float = 0.02,
        mup_input_mult: float = 1.0,
        mup_output_mult: float = 1.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        # Recompute each block's activations in the backward pass instead of
        # storing them — trades ~1 extra forward of compute for a large drop in
        # activation memory. Needed to fit wide models (e.g. 124M @ seq 2048) on
        # 16 GB without shrinking the (comparability-critical) tokens/step.
        self.gradient_checkpointing = gradient_checkpointing

        # Maximal Update Parameterization (muP). The width multiplier m_d relates
        # this model's width to a base/proxy width; init variance and the output
        # logit multiplier are scaled by m_d so per-step feature updates stay
        # width-invariant, letting muon_lr/adamw_lr found at the base width
        # transfer to larger widths (muTransfer). At n_embd == mup_base_width
        # (m_d == 1) this reduces to the original GPT-2-style parameterization.
        # NB: LRs need no width scaling here — all hidden matrices go to Muon
        # (whose update is spectrally normalized) and all AdamW params
        # (embedding/head/norms) have Theta(1) muP LR.
        self.mup_width_mult = n_embd / mup_base_width
        self.mup_base_std = mup_base_std
        self.mup_input_mult = mup_input_mult
        # 1/m_d output scaling (multiplier form of the muP readout), applied at the
        # logit projection instead of scaling the (tied) head's learning rate.
        self.output_mult = mup_output_mult / self.mup_width_mult

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

        if not tie_embedding:
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        # muP init: hidden-matrix variance is scaled by 1/m_d so activation
        # magnitudes stay width-invariant; the embedding (input layer) keeps a
        # width-independent std. Without embedding init nn.Embedding defaults to
        # N(0, 1), and since the head is tied to it the logits blow up (~10x std),
        # starting the loss far above ln(vocab) instead of at it.
        hidden_std = self.mup_base_std / math.sqrt(self.mup_width_mult)

        # Embedding (input layer): constant std, no width scaling. This tensor is
        # shared with the tied output head, so it also sets the readout scale —
        # the muP output behavior comes from self.output_mult (the 1/m_d factor).
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.mup_base_std)

        for block in self.blocks:
            for lin in (
                block.attn.q_proj,
                block.attn.kv_proj,
                block.attn.proj,
                block.mlp.fc1,
                block.mlp.fc2,
            ):
                nn.init.normal_(lin.weight, mean=0.0, std=hidden_std)
            # Scale residual-writing projections by 1/sqrt(2 * n_layer) so the
            # residual stream variance stays bounded with depth (GPT-2), on top of
            # the muP width scaling already baked into hidden_std.
            residual_std = hidden_std / math.sqrt(2 * self.n_layer)
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=residual_std)

        # If a genuinely untied head ever exists, initialize it at readout scale.
        # (Currently the head is always tied to tok_emb, so this is a no-op.)
        head = getattr(self, "head", None)
        if head is not None and head.weight is not self.tok_emb.weight:
            nn.init.normal_(head.weight, mean=0.0, std=self.mup_base_std)

    def _forward(self, idx, past_kvs=None):
        B, T = idx.shape
        if past_kvs is None:
            past_len = 0
            past_kvs = [None] * len(self.blocks)
        else:
            past_len = past_kvs[0][0].size(2)

        x = self.tok_emb(idx) * self.mup_input_mult  # muP embedding multiplier
        cos, sin = self.rope(past_len, T, idx.device)

        # Activation checkpointing only applies to the training/prefill path (no
        # KV cache); incremental decode runs under no_grad, where it's a no-op.
        use_ckpt = self.gradient_checkpointing and self.training and past_kvs[0] is None

        presents = []
        for block, past in zip(self.blocks, past_kvs):
            if use_ckpt:
                x, present = checkpoint.checkpoint(
                    block, x, cos, sin, past, use_reentrant=False
                )
            else:
                x, present = block(x, cos, sin, past)
            presents.append(present)

        x = self.ln_f(x)
        return x, presents  # hidden states (pre-projection)

    @property
    def lm_head_weight(self):
        """The classifier weight matrix (V, C), whether tied or a separate head."""
        return self.head.weight if hasattr(self, "head") else self.tok_emb.weight

    def project(self, hidden):
        """Project hidden states to vocabulary logits (with the muP output multiplier)."""
        return (hidden @ self.lm_head_weight.t()) * self.output_mult

    def forward(self, idx, return_hidden: bool = False):
        assert idx.size(1) <= self.block_size, (
            "Cannot forward, model block size is exhausted."
        )
        hidden, _ = self._forward(idx, past_kvs=None)
        # Cut Cross Entropy fuses the head projection with the loss, so it needs
        # the hidden states and lm_head_weight rather than materialized logits.
        if return_hidden:
            return hidden
        return self.project(hidden)

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        compile: bool = False,
    ):
        """Autoregressively sample ``max_new_tokens`` tokens with a KV cache.

        These tiny models are launch-overhead bound during decode, so ``compile=True``
        ``torch.compile``s the per-step forward (~2-3x faster steady-state). The first
        call pays a one-time compile warmup that exceeds the saving on a single short
        run — it only pays off when generating repeatedly (the compiled step is cached
        on the module and reused across calls).
        """
        self.eval()

        step = self._forward
        if compile:
            if getattr(self, "_forward_compiled", None) is None:
                # dynamic=True: one compiled artifact serves both the variable-length
                # prefill and the length-1 decode step (no per-length recompiles).
                self._forward_compiled = torch.compile(self._forward, dynamic=True)
            step = self._forward_compiled

        # Prefill the cache with the (cropped) prompt in a single pass.
        idx_cond = idx[:, -self.block_size :]
        hidden, past_kvs = step(idx_cond, past_kvs=None)

        for _ in range(max_new_tokens):
            # Only the last position is needed to sample the next token.
            next_logits = self.project(hidden[:, -1, :]) / temperature
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            # Feed only the new token; RoPE/attention use the cached keys/values.
            hidden, past_kvs = step(nxt, past_kvs=past_kvs)
        return idx


if __name__ == "__main__":
    from torchinfo import summary

    model = GPT(
        vocab_size=256, block_size=256, n_embd=384, n_head=6, n_kv_head=2, n_layer=6
    )
    summary(
        model, input_size=(1, 256), col_names=["output_size", "num_params", "mult_adds"], dtypes=[torch.int64]
    )

    # the logits should be roughly uniform, so the loss should be close to ln(vocab_size)
    with torch.no_grad():
        x = torch.randint(0, 256, (32, 256), dtype=torch.long, device=next(model.parameters()).device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        print(f"loss: {loss.item():.4f} (should be ~{math.log(256):.4f})")
