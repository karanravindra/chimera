"""tinylm GPT: pre-norm transformer with RoPE, QK-norm, ReLU^2 MLP, tied embeddings.

Deliberately separate from ``chimera.models.gpt``: this one resets RoPE positions per
packed document (via ``pos_ids`` from ``build_block_mask_and_pos``) instead of relying
on RoPE's relative-offset invariance, has no muP parameterization, and keeps the
architecture minimal. It becomes a candidate for the shared library when the unified
LLM redo picks the canonical GPT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera.models.attention import flex_attn
from chimera.models.rope import RotaryEmbedding, apply_rotary


class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


class MLP(nn.Module):
    def __init__(self, dim: int, m: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * m, bias=False)
        self.fc2 = nn.Linear(dim * m, dim, bias=False)
        self.act = ReLU2()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        # QK-norm: per-head RMSNorm over the head dimension, applied before RoPE
        # (same convention as chimera.models.gpt). Tames logit growth / attention
        # entropy collapse; the 1-D weights route to the AdamW side of Muon.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x,
        block_mask=None,
        pos_ids=None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        past_len = past_kv[0].shape[2] if past_kv is not None else 0

        # pos_ids (B, N) resets positions per packed document (training); None falls
        # back to absolute past_len..past_len+N positions (eval / KV-cache decode).
        cos, sin = self.rope(past_len, N, q.device, pos_ids=pos_ids)
        q = apply_rotary(self.q_norm(q), cos, sin)
        k = apply_rotary(self.k_norm(k), cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        if block_mask is not None:
            # Training path: fused causal + document masking via FlexAttention.
            out = flex_attn(q, k, v, block_mask=block_mask)
        else:
            # No block mask -> plain causal via FlashAttention (prefill / eval), or full
            # attention over the KV cache on a decode step (past_len > 0, N == 1).
            # Branch (not a bool var): under dynamic shapes `N > 1` is a SymBool,
            # which SDPA rejects; an if forces Dynamo to guard/specialize instead.
            if past_len == 0 and N > 1:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out(out)
        return out, new_kv


class PreNorm(nn.Module):
    def __init__(self, module: nn.Module, norm: nn.Module):
        super().__init__()
        self.module = module
        self.norm = norm

    def forward(self, x, **kwargs):
        return self.module(self.norm(x), **kwargs)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.attn = PreNorm(MultiHeadAttention(dim, n_heads), nn.RMSNorm(dim))
        self.mlp = PreNorm(MLP(dim, mlp_mult), nn.RMSNorm(dim))

    def forward(
        self,
        x,
        block_mask=None,
        pos_ids=None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        attn_out, new_kv = self.attn(
            x,
            block_mask=block_mask,
            pos_ids=pos_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.mlp(x)
        return x, new_kv


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim: int,
        n_heads: int,
        mlp_mult: int,
        n_layers: int,
        eos_id: int = 0,
        logit_softcap: float | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.eos_id = eos_id
        # Gemma-2-style final-logit soft-capping: logits = cap*tanh(logits/cap).
        # Train-time regularizer against logit blow-up; applied here so eval and
        # sampling see the same capped distribution the loss was trained under
        # (CCE applies the same cap via its softcap arg on the hidden path).
        self.logit_softcap = logit_softcap
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, mlp_mult) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(dim)

        self.apply(self._init_weights())

    def _init_weights(self):
        def init_fn(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        return init_fn

    def forward(
        self,
        x,
        return_hidden: bool = False,
        block_mask=None,
        pos_ids=None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        # block_mask + pos_ids (built with build_block_mask_and_pos) enable FlexAttention
        # document masking with per-document RoPE positions; pass both during training.
        # Leave them None for eval/sampling, where attention falls back to a plain causal
        # FlashAttention kernel with absolute positions.
        h = self.token_emb(x)

        new_past_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            block_past_kv = past_kv[i] if past_kv is not None else None
            h, new_kv = block(
                h,
                block_mask=block_mask,
                pos_ids=pos_ids,
                past_kv=block_past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_kv.append(new_kv)

        h = self.ln_f(h)
        if return_hidden:
            return (h, new_past_kv) if use_cache else h
        logits = F.linear(h, self.token_emb.weight)
        if self.logit_softcap is not None:
            cap = self.logit_softcap
            if logits.requires_grad:
                logits = cap * torch.tanh(logits / cap)
            else:
                # eval logits are (B, L, V) and huge (lm-eval batches ~131k tokens
                # x 16k vocab = 4GB bf16); out-of-place tanh doubles that and OOMs
                logits = logits.div_(cap).tanh_().mul_(cap)
        return (logits, new_past_kv) if use_cache else logits

    @torch.no_grad()
    def sample(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
        stop_token_ids: set[int] | None = None,
        seed: int | None = None,
        return_token_ids: bool = False,
        use_cache: bool = True,
    ) -> str | tuple[str, torch.Tensor]:
        self.eval()

        device = next(self.parameters()).device
        context_length = self.seq_len

        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None

        encoded = tokenizer.encode(prompt)

        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(device=device, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

        if bos_token_id is not None and (
            input_ids.shape[1] == 0 or input_ids[0, 0].item() != bos_token_id
        ):
            bos = torch.full(
                (input_ids.shape[0], 1), bos_token_id, dtype=torch.long, device=device
            )
            input_ids = torch.cat([bos, input_ids], dim=1)

        generated_ids = input_ids

        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)

        stop_token_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)

        past_kv = None
        model_input = generated_ids[:, -context_length:]

        for _ in range(max_new_tokens):
            if use_cache:
                out = self(model_input, past_kv=past_kv, use_cache=True)
                logits, past_kv = out
            else:
                model_input = generated_ids[:, -context_length:]
                logits = self(model_input)

            next_token_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                used_token_ids = torch.unique(generated_ids)
                used_logits = next_token_logits[:, used_token_ids]
                used_logits = torch.where(
                    used_logits < 0,
                    used_logits * repetition_penalty,
                    used_logits / repetition_penalty,
                )
                next_token_logits[:, used_token_ids] = used_logits

            if frequency_penalty != 0.0 or presence_penalty != 0.0:
                token_counts = torch.bincount(
                    generated_ids.flatten(),
                    minlength=next_token_logits.size(-1),
                ).to(next_token_logits.dtype)
                next_token_logits -= frequency_penalty * token_counts
                next_token_logits -= presence_penalty * (token_counts > 0)

            if temperature <= 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits /= temperature

                if 0 < top_k < next_token_logits.size(-1):
                    top_k_values, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
                    cutoff = top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < cutoff, float("-inf")
                    )

                if min_p > 0:
                    probabilities = F.softmax(next_token_logits, dim=-1)
                    max_probabilities = probabilities.max(dim=-1, keepdim=True).values
                    min_probability = min_p * max_probabilities
                    next_token_logits = next_token_logits.masked_fill(
                        probabilities < min_probability, float("-inf")
                    )

                if 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True, dim=-1
                    )
                    sorted_probabilities = F.softmax(sorted_logits, dim=-1)
                    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
                    remove_mask = cumulative_probabilities > top_p
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                    next_token_logits = filtered_logits.scatter(
                        dim=-1, index=sorted_indices, src=sorted_logits
                    )

                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            model_input = next_token

            if next_token.item() in stop_token_ids:
                break

            if use_cache and generated_ids.shape[1] > context_length:
                past_kv = None
                model_input = generated_ids[:, -context_length:]

        token_ids = generated_ids[0].tolist()

        try:
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            text = tokenizer.decode(token_ids)

        if return_token_ids:
            return text, generated_ids

        return text
