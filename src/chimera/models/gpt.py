import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from chimera.models.rope import RotaryEmbedding, apply_rotary

# flex_attention is ~75-80x slower than SDPA when run eagerly (it materializes
# the full score matrix instead of a fused kernel) -- it MUST run compiled.
# Compiling this one shared callable (instead of relying on the caller's own
# torch.compile(model, ...)) means attention stays fast even when the outer
# model is deliberately left uncompiled (e.g. --use-moe, whose data-dependent
# routing makes whole-model compile unreliable) or under generate()'s default
# eager decode path. dynamic=True avoids a recompile per new KV-cache length
# during incremental decode.
_flex_attention = torch.compile(flex_attention, dynamic=True)


def _causal_mask_mod(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


def _make_offset_causal_mask_mod(past_len: int):
    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx + past_len

    return mask_mod


def build_block_mask(B: int, T: int, past_len: int, device) -> BlockMask:
    """Causal block mask for flex_attention, covering both prefill (past_len=0,
    Q_LEN==KV_LEN==T) and incremental decode (query positions offset by the
    cached prefix length).

    Building a fresh BlockMask every call has real cost even under compile
    (~4x the compiled attention call itself, measured) -- callers should cache
    the prefill mask across steps (T is constant for a whole training run) and
    only rebuild for decode, where KV_LEN changes every step anyway.
    """
    if past_len == 0:
        return create_block_mask(_causal_mask_mod, B, None, T, T, device=device)
    return create_block_mask(
        _make_offset_causal_mask_mod(past_len), B, None, T, past_len + T, device=device
    )


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

    def hidden_params(self):
        return [self.fc1.weight, self.fc2.weight]

    def residual_params(self):
        return [self.fc2.weight]


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

    def hidden_params(self):
        return [self.q_proj.weight, self.kv_proj.weight, self.proj.weight]

    def residual_params(self):
        return [self.proj.weight]

    @staticmethod
    def cached_len(past_kv) -> int:
        return past_kv[0].size(2)

    def forward(self, x, cos, sin, block_mask, past_kv=None):
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

        # block_mask is built once per GPT._forward call (prefill: cached
        # across steps since T is constant for a whole run; decode: fresh
        # each step, since KV_LEN grows) and threaded down alongside cos/sin.
        out = _flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)

        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.proj(out), present


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V3 Multi-head Latent Attention.

    Compresses K/V (and optionally Q) into a low-rank latent per token instead
    of full per-head projections, shrinking the KV cache to
    ``kv_lora_rank + qk_rope_head_dim`` per token instead of
    ``2 * n_kv_head * head_dim`` for GQA. RoPE doesn't commute with the
    low-rank compression, so position information is carried by a small
    "decoupled" extra slice that bypasses compression: one shared (not
    per-head) rope-key, and a per-head rope-query.

    Unlike ``GroupedQueryAttention``, no separate QK-norm is applied to the
    up-projected per-head q/k here — DeepSeek-V3 only normalizes the
    compressed latents (``cq_norm``/``ckv_norm``), not the post-up-projection
    heads.
    """

    def __init__(
        self,
        dim: int,
        n_head: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
    ):
        super().__init__()
        self.n_head = n_head
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Packed: c_kv (the compressed KV latent) + the single shared rope-key.
        self.w_dkv = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.ckv_norm = nn.RMSNorm(kv_lora_rank)
        self.w_uk = nn.Linear(kv_lora_rank, n_head * qk_nope_head_dim, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, n_head * v_head_dim, bias=False)

        if q_lora_rank > 0:
            self.w_dq = nn.Linear(dim, q_lora_rank, bias=False)
            self.cq_norm = nn.RMSNorm(q_lora_rank)
            self.w_uq = nn.Linear(q_lora_rank, n_head * self.qk_head_dim, bias=False)
            self.q_proj = None
        else:
            # q_lora_rank == 0 skips Q compression entirely (DeepSeek's "Lite"
            # variant) -- only worthwhile at much larger width than this repo
            # trains at.
            self.w_dq = None
            self.q_proj = nn.Linear(dim, n_head * self.qk_head_dim, bias=False)

        self.proj = nn.Linear(n_head * v_head_dim, dim, bias=False)

    def hidden_params(self):
        params = [self.w_dkv.weight, self.w_uk.weight, self.w_uv.weight, self.proj.weight]
        if self.w_dq is not None:
            params += [self.w_dq.weight, self.w_uq.weight]
        else:
            params.append(self.q_proj.weight)
        return params

    def residual_params(self):
        return [self.proj.weight]

    @staticmethod
    def cached_len(past_kv) -> int:
        # c_kv: (B, T, kv_lora_rank) -- T is dim 1 (no head axis, unlike GQA's k/v).
        return past_kv[0].size(1)

    def forward(self, x, cos, sin, block_mask, past_kv=None):
        B, T, _ = x.shape

        if self.w_dq is not None:
            q_full = self.w_uq(self.cq_norm(self.w_dq(x)))
        else:
            q_full = self.q_proj(x)
        q_full = q_full.view(B, T, self.n_head, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q_full.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_rope = apply_rotary(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)

        ckv_krope = self.w_dkv(x)
        c_kv, k_rope = ckv_krope.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_rope = k_rope.unsqueeze(1)  # (B, 1, T, qk_rope_head_dim): shared across heads
        k_rope = apply_rotary(k_rope, cos, sin)

        if past_kv is not None:
            past_c_kv, past_k_rope = past_kv
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
        present = (c_kv, k_rope)

        # RMSNorm normalizes per-token (over the feature dim), so re-running it
        # on the full cached sequence each step is equivalent to normalizing
        # and caching each token once -- simpler, at the cost of some redundant
        # compute during decode.
        c_kv_normed = self.ckv_norm(c_kv)
        Tk = c_kv_normed.size(1)
        k_nope = (
            self.w_uk(c_kv_normed).view(B, Tk, self.n_head, self.qk_nope_head_dim).transpose(1, 2)
        )
        v = self.w_uv(c_kv_normed).view(B, Tk, self.n_head, self.v_head_dim).transpose(1, 2)
        k = torch.cat([k_nope, k_rope.expand(-1, self.n_head, -1, -1)], dim=-1)

        out = _flex_attention(q, k, v, block_mask=block_mask)

        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.v_head_dim)
        return self.proj(out), present


class SwiGLUExpert(nn.Module):
    """A single dense SwiGLU FFN, used for the always-on shared expert(s)."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Gate(nn.Module):
    """DeepSeek-V3 aux-loss-free router.

    Sigmoid affinity scores (not softmax) against learnable per-expert
    centroids. A per-expert bias is added to the scores for top-k *selection*
    only; the combining weight is the raw (bias-free) score of the selected
    experts, renormalized among the top-k. The bias is a plain buffer nudged
    by a fixed step each training forward based on observed load -- this
    entirely replaces a gradient-based load-balancing auxiliary loss.

    Note: under ``gradient_checkpointing=True`` a block's forward (hence this
    gate) runs twice per optimizer step (recompute in backward), which would
    double-apply the bias update. Not guarded against here -- halve
    ``bias_update_speed`` or avoid combining the two flags until fixed.
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        route_scale: float,
        bias_update_speed: float,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.route_scale = route_scale
        self.bias_update_speed = bias_update_speed
        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        self.register_buffer("bias", torch.zeros(n_routed_experts))

    def forward(self, x):
        # x: (N, dim) flattened tokens
        scores = torch.sigmoid(x.float() @ self.weight.float().t())  # (N, n_routed_experts)
        topk_idx = (scores + self.bias).topk(self.n_activated_experts, dim=-1).indices
        topk_scores = scores.gather(-1, topk_idx)
        weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        weights = (weights * self.route_scale).to(x.dtype)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(
                    topk_idx.flatten(), minlength=self.n_routed_experts
                ).float()
                self.bias += self.bias_update_speed * torch.sign(counts.mean() - counts)

        return weights, topk_idx


class DeepSeekMoE(nn.Module):
    """DeepSeek-V3 style MoE FFN: fine-grained routed experts (top-k, aux-loss-free
    balancing via ``Gate``) plus always-on shared expert(s), combined as
    ``y = shared(x) + sum_{i in topk} weight_i * expert_i(x)``.

    Routed-expert compute uses ``torch.nn.functional.grouped_mm`` (batched GEMM
    over jagged per-expert token counts) when running on CUDA with bf16
    activations; otherwise falls back to an eager masked loop over experts
    (used for CPU smoke tests and as a numerical reference for the grouped_mm
    path).
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_inter_dim: int,
        route_scale: float,
        bias_update_speed: float,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim

        self.gate = Gate(
            dim, n_routed_experts, n_activated_experts, route_scale, bias_update_speed
        )
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

        # Packed gate+up projection per expert, and down projection per expert.
        self.experts_w1 = nn.Parameter(torch.empty(n_routed_experts, 2 * moe_inter_dim, dim))
        self.experts_w2 = nn.Parameter(torch.empty(n_routed_experts, dim, moe_inter_dim))

        self.shared_experts = SwiGLUExpert(dim, n_shared_experts * moe_inter_dim)

    def hidden_params(self):
        return [self.experts_w1, self.experts_w2, self.shared_experts.w1.weight, self.shared_experts.w2.weight]

    def residual_params(self):
        return [self.experts_w2, self.shared_experts.w2.weight]

    def _grouped_mm_available(self, x) -> bool:
        # grouped_mm needs CUDA + bf16 operands. Check device only, not x's
        # incoming dtype, since under torch.autocast(dtype=bf16) tensors
        # crossing op boundaries are often still fp32-typed even though
        # compute happens in bf16 -- explicitly cast to bf16 below instead.
        return x.is_cuda

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        N = x_flat.size(0)

        weights, topk_idx = self.gate(x_flat)  # (N, k), (N, k)
        k = self.n_activated_experts

        expert_id_flat = topk_idx.reshape(-1)
        token_idx_flat = torch.arange(N, device=x.device).repeat_interleave(k)
        weight_flat = weights.reshape(-1)

        sort_idx = expert_id_flat.argsort()
        sorted_expert_ids = expert_id_flat[sort_idx]
        sorted_token_idx = token_idx_flat[sort_idx]
        sorted_weights = weight_flat[sort_idx]

        x_sorted = x_flat[sorted_token_idx]

        if self._grouped_mm_available(x):
            # grouped_mm requires bf16 operands; cast explicitly rather than
            # storing bf16 Parameters, since muP init/Muon updates want
            # full-precision master weights.
            x_sorted_bf16 = x_sorted.to(torch.bfloat16)
            w1 = self.experts_w1.transpose(-2, -1).to(torch.bfloat16)
            w2 = self.experts_w2.transpose(-2, -1).to(torch.bfloat16)
            counts = torch.bincount(sorted_expert_ids, minlength=self.n_routed_experts)
            offs = counts.cumsum(0).to(torch.int32)
            gu = F.grouped_mm(x_sorted_bf16, w1, offs=offs)
            gate_h, up_h = gu.chunk(2, dim=-1)
            h = F.silu(gate_h) * up_h
            out_sorted = F.grouped_mm(h, w2, offs=offs).to(x.dtype)
        else:
            # Eager per-expert reference path (CPU / no CUDA): masked loop,
            # not batched, used for correctness checks and non-CUDA smoke tests.
            out_sorted = torch.empty(x_sorted.size(0), D, device=x.device, dtype=x.dtype)
            for e in range(self.n_routed_experts):
                mask = sorted_expert_ids == e
                if not mask.any():
                    continue
                xe = x_sorted[mask]
                gate_h, up_h = (xe @ self.experts_w1[e].t()).chunk(2, dim=-1)
                h = F.silu(gate_h) * up_h
                out_sorted[mask] = (h @ self.experts_w2[e].t()).to(x.dtype)

        out_sorted = out_sorted * sorted_weights.unsqueeze(-1).to(out_sorted.dtype)

        out_flat = torch.zeros(N, D, device=x.device, dtype=out_sorted.dtype)
        out_flat.index_add_(0, sorted_token_idx, out_sorted)
        out_flat = out_flat + self.shared_experts(x_flat).to(out_flat.dtype)

        return out_flat.view(B, T, D)


def block_attn_res(
    blocks: tuple[torch.Tensor, ...],
    partial: torch.Tensor,
    proj: nn.Parameter,
    norm: nn.RMSNorm,
) -> torch.Tensor:
    """Attention Residuals gate (MoonshotAI, https://github.com/MoonshotAI/Attention-Residuals).

    Softmax-attends over completed block representations plus the current
    in-progress partial block, replacing the plain additive residual sum with
    a learned, input-dependent combination. ``proj`` is a per-layer pseudo-query
    vector (not a Linear: a real dim->1 Linear would land in Muon's 2D-matrix
    group via muon_param_groups, where Newton-Schulz orthogonalization on a
    rank-1 matrix would destroy its learned scale).

    blocks: N completed block representations, each (B, T, D)
    partial: (B, T, D) in-progress partial-block sum
    Returns: (B, T, D) gated combination, standing in for the residual stream.
    """
    V = torch.stack(blocks + (partial,), dim=0)  # (N+1, B, T, D)
    K = norm(V)
    logits = torch.einsum("d,nbtd->nbt", proj, K)
    weights = logits.softmax(dim=0)
    return torch.einsum("nbt,nbtd->btd", weights, V)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        mlp_hidden_dim: int,
        use_attn_res: bool = False,
        layer_number: int | None = None,
        layers_per_block: int | None = None,
        use_mla: bool = False,
        kv_lora_rank: int = 128,
        q_lora_rank: int = 0,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        use_moe: bool = False,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        n_activated_experts: int = 2,
        moe_inter_dim: int | None = None,
        route_scale: float = 1.0,
        bias_update_speed: float = 0.001,
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(dim)
        self.ln2 = nn.RMSNorm(dim)

        if use_mla:
            self.attn = MultiHeadLatentAttention(
                dim=dim,
                n_head=n_head,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
            )
        else:
            self.attn = GroupedQueryAttention(dim, n_head, n_kv_head)

        if use_moe:
            self.mlp = DeepSeekMoE(
                dim=dim,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                n_activated_experts=n_activated_experts,
                moe_inter_dim=moe_inter_dim if moe_inter_dim is not None else dim // 2,
                route_scale=route_scale,
                bias_update_speed=bias_update_speed,
            )
        else:
            self.mlp = MLP(dim, mlp_hidden_dim)

        self.use_attn_res = use_attn_res
        if use_attn_res:
            assert layer_number is not None and layers_per_block is not None
            self.layer_number = layer_number
            self.layers_per_block = layers_per_block
            # Pseudo-query vectors (see block_attn_res docstring), one per
            # gate; zero-init -> uniform softmax at init (== plain averaging).
            self.attn_res_proj = nn.Parameter(torch.zeros(dim))
            self.mlp_res_proj = nn.Parameter(torch.zeros(dim))
            self.attn_res_norm = nn.RMSNorm(dim)
            self.mlp_res_norm = nn.RMSNorm(dim)
        else:
            # Per-channel residual-branch gains; shape (dim,) broadcasts over
            # (B, T, dim) exactly like the earlier (1, 1, dim), but the
            # degenerate leading dims made torch.compile's backward return a
            # squeezed [1, dim] gradient at some widths (invalid-gradient
            # RuntimeError, seen at n_embd=96). The pre-hook below reshapes
            # legacy (1, 1, dim) checkpoint entries.
            self.scale1 = nn.Parameter(torch.ones(dim))
            self.scale2 = nn.Parameter(torch.ones(dim))
            self._register_load_state_dict_pre_hook(self._reshape_legacy_scales)

    def _reshape_legacy_scales(self, state_dict, prefix, *args):
        for name in ("scale1", "scale2"):
            key = prefix + name
            if key in state_dict and state_dict[key].ndim == 3:
                state_dict[key] = state_dict[key].reshape(-1)

    def forward(self, x, cos, sin, block_mask, past_kv=None):
        attn_out, present = self.attn(self.ln1(x), cos, sin, block_mask, past_kv)
        x = x + attn_out * self.scale1

        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out * self.scale2
        return x, present

    def forward_attn_res(self, blocks, partial, cos, sin, block_mask, past_kv=None):
        """Attention-Residuals forward: threads (blocks, partial) instead of x.

        ``blocks`` accumulates immutably (a new tuple each call) so this
        remains idempotent under activation-checkpoint recomputation.
        """
        h = block_attn_res(blocks, partial, self.attn_res_proj, self.attn_res_norm)
        if self.layer_number % self.layers_per_block == 0:
            blocks = blocks + (partial,)
            partial = None
        attn_out, present = self.attn(self.ln1(h), cos, sin, block_mask, past_kv)
        partial = attn_out if partial is None else partial + attn_out

        h2 = block_attn_res(blocks, partial, self.mlp_res_proj, self.mlp_res_norm)
        mlp_out = self.mlp(self.ln2(h2))
        partial = partial + mlp_out
        return blocks, partial, present


class GPT(nn.Module):
    """A decoder-only Transformer (GPT-style) for character-level language modeling.

    Uses grouped-query attention (or, with ``use_mla``, DeepSeek-V3 Multi-head
    Latent Attention) and rotary position embeddings, plus a KV cache during
    generation. The FFN is a dense MLP by default, or DeepSeek MoE (aux-loss-
    free, fine-grained routed experts + shared expert(s)) with ``use_moe``.
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
        use_attn_res: bool = False,
        attn_res_n_blocks: int = 8,
        use_mla: bool = False,
        kv_lora_rank: int = 128,
        q_lora_rank: int = 0,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        use_moe: bool = False,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        n_activated_experts: int = 2,
        moe_inter_dim: int | None = None,
        route_scale: float = 1.0,
        bias_update_speed: float = 0.001,
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
        # MLA's RoPE only covers the small decoupled qk_rope_head_dim slice,
        # not the full per-head dim GQA uses.
        self.rope = RotaryEmbedding(qk_rope_head_dim if use_mla else n_embd // n_head)

        # Attention Residuals (MoonshotAI): replaces the plain additive residual
        # with learned softmax attention over depth. attn_res_n_blocks == n_layer
        # is "Full AttnRes" (attention over every layer's output); smaller values
        # are "Block AttnRes" (attention over block-level summaries, O(N*d) memory).
        self.use_attn_res = use_attn_res
        if use_attn_res:
            if n_layer % attn_res_n_blocks != 0:
                raise ValueError(
                    f"n_layer ({n_layer}) must be divisible by attn_res_n_blocks "
                    f"({attn_res_n_blocks}); use attn_res_n_blocks == n_layer for "
                    f"Full AttnRes."
                )
            layers_per_block = n_layer // attn_res_n_blocks
        else:
            layers_per_block = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=n_embd,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    mlp_hidden_dim=4 * n_embd,
                    use_attn_res=use_attn_res,
                    layer_number=i if use_attn_res else None,
                    layers_per_block=layers_per_block,
                    use_mla=use_mla,
                    kv_lora_rank=kv_lora_rank,
                    q_lora_rank=q_lora_rank,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    v_head_dim=v_head_dim,
                    use_moe=use_moe,
                    n_routed_experts=n_routed_experts,
                    n_shared_experts=n_shared_experts,
                    n_activated_experts=n_activated_experts,
                    moe_inter_dim=moe_inter_dim,
                    route_scale=route_scale,
                    bias_update_speed=bias_update_speed,
                )
                for i in range(n_layer)
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

        # Architecture-agnostic: GQA/MLA and dense-MLP/DeepSeekMoE each expose
        # hidden_params()/residual_params() so this loop doesn't need to know
        # which variant is active. Router/gate weights are excluded from both
        # (initialized separately, width-independent, inside DeepSeekMoE).
        for block in self.blocks:
            for p in block.attn.hidden_params() + block.mlp.hidden_params():
                nn.init.normal_(p, mean=0.0, std=hidden_std)
            # Scale residual-writing projections by 1/sqrt(2 * n_layer) so the
            # residual stream variance stays bounded with depth (GPT-2), on top of
            # the muP width scaling already baked into hidden_std.
            residual_std = hidden_std / math.sqrt(2 * self.n_layer)
            for p in block.attn.residual_params() + block.mlp.residual_params():
                nn.init.normal_(p, mean=0.0, std=residual_std)

            if block.use_attn_res:
                nn.init.zeros_(block.attn_res_proj)
                nn.init.zeros_(block.mlp_res_proj)
                # attn_res_norm / mlp_res_norm keep RMSNorm's default ones-init gain.

        # If a genuinely untied head ever exists, initialize it at readout scale.
        # (Currently the head is always tied to tok_emb, so this is a no-op.)
        head = getattr(self, "head", None)
        if head is not None and head.weight is not self.tok_emb.weight:
            nn.init.normal_(head.weight, mean=0.0, std=self.mup_base_std)

    @torch._dynamo.disable
    def _get_block_mask(self, B: int, T: int, past_len: int, device) -> BlockMask:
        """Acquire (building or reusing) the flex_attention block mask.

        Forced to run eager (never traced/cudagraph-captured): under
        torch.compile(model, mode="reduce-overhead") the whole forward,
        including this call, would otherwise get captured by cudagraph
        trees -- but the mask cache holds a plain Python-attribute tensor
        reference across calls, and cudagraph replay overwrites that
        buffer on the next step, corrupting the "cached" mask silently.
        A graph break here makes create_block_mask's output a genuine
        input to the compiled region instead of captured internal state.
        """
        if past_len == 0:
            # Prefill/training: T is constant for a whole run (== block_size),
            # so cache the causal mask across calls instead of rebuilding it
            # every step -- building one costs ~4x the compiled attention
            # call itself.
            cache_key = (B, T, str(device))
            if getattr(self, "_block_mask_cache_key", None) != cache_key:
                self._block_mask_cache = build_block_mask(B, T, 0, device)
                self._block_mask_cache_key = cache_key
            return self._block_mask_cache
        # Decode: KV_LEN grows every step, so a fresh mask is unavoidable
        # here -- still cheap, since Q_LEN is 1.
        return build_block_mask(B, T, past_len, device)

    def _forward(self, idx, past_kvs=None):
        B, T = idx.shape
        if past_kvs is None:
            past_len = 0
            past_kvs = [None] * len(self.blocks)
        else:
            past_len = self.blocks[0].attn.cached_len(past_kvs[0])

        x = self.tok_emb(idx) * self.mup_input_mult  # muP embedding multiplier
        cos, sin = self.rope(past_len, T, idx.device)
        block_mask = self._get_block_mask(B, T, past_len, idx.device)

        # Activation checkpointing only applies to the training/prefill path (no
        # KV cache); incremental decode runs under no_grad, where it's a no-op.
        use_ckpt = self.gradient_checkpointing and self.training and past_kvs[0] is None

        presents = []
        if self.use_attn_res:
            blocks: tuple[torch.Tensor, ...] = ()
            partial = x
            for block, past in zip(self.blocks, past_kvs):
                if use_ckpt:
                    blocks, partial, present = checkpoint.checkpoint(
                        block.forward_attn_res, blocks, partial, cos, sin, block_mask, past,
                        use_reentrant=False,
                    )
                else:
                    blocks, partial, present = block.forward_attn_res(
                        blocks, partial, cos, sin, block_mask, past
                    )
                presents.append(present)
            x = partial
        else:
            for block, past in zip(self.blocks, past_kvs):
                if use_ckpt:
                    x, present = checkpoint.checkpoint(
                        block, x, cos, sin, block_mask, past, use_reentrant=False
                    )
                else:
                    x, present = block(x, cos, sin, block_mask, past)
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

    print("\n--- MLA + DeepSeek MoE ---")
    model_ds = GPT(
        vocab_size=256,
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6,
        use_mla=True,
        kv_lora_rank=64,
        q_lora_rank=0,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        use_moe=True,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
    )
    summary(
        model_ds, input_size=(1, 256), col_names=["output_size", "num_params", "mult_adds"], dtypes=[torch.int64]
    )
    with torch.no_grad():
        x = torch.randint(0, 256, (32, 256), dtype=torch.long, device=next(model_ds.parameters()).device)
        logits = model_ds(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        print(f"loss: {loss.item():.4f} (should be ~{math.log(256):.4f})")

    # KV-cache path (exercises MLA's (c_kv, k_rope) cache format via generate()).
    prompt = torch.randint(0, 256, (1, 8), dtype=torch.long, device=next(model_ds.parameters()).device)
    out = model_ds.generate(prompt, max_new_tokens=8)
    print(f"generate() output shape: {tuple(out.shape)}")
