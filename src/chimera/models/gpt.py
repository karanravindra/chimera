import math
import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from chimera.models.rope import RotaryEmbedding, apply_rotary

# Cut Cross Entropy (the training loss for this model, used lazily by
# LanguageModelModule) ships hardcoded Triton block configs tuned on Apple's
# dev hardware; letting Triton autotune instead is ~15% faster on both CCE
# kernels (~8% whole train step, measured in projects/llm/gpt/mfu_bench.py).
# The flag is read when cut_cross_entropy is first imported, so defaulting it
# here -- at model import, which always precedes the loss -- covers every
# consumer of this model without each script having to remember it.
os.environ.setdefault("CCE_AUTOTUNE", "1")

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


def _make_sparse_mask_mod(past_len: int, window: int, n_global: int):
    """Causal sliding-window + global-prefix mask (Longformer/attention-sink
    style): each query attends to the previous ``window`` tokens plus the
    first ``n_global`` tokens of the sequence. ``past_len`` offsets query
    positions during incremental decode."""

    def mask_mod(b, h, q_idx, kv_idx):
        q_abs = q_idx + past_len
        return (kv_idx <= q_abs) & ((q_abs - kv_idx < window) | (kv_idx < n_global))

    return mask_mod


def build_block_mask(
    B: int, T: int, past_len: int, device, window: int | None = None, n_global: int = 0
) -> BlockMask:
    """Block mask for flex_attention, covering both prefill (past_len=0,
    Q_LEN==KV_LEN==T) and incremental decode (query positions offset by the
    cached prefix length). Plain causal by default; ``window`` switches to
    causal sliding-window + ``n_global`` global-prefix tokens, where the
    BlockMask actually prunes fully-masked KV blocks (real compute savings,
    not just masking).

    Building a fresh BlockMask every call has real cost even under compile
    (~4x the compiled attention call itself, measured) -- callers should cache
    the prefill mask across steps (T is constant for a whole training run) and
    only rebuild for decode, where KV_LEN changes every step anyway.
    """
    if window is not None and window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if n_global < 0:
        raise ValueError(f"n_global must be >= 0, got {n_global}")

    if window is not None:
        mask_mod = _make_sparse_mask_mod(past_len, window, n_global)
    elif past_len == 0:
        mask_mod = _causal_mask_mod
    else:
        mask_mod = _make_offset_causal_mask_mod(past_len)
    return create_block_mask(mask_mod, B, None, T, past_len + T, device=device)


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

    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        attn_scale: float | None = None,
    ):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.attn_scale = attn_scale

        # Q keeps n_head heads; K and V share only n_kv_head heads (the GQA fix).
        self.q_proj = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, n_kv_head * self.head_dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # QK-norm: per-head RMSNorm over the head dimension before attention.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def hidden_params(self):
        return [self.q_proj.weight, self.kv_proj.weight, self.proj.weight]

    def residual_params(self):
        return [self.proj.weight]

    @staticmethod
    def cached_len(past_kv) -> int:
        return past_kv[0].size(2)

    def forward(self, x, cos, sin, block_mask, past_kv=None, use_cache: bool = True):
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
            if block_mask is None:
                raise ValueError(
                    "Cached attention requires an offset-aware block_mask; "
                    "SDPA is_causal=True uses the wrong alignment for Q_LEN < KV_LEN."
                )
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v) if use_cache else None

        # block_mask is built once per GPT._forward call (prefill: cached
        # across steps since T is constant for a whole run; decode: fresh
        # each step, since KV_LEN grows) and threaded down alongside cos/sin.
        # block_mask=None signals pure causal (GPT.use_flash_attn): dispatch
        # to SDPA's fused FlashAttention/cuDNN kernels, which beat flex's
        # Triton templates at small head_dim. flex handles everything else
        # (decode's offset mask, future custom/sparse masks).
        # SDPA's native GQA support is backend-limited. Expanding K/V on
        # non-CUDA devices provides a portable math-kernel fallback.
        enable_gqa = self.n_head != self.n_kv_head
        if enable_gqa and q.device.type != "cuda":
            groups = self.n_head // self.n_kv_head
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            enable_gqa = False

        if block_mask is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                enable_gqa=enable_gqa,
                scale=self.attn_scale,
            )
        else:
            out = _flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                enable_gqa=enable_gqa,
                scale=self.attn_scale,
            )

        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.proj(out), present


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-style Multi-head Latent Attention.

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
        self.ckv_norm = nn.RMSNorm(kv_lora_rank, eps=1e-6)
        self.w_uk = nn.Linear(kv_lora_rank, n_head * qk_nope_head_dim, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, n_head * v_head_dim, bias=False)

        if q_lora_rank > 0:
            self.w_dq = nn.Linear(dim, q_lora_rank, bias=False)
            self.cq_norm = nn.RMSNorm(q_lora_rank, eps=1e-6)
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

    def forward(self, x, cos, sin, block_mask, past_kv=None, use_cache: bool = True):
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
        # Cache the normalized latent. RMSNorm is token-local, so this is
        # exactly equivalent to normalizing the concatenated cache every step
        # and avoids repeatedly normalizing the entire prefix during decode.
        c_kv = self.ckv_norm(c_kv)
        k_rope = k_rope.unsqueeze(1)  # (B, 1, T, qk_rope_head_dim): shared across heads
        k_rope = apply_rotary(k_rope, cos, sin)

        if past_kv is not None:
            if block_mask is None:
                raise ValueError(
                    "Cached attention requires an offset-aware block_mask; "
                    "SDPA is_causal=True uses the wrong alignment for Q_LEN < KV_LEN."
                )
            past_c_kv, past_k_rope = past_kv
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
        present = (c_kv, k_rope) if use_cache else None

        # This is the straightforward MLA path: compressed cache storage is
        # retained, but K/V up-projections are not algebraically absorbed as in
        # DeepSeek's optimized inference kernel.
        Tk = c_kv.size(1)
        k_nope = (
            self.w_uk(c_kv).view(B, Tk, self.n_head, self.qk_nope_head_dim).transpose(1, 2)
        )
        v = self.w_uv(c_kv).view(B, Tk, self.n_head, self.v_head_dim).transpose(1, 2)
        k = torch.cat([k_nope, k_rope.expand(-1, self.n_head, -1, -1)], dim=-1)

        # See GroupedQueryAttention: block_mask=None means pure causal -> SDPA.
        if block_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
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
    """DeepSeek-V3-style aux-loss-free router.

    Sigmoid affinity scores (not softmax) against learnable per-expert
    centroids. A per-expert bias is added to the scores for top-k *selection*
    only; the combining weight is the raw (bias-free) score of the selected
    experts, renormalized among the top-k. The bias is a plain buffer nudged
    by a fixed step after each optimizer step based on accumulated load.

    Forward accumulates expert-load statistics but does not mutate the routing
    bias. Call ``GPT.update_moe_bias()`` exactly once after each optimizer step;
    this correctly combines gradient-accumulation microbatches and optionally
    synchronizes counts across distributed ranks. Checkpoint recomputation is
    explicitly prevented from collecting the same assignments twice.
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
        # Keep transient statistics out of registered buffers: DDP broadcasts
        # buffers from rank 0 before each forward by default, which would
        # corrupt rank-local counts before the explicit all-reduce at update.
        self.pending_counts = torch.zeros(n_routed_experts)
        self._collect_stats = True

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        moved_counts = fn(self.pending_counts)
        # These are transient statistics, so reset them on device/dtype
        # materialization and always retain FP32 count precision. Keep the
        # routing bias FP32 as in the DeepSeek reference as well.
        self.pending_counts = torch.zeros_like(moved_counts, dtype=torch.float32)
        self.bias = self.bias.float()
        return self

    @contextmanager
    def disable_stats(self):
        """Temporarily suppress load-stat collection during checkpoint replay."""
        previous = self._collect_stats
        self._collect_stats = False
        try:
            yield
        finally:
            self._collect_stats = previous

    @torch.no_grad()
    def update_bias(self, sync_distributed: bool = True) -> bool:
        """Apply one load-balancing update from all accumulated microbatches."""
        counts = self.pending_counts.clone()
        self.pending_counts.zero_()
        if sync_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if counts.sum().item() == 0:
            return False
        self.bias.add_(
            self.bias_update_speed * torch.sign(counts.mean() - counts)
        )
        return True

    @torch.no_grad()
    def reset_pending_counts(self):
        self.pending_counts.zero_()

    def forward(self, x):
        # x: (N, dim) flattened tokens
        scores = torch.sigmoid(x.float() @ self.weight.float().t())  # (N, n_routed_experts)
        topk_idx = (scores + self.bias).topk(self.n_activated_experts, dim=-1).indices
        topk_scores = scores.gather(-1, topk_idx)
        weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        weights = (weights * self.route_scale).to(x.dtype)

        if self.training and self._collect_stats:
            with torch.no_grad():
                counts = torch.bincount(
                    topk_idx.flatten(), minlength=self.n_routed_experts
                ).float()
                self.pending_counts.add_(counts)

        return weights, topk_idx


class DeepSeekMoE(nn.Module):
    """DeepSeek-V3 style MoE FFN: fine-grained routed experts (top-k, aux-loss-free
    balancing via ``Gate``) plus always-on shared expert(s), combined as
    ``y = shared(x) + sum_{i in topk} weight_i * expert_i(x)``.

    Routed-expert compute uses ``torch.nn.functional.grouped_mm`` (batched GEMM
    over jagged per-expert token counts) on supported CUDA devices, preserving
    explicit FP32 execution or using the active autocast dtype. Other devices
    fall back to an eager masked loop over experts, which also serves as a
    numerical reference path.
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
        # Keep standalone DeepSeekMoE construction safe. GPT._init_weights will
        # reinitialize these using its configured muP/depth-aware scales.
        nn.init.normal_(self.experts_w1, mean=0.0, std=0.02)
        nn.init.normal_(self.experts_w2, mean=0.0, std=0.02)

        self.shared_experts = SwiGLUExpert(dim, n_shared_experts * moe_inter_dim)

    def hidden_params(self):
        return [
            self.experts_w1,
            self.experts_w2,
            self.shared_experts.w1.weight,
            self.shared_experts.w2.weight,
        ]

    def residual_params(self):
        return [self.experts_w2, self.shared_experts.w2.weight]

    def _grouped_mm_available(self, x) -> bool:
        if not x.is_cuda or not hasattr(F, "grouped_mm"):
            return False
        compute_dtype = self._grouped_mm_dtype(x)
        if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        major, _ = torch.cuda.get_device_capability(x.device)
        return major >= 8

    @staticmethod
    def _grouped_mm_dtype(x):
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        return x.dtype

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
            # Keep full-precision master parameters, but respect explicit FP32
            # execution and the active autocast dtype instead of forcing BF16.
            compute_dtype = self._grouped_mm_dtype(x)
            x_grouped = x_sorted.to(compute_dtype)
            w1 = self.experts_w1.transpose(-2, -1).to(compute_dtype)
            w2 = self.experts_w2.transpose(-2, -1).to(compute_dtype)
            counts = torch.bincount(sorted_expert_ids, minlength=self.n_routed_experts)
            offs = counts.cumsum(0).to(torch.int32)
            gu = F.grouped_mm(x_grouped, w1, offs=offs)
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
        attn_scale: float | None = None,
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
        residual_mult: float = 1.0,
    ):
        super().__init__()
        # Depth-muP: a persistent (non-learned) scale on each residual BRANCH,
        # residual_mult = sqrt(mup_base_depth / n_layer). Unlike the GPT-2
        # 1/sqrt(2L) *init*, this multiplier stays in the forward pass throughout
        # training, so residual-stream magnitude stays ~O(1) across depth and the
        # LR/init optimum transfers across n_layer (not just width). == 1.0 (no-op)
        # when n_layer == mup_base_depth, so the depth-6 baseline is unchanged.
        self.residual_mult = float(residual_mult)
        self.ln1 = nn.RMSNorm(dim, eps=1e-6)
        self.ln2 = nn.RMSNorm(dim, eps=1e-6)

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
            self.attn = GroupedQueryAttention(
                dim, n_head, n_kv_head, attn_scale=attn_scale
            )

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
            self.attn_res_norm = nn.RMSNorm(dim, eps=1e-6)
            self.mlp_res_norm = nn.RMSNorm(dim, eps=1e-6)
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

    def forward(
        self, x, cos, sin, block_mask, past_kv=None, use_cache: bool = True
    ):
        attn_out, present = self.attn(
            self.ln1(x), cos, sin, block_mask, past_kv, use_cache
        )
        x = x + attn_out * self.scale1 * self.residual_mult

        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out * self.scale2 * self.residual_mult
        return x, present

    def forward_attn_res(
        self,
        blocks,
        partial,
        cos,
        sin,
        block_mask,
        past_kv=None,
        use_cache: bool = True,
    ):
        """Attention-Residuals forward: threads (blocks, partial) instead of x.

        ``blocks`` accumulates immutably (a new tuple each call) so this
        remains idempotent under activation-checkpoint recomputation.
        """
        h = block_attn_res(blocks, partial, self.attn_res_proj, self.attn_res_norm)
        if self.layer_number % self.layers_per_block == 0:
            blocks = blocks + (partial,)
            partial = None
        attn_out, present = self.attn(
            self.ln1(h), cos, sin, block_mask, past_kv, use_cache
        )
        attn_out = attn_out * self.residual_mult  # Depth-muP branch scale
        partial = attn_out if partial is None else partial + attn_out

        h2 = block_attn_res(blocks, partial, self.mlp_res_proj, self.mlp_res_norm)
        mlp_out = self.mlp(self.ln2(h2)) * self.residual_mult  # Depth-muP branch scale
        partial = partial + mlp_out
        return blocks, partial, present


class GPT(nn.Module):
    """A decoder-only Transformer (GPT-style) for character-level language modeling.

    Uses grouped-query attention (or, with ``use_mla``, DeepSeek-style Multi-head
    Latent Attention) and rotary position embeddings, plus a KV cache during
    generation. The FFN is a dense MLP by default, or DeepSeek-style MoE (aux-loss-
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
        mup_base_depth: int = 6,
        mup_base_std: float = 0.02,
        mup_input_mult: float = 1.0,
        mup_output_mult: float = 1.0,
        gradient_checkpointing: bool = False,
        use_flash_attn: bool = True,
        attn_window: int | None = None,
        attn_global_tokens: int = 16,
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
        if vocab_size < 1 or block_size < 1 or n_embd < 1:
            raise ValueError("vocab_size, block_size, and n_embd must all be >= 1")
        if mup_base_width < 1:
            raise ValueError(f"mup_base_width must be >= 1, got {mup_base_width}")
        if mup_base_depth < 1:
            raise ValueError(f"mup_base_depth must be >= 1, got {mup_base_depth}")
        if n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {n_layer}")
        if n_head < 1 or n_kv_head < 1:
            raise ValueError("n_head and n_kv_head must both be >= 1")
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
            )
        if n_head % n_kv_head != 0:
            raise ValueError(
                f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
            )
        if attn_window is not None and attn_window < 1:
            raise ValueError(f"attn_window must be >= 1, got {attn_window}")
        if attn_global_tokens < 0:
            raise ValueError(
                f"attn_global_tokens must be >= 0, got {attn_global_tokens}"
            )
        if use_attn_res and attn_res_n_blocks < 1:
            raise ValueError(
                f"attn_res_n_blocks must be >= 1, got {attn_res_n_blocks}"
            )
        if use_moe and not (1 <= n_activated_experts <= n_routed_experts):
            raise ValueError(
                "n_activated_experts must be between 1 and n_routed_experts"
            )
        self.block_size = block_size
        self.n_layer = n_layer
        # Recompute each block's activations in the backward pass instead of
        # storing them — trades ~1 extra forward of compute for a large drop in
        # activation memory. Needed to fit wide models (e.g. 124M @ seq 2048) on
        # 16 GB without shrinking the (comparability-critical) tokens/step.
        self.gradient_checkpointing = gradient_checkpointing
        # Pure-causal attention (training / prefill) dispatches to SDPA's fused
        # FlashAttention/cuDNN kernels instead of flex_attention -- measurably
        # faster at small head_dim, where flex's Triton templates lag. flex is
        # kept for anything needing a mask_mod: incremental decode (offset
        # causal) and sparse attention masks.
        self.use_flash_attn = use_flash_attn
        # Sparse attention: causal sliding window of ``attn_window`` tokens
        # plus ``attn_global_tokens`` always-visible prefix tokens (attention
        # sinks). None = dense causal. When set, attention always runs through
        # flex_attention with a block mask that prunes fully-masked KV blocks,
        # so long-context attention cost grows ~linearly instead of
        # quadratically. use_flash_attn's SDPA fast path only applies to the
        # dense-causal case and is bypassed when a window is set.
        self.attn_window = attn_window
        self.attn_global_tokens = attn_global_tokens

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
        # Depth-muP: persistent residual-branch scale = sqrt(base_depth / n_layer).
        # Anchored so n_layer == mup_base_depth gives 1.0 (the depth-6 baseline is
        # byte-for-byte unchanged). Combined with the base-depth-anchored residual
        # init below, the residual-stream init variance matches the old GPT-2
        # 1/sqrt(2L) scheme at every depth -- the ONLY change is that the depth
        # factor now lives in the forward pass (so it persists through training),
        # which is what makes the LR/init optimum transfer across depth.
        self.mup_base_depth = mup_base_depth
        self.residual_mult = math.sqrt(mup_base_depth / n_layer)
        self.mup_base_std = mup_base_std
        self.mup_input_mult = mup_input_mult
        # 1/m_d output scaling (multiplier form of the muP readout), applied at the
        # logit projection instead of scaling the (tied) head's learning rate.
        self.output_mult = mup_output_mult / self.mup_width_mult

        # Canonical muP attention scaling, backward-compatible with standard
        # 1/sqrt(head_dim) scaling at the configured base width:
        # sqrt(base_head_dim) / head_dim. MLA's QK width is configured
        # independently of model width, so its standard scaling is unchanged.
        if use_mla:
            attn_scale = None
        else:
            head_dim = n_embd // n_head
            base_head_dim = mup_base_width / n_head
            attn_scale = math.sqrt(base_head_dim) / head_dim

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
            # The paper requires the output layer to aggregate all completed
            # blocks plus the final partial block with its own pseudo-query.
            self.attn_res_output_proj = nn.Parameter(torch.zeros(n_embd))
            self.attn_res_output_norm = nn.RMSNorm(n_embd, eps=1e-6)
        else:
            layers_per_block = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=n_embd,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    mlp_hidden_dim=4 * n_embd,
                    attn_scale=attn_scale,
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
                    residual_mult=self.residual_mult,
                )
                for i in range(n_layer)
            ]
        )
        self.ln_f = nn.RMSNorm(n_embd, eps=1e-6)

        if not tie_embedding:
            self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        # muP init: hidden-matrix variance is scaled by 1/m_d so activation
        # magnitudes stay width-invariant; the embedding (input layer) keeps a
        # width-independent std. Without embedding init nn.Embedding defaults to
        # N(0, 1), and since the head is tied to it the logits blow up (~10x std),
        # starting the loss far above ln(vocab) instead of at it.
        hidden_std = self.mup_base_std / math.sqrt(self.mup_width_mult)

        # Embedding (input layer): constant std, no width scaling. When the
        # readout is tied, the muP output behavior comes from self.output_mult
        # (the 1/m_d factor).
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.mup_base_std)

        # Architecture-agnostic: GQA/MLA and dense-MLP/DeepSeekMoE each expose
        # hidden_params()/residual_params() so this loop doesn't need to know
        # which variant is active. Router/gate weights are excluded from both
        # (initialized separately, width-independent, inside DeepSeekMoE).
        for block in self.blocks:
            for p in block.attn.hidden_params() + block.mlp.hidden_params():
                nn.init.normal_(p, mean=0.0, std=hidden_std)
            if isinstance(block.mlp, DeepSeekMoE):
                # Router logits should retain the same scale as width changes.
                nn.init.normal_(block.mlp.gate.weight, mean=0.0, std=hidden_std)
            # Depth-muP residual init: anchored at mup_base_depth, not n_layer.
            # The remaining depth factor sqrt(base_depth/n_layer) lives in the
            # forward pass (block.residual_mult), so init variance here still
            # matches the old GPT-2 1/sqrt(2*n_layer) scheme at every depth
            # (residual_std * residual_mult == hidden_std/sqrt(2*n_layer)), while
            # the persistent forward factor is what enables depth transfer.
            residual_std = hidden_std / math.sqrt(2 * self.mup_base_depth)
            for p in block.attn.residual_params() + block.mlp.residual_params():
                nn.init.normal_(p, mean=0.0, std=residual_std)

            if block.use_attn_res:
                nn.init.zeros_(block.attn_res_proj)
                nn.init.zeros_(block.mlp_res_proj)
                # AttnRes RMSNorm gains keep their default ones initialization.

        # Initialize a genuinely untied head at readout scale.
        head = getattr(self, "head", None)
        if head is not None and head.weight is not self.tok_emb.weight:
            nn.init.normal_(head.weight, mean=0.0, std=self.mup_base_std)

    def _checkpoint_kwargs(self, block: TransformerBlock) -> dict:
        kwargs = {"use_reentrant": False}
        if isinstance(block.mlp, DeepSeekMoE):
            gate = block.mlp.gate
            kwargs["context_fn"] = lambda: (nullcontext(), gate.disable_stats())
        return kwargs

    @torch.no_grad()
    def update_moe_bias(self, sync_distributed: bool = True) -> int:
        """Update all MoE routing biases once after an optimizer step.

        Gate forwards accumulate assignment counts across gradient-accumulation
        microbatches. With ``sync_distributed=True``, counts are summed across
        ranks before every replica applies the same update.

        Returns the number of gates that had observations and were updated.
        """
        updated = 0
        for block in self.blocks:
            if isinstance(block.mlp, DeepSeekMoE):
                updated += int(block.mlp.gate.update_bias(sync_distributed))
        return updated

    @torch.no_grad()
    def reset_moe_bias_statistics(self):
        """Discard accumulated router counts without changing routing bias."""
        for block in self.blocks:
            if isinstance(block.mlp, DeepSeekMoE):
                block.mlp.gate.reset_pending_counts()

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
            # so cache the mask across calls instead of rebuilding it
            # every step -- building one costs ~4x the compiled attention
            # call itself. attn_window/attn_global_tokens are fixed per model
            # instance, so they don't need to be part of the key.
            cache_key = (B, T, str(device))
            if getattr(self, "_block_mask_cache_key", None) != cache_key:
                self._block_mask_cache = build_block_mask(
                    B, T, 0, device, self.attn_window, self.attn_global_tokens
                )
                self._block_mask_cache_key = cache_key
            return self._block_mask_cache
        # Decode: KV_LEN grows every step, so a fresh mask is unavoidable
        # here -- still cheap, since Q_LEN is 1.
        return build_block_mask(
            B, T, past_len, device, self.attn_window, self.attn_global_tokens
        )

    def _forward(self, idx, past_kvs=None, use_cache: bool = True):
        B, T = idx.shape
        if past_kvs is not None and not use_cache:
            raise ValueError("past_kvs requires use_cache=True")
        if past_kvs is None:
            past_len = 0
            past_kvs = [None] * len(self.blocks)
        else:
            past_len = self.blocks[0].attn.cached_len(past_kvs[0])

        x = self.tok_emb(idx) * self.mup_input_mult  # muP embedding multiplier
        cos, sin = self.rope(past_len, T, idx.device)
        # block_mask=None -> pure causal via SDPA/flash in the attention
        # modules; a BlockMask -> flex_attention (decode offset / sparse masks).
        if self.use_flash_attn and past_len == 0 and self.attn_window is None:
            block_mask = None
        else:
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
                        block.forward_attn_res,
                        blocks,
                        partial,
                        cos,
                        sin,
                        block_mask,
                        past,
                        use_cache,
                        **self._checkpoint_kwargs(block),
                    )
                else:
                    blocks, partial, present = block.forward_attn_res(
                        blocks, partial, cos, sin, block_mask, past, use_cache
                    )
                if use_cache:
                    presents.append(present)
            x = block_attn_res(
                blocks,
                partial,
                self.attn_res_output_proj,
                self.attn_res_output_norm,
            )
        else:
            for block, past in zip(self.blocks, past_kvs):
                if use_ckpt:
                    x, present = checkpoint.checkpoint(
                        block,
                        x,
                        cos,
                        sin,
                        block_mask,
                        past,
                        use_cache,
                        **self._checkpoint_kwargs(block),
                    )
                else:
                    x, present = block(x, cos, sin, block_mask, past, use_cache)
                if use_cache:
                    presents.append(present)

        x = self.ln_f(x)
        return x, presents if use_cache else None  # hidden states (pre-projection)

    @property
    def lm_head_weight(self):
        """The classifier weight matrix (V, C), whether tied or a separate head."""
        return self.head.weight if hasattr(self, "head") else self.tok_emb.weight

    @torch.no_grad()
    def untie_head(self):
        """Fork the tied embedding into an independent output head, mid-training.

        Replicates modded-nanogpt's dynamic untie ("untie embed/lm_head at 2/3 of
        training"): the model trains with a single shared (tied) weight while it
        learns basic structure, then the head is split off into its own parameter
        so input and output representations can specialize for the final loss.

        The new head is initialized to the current tied weight (continuity, not a
        re-init). ``self.tok_emb`` keeps its own tensor identity, so the compiled
        ``forward(return_hidden=True)`` graph (which reads ``tok_emb`` for the input
        embedding but not the head) is unaffected — no recompile. Cut Cross Entropy
        re-reads ``lm_head_weight`` eagerly each step, so it picks up the new head
        immediately.

        Returns ``(tok_emb_weight, head_weight)`` so the optimizer can copy the
        embedding's moment estimates into the fresh head param, or ``None`` if the
        head already exists (idempotent).
        """
        if hasattr(self, "head"):
            return None
        emb = self.tok_emb.weight  # (V, C)
        head = nn.Linear(emb.size(1), emb.size(0), bias=False).to(
            device=emb.device, dtype=emb.dtype
        )
        head.weight.copy_(emb)
        self.head = head
        return self.tok_emb.weight, self.head.weight

    def project(self, hidden):
        """Project hidden states to vocabulary logits (with the muP output multiplier)."""
        return (hidden @ self.lm_head_weight.t()) * self.output_mult

    def forward(self, idx, return_hidden: bool = False):
        assert idx.size(1) <= self.block_size, (
            "Cannot forward, model block size is exhausted."
        )
        hidden, _ = self._forward(idx, past_kvs=None, use_cache=False)
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
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must have shape (batch, sequence) with a non-empty prompt")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if idx.size(1) + max_new_tokens > self.block_size:
            raise ValueError(
                f"prompt length ({idx.size(1)}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds block_size ({self.block_size}); this cached generator does "
                "not silently extrapolate or evict KV entries"
            )
        self.eval()

        step = self._forward
        if compile:
            if getattr(self, "_forward_compiled", None) is None:
                # dynamic=True: one compiled artifact serves both the variable-length
                # prefill and the length-1 decode step (no per-length recompiles).
                self._forward_compiled = torch.compile(self._forward, dynamic=True)
            step = self._forward_compiled

        # Prefill the cache with the prompt in a single pass. The validation
        # above guarantees that the cache remains within block_size.
        hidden, past_kvs = step(idx, past_kvs=None, use_cache=True)

        for _ in range(max_new_tokens):
            # Only the last position is needed to sample the next token.
            next_logits = self.project(hidden[:, -1, :]) / temperature
            probs = torch.softmax(next_logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            # Feed only the new token; RoPE/attention use the cached keys/values.
            hidden, past_kvs = step(nxt, past_kvs=past_kvs, use_cache=True)
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