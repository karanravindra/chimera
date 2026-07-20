"""tinylm GPT with Virtual Width Networks (VWN) and optional mHC-Lite carry routing.

Pre-norm Transformer with RoPE, QK-norm, ReLU^2 MLP, tied base
embeddings, and Generalized Hyper-Connections (GHC). The default constrains only the
square virtual-slot carry map using mHC Lite; read and write maps remain
the rectangular dynamic maps defined by VWN.

The persistent residual state has virtual width::

    virtual_dim = (vwn_n / vwn_m) * dim

while attention and the MLP continue to operate at backbone width ``dim``.
The default ``(vwn_m, vwn_n) = (2, 3)`` is the paper's practical 1.5x
configuration. Set ``vwn_m=vwn_n=1`` to recover ordinary residual
connections and the original model width.

This model resets RoPE positions per packed document (via ``pos_ids`` from
``build_block_mask_and_pos``) instead of relying on RoPE's relative-offset
invariance. It has no muP parameterization and is project-local so this experiment
remains the single source of truth for its architecture.

Looped depth (``n_loops > 1``): the ``n_layers`` unique blocks are applied
``n_loops`` times in sequence (Ouro-style fixed loop count, full unrolled
backprop). Before every loop after the first, the expanded input embedding is
re-injected Huginn-style by concatenation through a learned adapter whose
weight starts as ``[I | small]`` — identity on the running state, near-zero on
the injected input — so training begins as a plain carry. The fixed python
loop unrolls into a single static graph under torch.compile.
"""

import itertools
import math
from typing import Literal

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

        # QK-norm: per-head RMSNorm over the head dimension, applied before RoPE.
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

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
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
            # No block mask -> plain causal via FlashAttention (prefill / eval), or
            # full attention over the KV cache on a one-token decode step.
            if past_len == 0 and N > 1:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out(out)
        return out, new_kv


def safe_tanh(x: torch.Tensor) -> torch.Tensor:
    """Run tanh in fp32, then return to the activation dtype."""
    return torch.tanh(x.float()).to(dtype=x.dtype)


# The routing maps are tiny (n, m <= a handful), so expressing their application
# as batched matmuls dispatches cuBLAS gemv kernels per token — launch/bandwidth
# bound and unfusable. Broadcast multiply + sum keeps the same math but lets
# torch.compile fuse routing into one Triton kernel with the surrounding ops.


def apply_map(slots: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``out[..., j, :] = sum_i weight[..., i, j] * slots[..., i, :]``.

    Fusable equivalent of ``(slots^T @ weight)^T`` for [..., n, d] slots and
    [..., n, j] weights.
    """
    return (weight.unsqueeze(-1) * slots.unsqueeze(-2)).sum(-3)


def project_slots(slots: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``out[..., m] = sum_d slots[..., d] * weight[d, m]``.

    Fusable equivalent of ``slots @ weight`` for a static [d, m] weight with a
    tiny output dim ``m``.
    """
    return (slots.unsqueeze(-1) * weight).sum(-2)


def _all_permutation_indices(n: int) -> torch.Tensor:
    """Return every permutation of ``range(n)`` as a [n!, n] tensor."""
    return torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long)


def _sample_permutation_indices(
    n: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Build a deterministic permutation subset with identity at index zero."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")

    max_unique = math.factorial(n)
    if count > max_unique:
        raise ValueError(
            f"requested {count} unique permutations for n={n}, but only "
            f"{max_unique} exist"
        )

    identity = torch.arange(n, dtype=torch.long)
    permutations = [identity]
    seen = {tuple(identity.tolist())}

    # Add cyclic shifts first so a small subset has predictable global mixing.
    for shift in range(1, n):
        if len(permutations) >= count:
            break
        permutation = identity.roll(-shift)
        key = tuple(permutation.tolist())
        if key not in seen:
            permutations.append(permutation)
            seen.add(key)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    while len(permutations) < count:
        permutation = torch.randperm(n, generator=generator)
        key = tuple(permutation.tolist())
        if key in seen:
            continue
        permutations.append(permutation)
        seen.add(key)

    return torch.stack(permutations)


def make_permutation_indices(
    n: int,
    count: int | None = None,
    seed: int = 0,
    full_basis_max_slots: int = 4,
) -> torch.Tensor:
    """Create the fixed permutation basis used by mHC Lite.

    With ``count=None``, all n! permutations are used for n <= 4. For larger
    slot counts, a deterministic 32-permutation subset is used by default to
    avoid factorial growth. Any convex combination of the subset is still
    exactly doubly stochastic, but it spans only part of the Birkhoff polytope.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")

    total = math.factorial(n)
    if count is None:
        count = total if n <= full_basis_max_slots else min(32, total)

    if count == total and n <= full_basis_max_slots:
        return _all_permutation_indices(n)

    return _sample_permutation_indices(n=n, count=count, seed=seed)


class MHCLiteCarry(nn.Module):
    """Exactly doubly stochastic mixing over the VWN virtual slots.

    The token-dependent carry matrix is represented as a convex combination
    of fixed permutation matrices. One global mixture vector is produced for
    each token from its complete virtual-width state; using independent
    row-wise mixture weights would preserve row sums but not column sums.
    """

    def __init__(
        self,
        n_slots: int,
        block_dim: int,
        num_permutations: int | None = None,
        permutation_seed: int = 0,
        dynamic_scale_init: float = 1e-2,
        non_identity_logit: float = -8.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.n_slots = n_slots
        self.block_dim = block_dim
        self.virtual_dim = n_slots * block_dim

        permutation_indices = make_permutation_indices(
            n=n_slots,
            count=num_permutations,
            seed=permutation_seed,
        )

        identity = torch.arange(n_slots)
        identity_matches = (permutation_indices == identity).all(dim=-1)
        if not identity_matches.any():
            permutation_indices = torch.cat(
                [identity.unsqueeze(0), permutation_indices], dim=0
            )
            identity_index = 0
        else:
            identity_index = int(identity_matches.nonzero(as_tuple=False)[0, 0])

        # If P = eye[permutation], VWN's carry is P^T @ slots. Indexing slots
        # by inverse_permutation produces the same result without materializing
        # a dense [K, n, n] basis in the forward pass.
        inverse_permutations = permutation_indices.argsort(dim=-1)
        self.register_buffer("permutation_indices", permutation_indices)
        self.register_buffer("inverse_permutations", inverse_permutations)
        # Dense [K, n, n] basis. n is small, so mixing via the dense matrix is
        # cheaper (and fusable) versus gathering K permuted copies of the slots.
        self.register_buffer(
            "permutation_basis", torch.eye(n_slots)[permutation_indices]
        )

        num_basis = permutation_indices.shape[0]
        self.norm = nn.RMSNorm(self.virtual_dim, eps=eps)

        # Kept as a raw parameter so GPT._init_weights does not overwrite the
        # required zero initialization.
        self.dynamic_logits_fn = nn.Parameter(torch.zeros(self.virtual_dim, num_basis))
        self.dynamic_scale = nn.Parameter(torch.tensor(dynamic_scale_init))

        static_logits = torch.full((num_basis,), non_identity_logit)
        static_logits[identity_index] = 0.0
        self.static_logits = nn.Parameter(static_logits)

    @property
    def num_permutations(self) -> int:
        return self.permutation_indices.shape[0]

    def coefficients(self, slots: torch.Tensor) -> torch.Tensor:
        """Return token-wise convex-mixture coefficients [..., K]."""
        if slots.shape[-2:] != (self.n_slots, self.block_dim):
            raise ValueError(
                "expected slots ending in "
                f"({self.n_slots}, {self.block_dim}), got {slots.shape[-2:]}"
            )

        flat = slots.flatten(start_dim=-2)
        normed = self.norm(flat)
        dynamic_logits = project_slots(
            normed,
            self.dynamic_logits_fn.to(dtype=normed.dtype),
        )
        logits = self.static_logits.to(dtype=normed.dtype) + (
            self.dynamic_scale.to(dtype=normed.dtype) * dynamic_logits
        )
        return F.softmax(logits.float(), dim=-1).to(dtype=slots.dtype)

    def mixing_matrix(self, slots: torch.Tensor) -> torch.Tensor:
        """Materialize the exactly doubly stochastic matrix for diagnostics."""
        coefficients = self.coefficients(slots)
        basis = self.permutation_basis.to(dtype=slots.dtype)
        return (coefficients[..., None, None] * basis).sum(-3)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Mix slots as ``R^T @ slots`` via the dense [n, n] mixing matrix.

        R is a convex combination of K small permutation matrices; forming it
        per token ([..., n, n]) and applying with a broadcast sum is far less
        memory traffic than gathering K permuted copies of the slots, and it
        fuses under torch.compile.
        """
        return apply_map(slots, self.mixing_matrix(slots))


class GeneralizedHyperConnection(nn.Module):
    """Dynamic GHC with either unrestricted or mHC-Lite carry routing.

    The external state is ``[..., virtual_dim]`` with ``n`` slots, each of
    width ``dim / m``. The rectangular VWN read and write maps remain ordinary
    dynamic GHC maps. Only the square n x n persistent carry map is optionally
    constrained with mHC Lite.
    """

    def __init__(
        self,
        dim: int,
        m: int,
        n: int,
        eps: float = 1e-6,
        carry_mode: Literal["ghc", "mhc_lite"] = "mhc_lite",
        mhc_lite_num_permutations: int | None = None,
        mhc_lite_permutation_seed: int = 0,
    ):
        super().__init__()

        if m < 1:
            raise ValueError(f"m must be positive, got {m}")
        if n < m:
            raise ValueError(f"n must be >= m, got m={m}, n={n}")
        if dim % m != 0:
            raise ValueError(f"dim={dim} must be divisible by m={m}")
        if carry_mode not in {"ghc", "mhc_lite"}:
            raise ValueError(
                f"carry_mode must be 'ghc' or 'mhc_lite', got {carry_mode!r}"
            )

        self.dim = dim
        self.m = m
        self.n = n
        self.block_dim = dim // m
        self.virtual_dim = n * self.block_dim
        self.factor = self.block_dim**-0.5
        self.carry_mode = carry_mode

        # Rectangular read map [n, m]. Initially, the first m slots are the
        # m blocks consumed by the D-wide Transformer branch.
        static_read = torch.zeros(n, m)
        static_read[:m, :m] = torch.eye(m)
        self.static_read = nn.Parameter(static_read)
        self.dynamic_read_fn = nn.Parameter(torch.zeros(self.block_dim, m))
        self.dynamic_read_scale = nn.Parameter(torch.ones(n, m))

        # Rectangular write map beta [n, m]. At initialization, virtual slot j
        # receives branch output block j mod m.
        static_beta = torch.zeros(n, m)
        slot_ids = torch.arange(n)
        static_beta[slot_ids, slot_ids.remainder(m)] = 1.0
        self.static_beta = nn.Parameter(static_beta)
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(self.block_dim, m))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(n, m))

        self.routing_norm = nn.RMSNorm(self.block_dim, eps=eps)

        if carry_mode == "mhc_lite":
            self.carry_router = MHCLiteCarry(
                n_slots=n,
                block_dim=self.block_dim,
                num_permutations=mhc_lite_num_permutations,
                permutation_seed=mhc_lite_permutation_seed,
                eps=eps,
            )
        else:
            self.carry_router = None
            self.static_carry = nn.Parameter(torch.eye(n))
            self.dynamic_carry_fn = nn.Parameter(torch.zeros(self.block_dim, n))
            self.dynamic_carry_scale = nn.Parameter(torch.ones(n, n))

    def width_connection(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read a D-wide branch input and construct the persistent carry."""
        if h.shape[-1] != self.virtual_dim:
            raise ValueError(
                f"expected hidden width {self.virtual_dim}, got {h.shape[-1]}"
            )

        prefix = h.shape[:-1]
        slots = h.reshape(prefix + (self.n, self.block_dim))
        norm_slots = self.routing_norm(slots)

        dynamic_read = safe_tanh(
            project_slots(
                norm_slots,
                self.dynamic_read_fn.to(dtype=norm_slots.dtype),
            )
            * self.factor
        )
        read = self.static_read.to(dtype=slots.dtype) + (
            dynamic_read * self.dynamic_read_scale.to(dtype=slots.dtype)
        )

        # (slots^T @ read)^T -> [..., m, block_dim]
        branch_blocks = apply_map(slots, read)
        branch_input = branch_blocks.reshape(prefix + (self.dim,))

        if self.carry_mode == "mhc_lite":
            assert self.carry_router is not None
            carry = self.carry_router(slots)
        else:
            dynamic_carry = safe_tanh(
                project_slots(
                    norm_slots,
                    self.dynamic_carry_fn.to(dtype=norm_slots.dtype),
                )
                * self.factor
            )
            carry_matrix = self.static_carry.to(dtype=slots.dtype) + (
                dynamic_carry * self.dynamic_carry_scale.to(dtype=slots.dtype)
            )
            carry = apply_map(slots, carry_matrix)

        dynamic_beta = safe_tanh(
            project_slots(
                norm_slots,
                self.dynamic_beta_fn.to(dtype=norm_slots.dtype),
            )
            * self.factor
        )
        beta = self.static_beta.to(dtype=slots.dtype) + (
            dynamic_beta * self.dynamic_beta_scale.to(dtype=slots.dtype)
        )

        return branch_input, carry, beta

    def depth_connection(
        self,
        carry: torch.Tensor,
        branch_output: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Split a D-wide branch output and write it to the virtual state."""
        if branch_output.shape[-1] != self.dim:
            raise ValueError(
                f"expected branch width {self.dim}, got {branch_output.shape[-1]}"
            )

        prefix = branch_output.shape[:-1]
        output_blocks = branch_output.reshape(prefix + (self.m, self.block_dim))
        # beta @ output_blocks -> [..., n, block_dim], as a fusable broadcast sum.
        written = (
            beta.to(dtype=output_blocks.dtype).unsqueeze(-1)
            * output_blocks.unsqueeze(-3)
        ).sum(-2)
        slots = carry + written
        return slots.reshape(prefix + (self.virtual_dim,)).contiguous()


class LastDimGroupNorm(nn.Module):
    """GroupNorm over the final dimension of a [B, L, D] tensor."""

    def __init__(self, dim: int, group_size: int, eps: float = 1e-5):
        super().__init__()
        if dim % group_size != 0:
            raise ValueError(f"dim={dim} must be divisible by group_size={group_size}")
        self.dim = dim
        self.norm = nn.GroupNorm(
            num_groups=dim // group_size,
            num_channels=dim,
            eps=eps,
            affine=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim, 1)
        x = self.norm(x)
        return x.reshape(shape)


class TransformerBlock(nn.Module):
    """A Transformer block operating on a persistent virtual-width state."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_mult: int,
        vwn_m: int,
        vwn_n: int,
        vwn_carry_mode: Literal["ghc", "mhc_lite"],
        mhc_lite_num_permutations: int | None,
        mhc_lite_permutation_seed: int,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.attn_connection = GeneralizedHyperConnection(
            dim,
            vwn_m,
            vwn_n,
            carry_mode=vwn_carry_mode,
            mhc_lite_num_permutations=mhc_lite_num_permutations,
            mhc_lite_permutation_seed=mhc_lite_permutation_seed,
        )

        self.mlp_norm = nn.RMSNorm(dim)
        self.mlp = MLP(dim, mlp_mult)
        self.mlp_connection = GeneralizedHyperConnection(
            dim,
            vwn_m,
            vwn_n,
            carry_mode=vwn_carry_mode,
            mhc_lite_num_permutations=mhc_lite_num_permutations,
            mhc_lite_permutation_seed=mhc_lite_permutation_seed,
        )

    def forward(
        self,
        h,
        block_mask=None,
        pos_ids=None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        # Attention: virtual-width read -> D-width attention -> virtual-width write.
        x, carry, beta = self.attn_connection.width_connection(h)
        attn_out, new_kv = self.attn(
            self.attn_norm(x),
            block_mask=block_mask,
            pos_ids=pos_ids,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        h = self.attn_connection.depth_connection(carry, attn_out, beta)

        # MLP: virtual-width read -> D-width MLP -> virtual-width write.
        x, carry, beta = self.mlp_connection.width_connection(h)
        mlp_out = self.mlp(self.mlp_norm(x))
        h = self.mlp_connection.depth_connection(carry, mlp_out, beta)
        return h, new_kv


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim: int,
        n_heads: int,
        mlp_mult: int,
        n_layers: int,
        n_loops: int = 1,
        eos_id: int = 0,
        logit_softcap: float | None = None,
        vwn_m: int = 2,
        vwn_n: int = 3,
        vwn_group_norm: bool | None = None,
        vwn_carry_mode: Literal["ghc", "mhc_lite"] = "mhc_lite",
        mhc_lite_num_permutations: int | None = None,
        mhc_lite_permutation_seed: int = 0,
    ):
        super().__init__()

        if vwn_m < 1:
            raise ValueError(f"vwn_m must be positive, got {vwn_m}")
        if vwn_n < vwn_m:
            raise ValueError(f"vwn_n must be >= vwn_m, got m={vwn_m}, n={vwn_n}")
        if dim % vwn_m != 0:
            raise ValueError(f"dim={dim} must be divisible by vwn_m={vwn_m}")
        if vwn_carry_mode not in {"ghc", "mhc_lite"}:
            raise ValueError(
                f"vwn_carry_mode must be 'ghc' or 'mhc_lite', got {vwn_carry_mode!r}"
            )
        if n_loops < 1:
            raise ValueError(f"n_loops must be positive, got {n_loops}")

        self.n_loops = n_loops
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.dim = dim
        self.vwn_m = vwn_m
        self.vwn_n = vwn_n
        self.block_dim = dim // vwn_m
        self.virtual_dim = vwn_n * self.block_dim
        self.virtual_width_ratio = vwn_n / vwn_m
        self.vwn_carry_mode = vwn_carry_mode
        self.mhc_lite_num_permutations = mhc_lite_num_permutations
        self.mhc_lite_permutation_seed = mhc_lite_permutation_seed

        # Gemma-2-style final-logit soft-capping.
        self.logit_softcap = logit_softcap

        # Factorized over-width embedding: a D-wide table remains tied to the
        # unembedding, while a learned projection creates the D'-wide VWN state.
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.input_expand = (
            nn.Identity()
            if self.virtual_dim == dim
            else nn.Linear(dim, self.virtual_dim, bias=False)
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    mlp_mult=mlp_mult,
                    vwn_m=vwn_m,
                    vwn_n=vwn_n,
                    vwn_carry_mode=vwn_carry_mode,
                    mhc_lite_num_permutations=mhc_lite_num_permutations,
                    mhc_lite_permutation_seed=mhc_lite_permutation_seed,
                )
                for _ in range(n_layers)
            ]
        )

        # The paper omits pre-reduce GroupNorm for its fractional 1.5x setup.
        # In auto mode, use it only for integer expansions >1, with group size D.
        if vwn_group_norm is None:
            vwn_group_norm = self.virtual_dim > dim and vwn_n % vwn_m == 0
        if vwn_group_norm:
            if self.virtual_dim % dim != 0:
                raise ValueError(
                    "vwn_group_norm=True requires an integer virtual-width ratio"
                )
            self.pre_reduce_norm = LastDimGroupNorm(self.virtual_dim, group_size=dim)
        else:
            self.pre_reduce_norm = nn.Identity()

        self.output_reduce = (
            nn.Identity()
            if self.virtual_dim == dim
            else nn.Linear(self.virtual_dim, dim, bias=False)
        )
        self.ln_f = nn.RMSNorm(dim)

        # Loop input injection: state and expanded input embedding concatenated
        # through a learned adapter before every loop after the first.
        self.loop_adapter = (
            nn.Linear(2 * self.virtual_dim, self.virtual_dim, bias=False)
            if n_loops > 1
            else None
        )

        self.apply(self._init_weights())

        if self.loop_adapter is not None:
            # [I | small]: start as an identity carry of the running state with a
            # near-silent input-injection path (overrides the generic init above).
            with torch.no_grad():
                self.loop_adapter.weight[:, : self.virtual_dim] = torch.eye(
                    self.virtual_dim
                )
                self.loop_adapter.weight[:, self.virtual_dim :].mul_(0.1)

    def _init_weights(self):
        def init_fn(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        return init_fn

    def no_weight_decay(self) -> set[str]:
        """Parameter names the VWN paper says to exclude from weight decay.

        Merge this set into any existing no-weight-decay policy used by the
        trainer. Dynamic routing matrices and scales should still receive decay.
        """
        static_routing_suffixes = (
            "static_read",
            "static_carry",
            "static_beta",
            "static_logits",
        )
        return {
            name
            for name, _ in self.named_parameters()
            if name.endswith(static_routing_suffixes)
        }

    def forward(
        self,
        x,
        return_hidden: bool = False,
        block_mask=None,
        pos_ids=None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        # block_mask + pos_ids enable packed-document FlexAttention with
        # per-document RoPE positions. Leave both None for eval/sampling.
        h = self.input_expand(self.token_emb(x))
        e = h  # expanded input embedding, re-injected at each loop boundary

        # Weight-shared loop over the unique blocks. KV caches are per
        # (loop, block) application: n_loops * n_layers entries.
        new_past_kv = [] if use_cache else None
        for loop_idx in range(self.n_loops):
            if loop_idx > 0:
                h = self.loop_adapter(torch.cat([h, e], dim=-1))
            for i, block in enumerate(self.blocks):
                kv_idx = loop_idx * len(self.blocks) + i
                block_past_kv = past_kv[kv_idx] if past_kv is not None else None
                h, new_kv = block(
                    h,
                    block_mask=block_mask,
                    pos_ids=pos_ids,
                    past_kv=block_past_kv,
                    use_cache=use_cache,
                )
                if use_cache:
                    new_past_kv.append(new_kv)

        # VWN state D' -> optional GroupNorm -> learned reduce -> final RMSNorm.
        h = self.pre_reduce_norm(h)
        h = self.output_reduce(h)
        h = self.ln_f(h)

        if return_hidden:
            return (h, new_past_kv) if use_cache else h

        # Tied to the base D-wide input embedding table.
        logits = F.linear(h, self.token_emb.weight)
        if self.logit_softcap is not None:
            cap = self.logit_softcap
            if logits.requires_grad:
                logits = cap * torch.tanh(logits / cap)
            else:
                # Eval logits are huge; avoid doubling memory with out-of-place tanh.
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
                logits, past_kv = self(model_input, past_kv=past_kv, use_cache=True)
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
                    cumulative_probabilities = torch.cumsum(
                        sorted_probabilities, dim=-1
                    )
                    remove_mask = cumulative_probabilities > top_p
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(
                        remove_mask, float("-inf")
                    )
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
