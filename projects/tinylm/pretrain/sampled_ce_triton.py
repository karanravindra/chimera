"""Fused Triton sampled-softmax CE for the tied lm_head.

Same objective as the eager version (Jean et al. 2015 style sampled softmax,
shared uniform negatives, per-token collision masking, optional logit
softcap), but the forward never materializes the [T, k] logits: a
flash-attention-style kernel streams K-tiles of the sampled head rows through
an online logsumexp, fusing

    pos gather-dot + neg GEMM + softcap + collision mask + LSE + NLL

into a single pass. Backward recomputes the logit tiles from the saved
per-token LSE (also flash-style) and emits the scaled softmax grads G once;
the two dense grad GEMMs (dH = G @ W_neg, dW_neg = G^T @ H) then run in
cuBLAS, and the row scatters go through deterministic-shape index_add_.

vs. the eager / torch.compile reference this removes:
  * the [T, D] gather of weight[targets]      (pos logit is a fused gather-dot)
  * the [T, k] neg-logit write + read          (fwd epilogue is fused)
  * the fp32 [T, k+1] cat/copy inside F.cross_entropy
  * [T, k] activation memory held across the graph -- backward keeps only
    lse/pos ([T] fp32 each) + the [k, D] gathered head rows, and pays one
    logit recompute GEMM instead (same trade as flash attention).

RNG stays outside the autograd.Function, same contract as the eager version
(randint with a data-dependent size graph-breaks under torch.compile, and
keeping it out makes seeding/parity testing trivial).

Numerics: tl.dot accumulates in fp32 regardless of input dtype; softcap,
masking, LSE and the loss are computed in fp32. For fp32 inputs tl.dot uses
TF32 on Ampere+ (same as cuBLAS with allow_tf32=True).

Bias caveat unchanged from the eager version: the k+1-candidate log-partition
lower-bounds the true logsumexp over V, so this is a train-time objective
only; eval/report must use the full softmax.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["sampled_cross_entropy", "sampled_ce_from_negatives"]


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@triton.jit
def _tanh(x):
    # Portable tanh (no libdevice dependency): saturates correctly since
    # exp(2x) -> inf gives 1.0 and exp(2x) -> 0 gives -1.0 under IEEE fp32.
    return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)


def _configs():
    return [
        triton.Config({"BLOCK_T": bt, "BLOCK_K": bk, "BLOCK_D": bd},
                      num_warps=w, num_stages=s)
        for bt, bk, bd, w, s in [
            (64, 64, 64, 4, 3),
            (64, 128, 64, 4, 3),
            (64, 64, 128, 4, 3),
            (128, 64, 64, 8, 3),
            (128, 128, 32, 8, 2),
            (32, 128, 128, 4, 4),
        ]
    ]


@triton.autotune(configs=_configs(), key=["K", "D"])
@triton.jit
def _fwd_kernel(
    H, W, WNEG, TGT, NEG,          # inputs
    LOSS, LSE, POS,                # outputs: per-token nll, logsumexp, capped pos logit
    T, K, D,
    s_ht, s_hd, s_wv, s_wd, s_nk, s_nd,
    softcap,
    HAS_CAP: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < T
    tgt = tl.load(TGT + t, mask=tmask, other=0)

    # ---- positive logit: fused gather-dot h[t] . W[tgt[t]] ----------------
    pos = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m2 = tmask[:, None] & (d < D)[None, :]
        h = tl.load(H + t[:, None] * s_ht + d[None, :] * s_hd, mask=m2, other=0.0)
        w = tl.load(W + tgt[:, None] * s_wv + d[None, :] * s_wd, mask=m2, other=0.0)
        pos += tl.sum(h.to(tl.float32) * w.to(tl.float32), 1)
    if HAS_CAP:
        pos = softcap * _tanh(pos / softcap)

    # ---- online logsumexp over sampled negatives, seeded with the pos ----
    m_i = pos                                            # running max
    l_i = tl.full((BLOCK_T,), 1.0, dtype=tl.float32)     # exp(pos - m_i) = 1

    for k0 in range(0, K, BLOCK_K):
        koff = k0 + tl.arange(0, BLOCK_K)
        kmask = koff < K
        acc = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            dmask = d < D
            h = tl.load(H + t[:, None] * s_ht + d[None, :] * s_hd,
                        mask=tmask[:, None] & dmask[None, :], other=0.0)
            wn = tl.load(WNEG + koff[:, None] * s_nk + d[None, :] * s_nd,
                         mask=kmask[:, None] & dmask[None, :], other=0.0)
            acc = tl.dot(h, tl.trans(wn), acc)
        if HAS_CAP:
            acc = softcap * _tanh(acc / softcap)
        nid = tl.load(NEG + koff, mask=kmask, other=-1)
        dead = (nid[None, :] == tgt[:, None]) | (koff[None, :] >= K)
        acc = tl.where(dead, float("-inf"), acc)

        m_new = tl.maximum(m_i, tl.max(acc, 1))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(acc - m_new[:, None]), 1)
        m_i = m_new

    lse = m_i + tl.log(l_i)
    tl.store(LOSS + t, lse - pos, mask=tmask)
    tl.store(LSE + t, lse, mask=tmask)
    tl.store(POS + t, pos, mask=tmask)


@triton.autotune(configs=_configs(), key=["K", "D"])
@triton.jit
def _bwd_kernel(
    H, WNEG, TGT, NEG, LSE, POS, DOUT,   # DOUT: 0-dim upstream grad of the mean
    G, GPOS,                              # out: dL/d(raw neg logits) [T,K], dL/d(raw pos logit) [T]
    T, K, D,
    s_ht, s_hd, s_nk, s_nd, s_gt, s_gk,
    softcap,
    HAS_CAP: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < T
    tgt = tl.load(TGT + t, mask=tmask, other=0)
    lse = tl.load(LSE + t, mask=tmask, other=0.0)
    pos = tl.load(POS + t, mask=tmask, other=0.0)
    scale = tl.load(DOUT).to(tl.float32) / T             # mean reduction

    # d loss / d raw pos logit
    gp = (tl.exp(pos - lse) - 1.0) * scale
    if HAS_CAP:
        gp = gp * (1.0 - (pos / softcap) * (pos / softcap))
    tl.store(GPOS + t, gp, mask=tmask)

    # recompute logit tiles, emit d loss / d raw neg logits
    for k0 in range(0, K, BLOCK_K):
        koff = k0 + tl.arange(0, BLOCK_K)
        kmask = koff < K
        acc = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            dmask = d < D
            h = tl.load(H + t[:, None] * s_ht + d[None, :] * s_hd,
                        mask=tmask[:, None] & dmask[None, :], other=0.0)
            wn = tl.load(WNEG + koff[:, None] * s_nk + d[None, :] * s_nd,
                         mask=kmask[:, None] & dmask[None, :], other=0.0)
            acc = tl.dot(h, tl.trans(wn), acc)
        if HAS_CAP:
            capped = softcap * _tanh(acc / softcap)
            dcap = 1.0 - (capped / softcap) * (capped / softcap)
        else:
            capped = acc
            dcap = 1.0
        p = tl.exp(capped - lse[:, None])                # softmax prob vs saved LSE
        nid = tl.load(NEG + koff, mask=kmask, other=-1)
        dead = (nid[None, :] == tgt[:, None]) | (koff[None, :] >= K)
        g = tl.where(dead, 0.0, p * dcap * scale)
        tl.store(G + t[:, None] * s_gt + koff[None, :] * s_gk,
                 g.to(G.dtype.element_ty),
                 mask=tmask[:, None] & kmask[None, :])


# --------------------------------------------------------------------------
# autograd wrapper
# --------------------------------------------------------------------------


class _SampledCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, neg_idx, softcap):
        # hidden [T, D], weight [V, D], targets [T] int64, neg_idx [K] int64
        T, D = hidden.shape
        K = neg_idx.numel()
        w_neg = weight.index_select(0, neg_idx).contiguous()   # [K, D], k*D copy

        loss = hidden.new_empty(T, dtype=torch.float32)
        lse = hidden.new_empty(T, dtype=torch.float32)
        pos = hidden.new_empty(T, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),)
        _fwd_kernel[grid](
            hidden, weight, w_neg, targets, neg_idx, loss, lse, pos,
            T, K, D,
            hidden.stride(0), hidden.stride(1),
            weight.stride(0), weight.stride(1),
            w_neg.stride(0), w_neg.stride(1),
            1.0 if softcap is None else float(softcap),
            HAS_CAP=softcap is not None,
        )
        ctx.save_for_backward(hidden, weight, targets, neg_idx, w_neg, lse, pos)
        ctx.softcap = softcap
        return loss.mean()

    @staticmethod
    def backward(ctx, dout):
        hidden, weight, targets, neg_idx, w_neg, lse, pos = ctx.saved_tensors
        T, D = hidden.shape
        K = neg_idx.numel()

        # G in the activation dtype so both grad GEMMs hit tensor cores.
        G = hidden.new_empty((T, K))
        gpos = hidden.new_empty(T, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),)
        _bwd_kernel[grid](
            hidden, w_neg, targets, neg_idx, lse, pos,
            dout.contiguous().to(torch.float32),
            G, gpos,
            T, K, D,
            hidden.stride(0), hidden.stride(1),
            w_neg.stride(0), w_neg.stride(1),
            G.stride(0), G.stride(1),
            1.0 if ctx.softcap is None else float(ctx.softcap),
            HAS_CAP=ctx.softcap is not None,
        )
        gpos_h = gpos.to(hidden.dtype)

        d_hidden = d_weight = None
        if ctx.needs_input_grad[0]:
            d_hidden = G @ w_neg                                        # [T, D]
            d_hidden.add_(
                (weight.index_select(0, targets) * gpos_h.unsqueeze(1))
                .to(d_hidden.dtype)
            )
        if ctx.needs_input_grad[1]:
            d_weight = torch.zeros_like(weight)
            d_weight.index_add_(0, neg_idx, (G.mT @ hidden).to(weight.dtype))
            d_weight.index_add_(
                0, targets, (hidden * gpos_h.unsqueeze(1)).to(weight.dtype)
            )
        return d_hidden, d_weight, None, None, None


# --------------------------------------------------------------------------
# public API (mirrors the eager version)
# --------------------------------------------------------------------------


def sampled_ce_from_negatives(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    neg_idx: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    """Deterministic core given pre-drawn negatives (RNG stays outside)."""
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    if not flat_hidden.is_contiguous():
        flat_hidden = flat_hidden.contiguous()
    flat_targets = targets.reshape(-1).contiguous()
    assert flat_hidden.is_cuda and weight.is_cuda
    assert flat_targets.dtype == torch.int64 and neg_idx.dtype == torch.int64
    assert weight.shape[1] == flat_hidden.shape[1]
    return _SampledCEFn.apply(flat_hidden, weight, flat_targets,
                              neg_idx.contiguous(), softcap)


def sampled_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int,
    softcap: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean sampled-softmax CE of [B, N, D] hidden against a [V, D] head."""
    neg_idx = torch.randint(
        weight.shape[0], (num_samples,), device=hidden.device, generator=generator
    )
    return sampled_ce_from_negatives(hidden, weight, targets, neg_idx, softcap)


# --------------------------------------------------------------------------
# parity tests + benchmark (run on a CUDA box: python sampled_ce_triton.py)
# --------------------------------------------------------------------------


def _reference(hidden, weight, targets, neg_idx, softcap=None):
    import torch.nn.functional as F
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    pos_logits = (flat_hidden * weight[flat_targets]).sum(-1)
    neg_logits = flat_hidden @ weight[neg_idx].t()
    if softcap is not None:
        pos_logits = softcap * torch.tanh(pos_logits / softcap)
        neg_logits = softcap * torch.tanh(neg_logits / softcap)
    collision = neg_idx.unsqueeze(0) == flat_targets.unsqueeze(1)
    neg_logits = neg_logits.masked_fill(collision, float("-inf"))
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
    return F.cross_entropy(logits.float(), torch.zeros_like(flat_targets))


def _test_parity():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True  # match tl.dot fp32 numerics

    # fp32 tolerance is TF32-bound: vs an fp64 oracle, eager and triton grads
    # are both ~1.5e-3 off (RTX 5070 Ti), so their mutual error can reach ~3e-3.
    for dtype, rtol in [(torch.float32, 5e-3), (torch.bfloat16, 3e-2)]:
        for softcap in (None, 20.0):
            # awkward sizes on purpose: nothing divides the block sizes
            T, D, V, K = 1021, 193, 5000, 211
            h = 0.7 * torch.randn(T, D, device="cuda", dtype=dtype)
            w = 0.7 * torch.randn(V, D, device="cuda", dtype=dtype)
            y = torch.randint(V, (T,), device="cuda")
            neg = torch.randint(V, (K,), device="cuda")
            neg[:8] = y[:8]        # force collisions
            neg[9] = neg[10]       # force a duplicate negative

            h1 = h.clone().requires_grad_(True)
            w1 = w.clone().requires_grad_(True)
            h2 = h.clone().requires_grad_(True)
            w2 = w.clone().requires_grad_(True)

            l1 = _reference(h1, w1, y, neg, softcap)
            l2 = sampled_ce_from_negatives(h2, w2, y, neg, softcap)
            (l1 * 0.37).backward()
            (l2 * 0.37).backward()

            def rel(a, b):
                return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()

            errs = (abs(l1.item() - l2.item()) / abs(l1.item()),
                    rel(h2.grad, h1.grad), rel(w2.grad, w1.grad))
            status = "ok " if max(errs) < rtol else "FAIL"
            print(f"[{status}] {str(dtype):>15} cap={softcap}  "
                  f"loss {errs[0]:.1e}  dH {errs[1]:.1e}  dW {errs[2]:.1e}")
            assert max(errs) < rtol, errs


def _bench():
    from triton.testing import do_bench

    torch.manual_seed(0)
    T, D, V, K = 16384, 2048, 131072, 4096
    dt = torch.bfloat16
    h = torch.randn(T, D, device="cuda", dtype=dt, requires_grad=True)
    w = (0.02 * torch.randn(V, D, device="cuda", dtype=dt)).requires_grad_(True)
    y = torch.randint(V, (T,), device="cuda")
    neg = torch.randint(V, (K,), device="cuda")
    cap = 30.0

    ref_core = torch.compile(_reference, dynamic=False)

    def fwd(fn):
        def run():
            fn(h, w, y, neg, cap)
        return run

    def fwd_bwd(fn):
        def run():
            h.grad = w.grad = None
            fn(h, w, y, neg, cap).backward()
        return run

    variants = [
        ("eager reference", _reference),
        ("compiled reference", ref_core),
        ("triton fused", sampled_ce_from_negatives),
    ]
    print(f"\nT={T} D={D} V={V} k={K} {dt}")
    for name, fn in variants:
        fwd(fn)(); fwd_bwd(fn)()  # warmup / compile / autotune
        t_f = do_bench(fwd(fn))
        t_fb = do_bench(fwd_bwd(fn))
        torch.cuda.reset_peak_memory_stats()
        fwd_bwd(fn)()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 2**20
        print(f"{name:>20}: fwd {t_f:7.3f} ms   fwd+bwd {t_fb:7.3f} ms   "
              f"peak {mem:7.0f} MiB")


if __name__ == "__main__":
    _test_parity()
    _bench()
