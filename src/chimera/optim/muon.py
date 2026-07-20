"""
Muon optimizer (MomentUm Orthogonalized by Newton-schulz), with an auxiliary
AdamW for the parameters Muon should not touch.

:class:`Muon` is a single-object optimizer that internally runs a
shape-bucketed batched Newton-Schulz Muon for the hidden weight matrices and
delegates to ``torch.optim.AdamW`` for everything else (token embedding,
output head, norm gains, router/gate weights).

2D params sharing a (transpose-normalized) shape are *stacked and stepped as
one batch*: one momentum ``lerp_`` and one batched NS chain per bucket instead
of a sequential per-parameter loop. Profiling the 384-wide GPT train step
showed the previous per-param path (``torch.optim.Muon``) launching ~400 tiny
cutlass GEMMs with launch gaps costing a third of their compute -- the only
launch-bound region of the whole step. Bucketing collapses a 6-layer model's
24 matrices into 3 batched NS chains. Params whose slices differ only by
orientation (e.g. fc1 ``(1536, 384)`` and fc2 ``(384, 1536)``) share a bucket
via ``.mT``; the per-matrix ``sqrt(rows/cols)`` scale is applied per slice
from each param's original orientation, so the math is identical to the
per-param loop. ndim>=3 params (e.g. DeepSeek MoE's packed per-expert
weights) are already batched and form their own single-param buckets.

Everything is wrapped under one ``.step()``/``.zero_grad()``/``.param_groups``
so training loops still just do ``Muon(muon_param_groups(model, ...))``.

Usage:
    from chimera.optim import Muon, muon_param_groups

    groups = muon_param_groups(model, muon_lr=0.02, adamw_lr=3e-4)
    optimizer = Muon(groups)
"""

from typing import Iterable

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Orthogonalize ``G`` via a quintic Newton-Schulz iteration.

    Returns a matrix with roughly the same singular vectors as ``G`` but with all
    singular values pushed toward 1. Runs in bfloat16; the quintic coefficients
    are tuned so the iteration converges from any starting spectral norm <= 1.

    ``G`` may be batched (``ndim >= 3``, e.g. DeepSeek MoE's packed per-expert
    weights (n_experts, out, in)) — matmuls and ``.mT`` broadcast over leading
    dims, orthogonalizing each batch slice's last two dims independently.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # normalize so the largest singular value is <= 1 before iterating
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Single-object optimizer: shape-bucketed batched Newton-Schulz Muon +
    torch.optim.AdamW (everything else), keyed off ``use_muon``.

    Expects ``param_groups`` (dicts) each carrying ``use_muon: bool`` — exactly
    :func:`muon_param_groups`'s output. Muon groups accept ``lr``, ``momentum``,
    ``weight_decay``, ``ns_steps``, ``nesterov``; AdamW groups accept ``lr``,
    ``betas``, ``eps``, ``weight_decay``. Missing keys are filled with sensible
    defaults. A ``use_muon=True`` group's 2D params are bucketed by
    transpose-normalized shape and each bucket is stepped as one stacked batch
    (see module docstring); ndim>=3 params each form a single-param bucket.
    """

    def __init__(self, param_groups: Iterable[dict]):
        param_groups = [dict(g) for g in param_groups]
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("each param group must set 'use_muon'")
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, dict())

        # Build one delegate per input group. (group_index, kind, delegate)
        # where kind is "muon" (a list of shape buckets, stepped manually
        # below) or "adamw" (a torch.optim.AdamW instance).
        self._delegates = []
        for i, group in enumerate(self.param_groups):
            if group["use_muon"]:
                # Bucket 2D params by transpose-normalized shape (wide
                # orientation, rows <= cols) so e.g. fc1 (1536, 384) and fc2
                # (384, 1536) stack into one batch. ndim>=3 params (already
                # batched, e.g. MoE experts) get a single-param bucket each.
                buckets: dict = {}
                for p in group["params"]:
                    if p.ndim == 2:
                        trans = p.size(0) > p.size(1)
                        key = (p.size(1), p.size(0)) if trans else tuple(p.shape)
                    else:
                        trans, key = False, (id(p),)
                    buckets.setdefault(key, {"params": [], "trans": []})
                    buckets[key]["params"].append(p)
                    buckets[key]["trans"].append(trans)
                for b in buckets.values():
                    # Per-slice sqrt(rows/cols) scale from each param's
                    # ORIGINAL orientation (transposing must not change it).
                    # momentum_buf stays None until first step(): allocating
                    # at construction time would fix its dtype before
                    # Lightning's precision plugin casts the model (e.g.
                    # bf16-true), mismatching p.grad on the first real step.
                    b["scale"] = [
                        max(1, p.size(-2) / p.size(-1)) ** 0.5 for p in b["params"]
                    ]
                    b["scale_t"] = None  # cached device tensor, built lazily
                    b["momentum_buf"] = None
                self._delegates.append((i, "muon", list(buckets.values())))
            else:
                opt = torch.optim.AdamW(
                    group["params"],
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                )
                self._delegates.append((i, "adamw", opt))

    @torch.no_grad()
    def add_adamw_param(self, new_p, copy_state_from=None):
        """Add ``new_p`` to the (existing) AdamW group after construction.

        Used for modded-nanogpt dynamic untie: at the split step the tied embedding
        is forked into a separate head param, which must join AdamW mid-training.
        Optionally deep-copies the Adam moment state (``exp_avg``/``exp_avg_sq``/
        ``step``) from ``copy_state_from`` (the embedding) so the fresh head
        continues with warmed-up moments instead of a cold restart.
        """
        for i, kind, delegate in self._delegates:
            if kind != "adamw":
                continue
            self.param_groups[i]["params"].append(new_p)
            delegate.param_groups[0]["params"].append(new_p)
            if copy_state_from is not None and copy_state_from in delegate.state:
                delegate.state[new_p] = {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in delegate.state[copy_state_from].items()
                }
            return
        raise RuntimeError("no AdamW param group to add to (all groups use Muon)")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, kind, delegate in self._delegates:
            group = self.param_groups[i]
            if kind == "adamw":
                # Sync hyperparams every step: an LR scheduler mutates
                # self.param_groups[i]["lr"], which must propagate into the
                # delegate optimizer's own param group before it steps.
                sub = delegate.param_groups[0]
                sub["lr"] = group["lr"]
                sub["weight_decay"] = group["weight_decay"]
                sub["betas"] = group["betas"]
                sub["eps"] = group["eps"]
                delegate.step()
                continue

            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for b in delegate:
                params, trans = b["params"], b["trans"]
                if any(p.grad is None for p in params):
                    # all-or-nothing per bucket keeps the stacked momentum
                    # buffer aligned; partial-grad steps don't occur in
                    # practice (every hidden matrix gets a grad every step).
                    continue
                if len(params) == 1:
                    g = params[0].grad
                    grads = (g.mT if trans[0] else g).unsqueeze(0)
                else:
                    grads = torch.stack(
                        [p.grad.mT if t else p.grad for p, t in zip(params, trans)]
                    )
                buf = b["momentum_buf"]
                if buf is None:
                    buf = b["momentum_buf"] = torch.zeros_like(grads)
                buf.lerp_(grads, 1 - momentum)
                update = grads.lerp(buf, momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                if any(s != 1.0 for s in b["scale"]):
                    if b["scale_t"] is None:
                        b["scale_t"] = torch.tensor(
                            b["scale"], device=update.device, dtype=update.dtype
                        ).view(-1, *([1] * (update.ndim - 1)))
                    update = update * b["scale_t"]
                for j, (p, t) in enumerate(zip(params, trans)):
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    u = update[j].mT if t else update[j]
                    p.add_(u.reshape(p.shape).to(p.dtype), alpha=-lr)
        return loss


def muon_param_groups(
    model,
    *,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    muon_weight_decay: float = 0.0,
    adamw_weight_decay: float = 0.0,
    momentum: float = 0.95,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-10,
    adamw_name_keywords: tuple = ("emb", "head"),
) -> list[dict]:
    """Split a model's parameters into a Muon group and an AdamW group.

    Hidden weight *matrices* (``ndim >= 2``) go to Muon; everything else — the
    token embedding and output head (matched by name via ``adamw_name_keywords``)
    plus all 1D parameters (biases, norm gains) — goes to AdamW. Empty groups are
    dropped so the optimizer never sees an empty param list.

    Note: the Muon group can contain a mix of ndim==2 and ndim>=3 (batched,
    e.g. MoE expert weights) parameters — :class:`Muon` splits them internally.
    """
    muon_params, adamw_params = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embed_or_head = any(k in name for k in adamw_name_keywords)
        if p.ndim >= 2 and not is_embed_or_head:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    groups = []
    if muon_params:
        groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                momentum=momentum,
                weight_decay=muon_weight_decay,
            )
        )
    if adamw_params:
        groups.append(
            dict(
                params=adamw_params,
                use_muon=False,
                lr=adamw_lr,
                betas=betas,
                eps=eps,
                weight_decay=adamw_weight_decay,
            )
        )
    return groups
