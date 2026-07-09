"""
Muon optimizer (MomentUm Orthogonalized by Newton-schulz), with an auxiliary
AdamW for the parameters Muon should not touch.

:class:`Muon` is a single-object optimizer that internally delegates to
``torch.optim.Muon`` (ordinary 2D hidden weight matrices) + ``torch.optim.AdamW``
(everything else — token embedding, output head, norm gains, router/gate
weights), plus a small hand-rolled batched Newton-Schulz path for any
Muon-eligible parameter with ``ndim >= 3`` (e.g. DeepSeek MoE's packed
per-expert weights, shape ``(n_experts, out, in)``) — ``torch.optim.Muon``
hard-rejects anything but ``ndim == 2``.

Benchmarked (see chimera memory / session notes) on this repo's tiny (48-dim)
GPT proxy width:
  - ``torch.optim.Muon`` + ``torch.optim.AdamW`` is ~1.44x faster per full
    training step than a naive per-parameter Python-loop Muon (this class's
    previous implementation) on ordinary 2D matrices.
  - For MoE's batched (ndim>=3) expert weights, splitting each expert into a
    separate 2D leaf so ``torch.optim.Muon`` can take it is ~5.2x SLOWER than
    keeping the batched tensor and doing Newton-Schulz on it directly:
    Muon's cost scales with the *number* of distinct parameter tensors, not
    their total size, and splitting an 8-expert weight multiplies the tensor
    count 8x.

Everything is wrapped under one ``.step()``/``.zero_grad()``/``.param_groups``
so :class:`chimera.modules.LanguageModelModule` and every project's
``train.py`` are unaffected — they still just do
``Muon(muon_param_groups(model, ...))``.

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
    """Single-object optimizer: torch.optim.Muon (2D) + batched Newton-Schulz
    (ndim>=3) + torch.optim.AdamW (everything else), keyed off ``use_muon``.

    Expects ``param_groups`` (dicts) each carrying ``use_muon: bool`` — exactly
    :func:`muon_param_groups`'s output. Muon groups accept ``lr``, ``momentum``,
    ``weight_decay``, ``ns_steps``, ``nesterov``; AdamW groups accept ``lr``,
    ``betas``, ``eps``, ``weight_decay``. Missing keys are filled with sensible
    defaults. A ``use_muon=True`` group's params are split internally by
    ``ndim`` — ``==2`` goes to ``torch.optim.Muon``, ``>=3`` goes to the
    batched Newton-Schulz path — transparent to the caller.
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

        # Build one delegate per input group, split further by ndim for
        # use_muon groups. (group_index, kind, delegate) where kind is
        # "muon2d" (a torch.optim.Muon instance), "muon_batched" (a plain
        # dict of params + momentum buffers, stepped manually below), or
        # "adamw" (a torch.optim.AdamW instance).
        self._delegates = []
        for i, group in enumerate(self.param_groups):
            if group["use_muon"]:
                twoD = [p for p in group["params"] if p.ndim == 2]
                batched = [p for p in group["params"] if p.ndim != 2]
                if twoD:
                    opt = torch.optim.Muon(
                        twoD,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_steps=group["ns_steps"],
                    )
                    self._delegates.append((i, "muon2d", opt))
                if batched:
                    # momentum_bufs stay None until first step(): eagerly
                    # allocating here (at construction time) would fix their
                    # dtype before Lightning's precision plugin later casts
                    # the model's params (e.g. bf16-true), causing a dtype
                    # mismatch against p.grad on the first real step.
                    state = {"params": batched, "momentum_bufs": [None] * len(batched)}
                    self._delegates.append((i, "muon_batched", state))
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
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, kind, delegate in self._delegates:
            group = self.param_groups[i]
            if kind == "muon2d":
                # Sync hyperparams every step: an LR scheduler mutates
                # self.param_groups[i]["lr"], which must propagate into the
                # delegate optimizer's own param group before it steps.
                sub = delegate.param_groups[0]
                sub["lr"] = group["lr"]
                sub["weight_decay"] = group["weight_decay"]
                sub["momentum"] = group["momentum"]
                sub["nesterov"] = group["nesterov"]
                sub["ns_steps"] = group["ns_steps"]
                delegate.step()
            elif kind == "adamw":
                sub = delegate.param_groups[0]
                sub["lr"] = group["lr"]
                sub["weight_decay"] = group["weight_decay"]
                sub["betas"] = group["betas"]
                sub["eps"] = group["eps"]
                delegate.step()
            else:  # muon_batched
                lr = group["lr"]
                wd = group["weight_decay"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                bufs = delegate["momentum_bufs"]
                for j, (p, buf) in enumerate(zip(delegate["params"], bufs)):
                    if p.grad is None:
                        continue
                    if buf is None:
                        buf = torch.zeros_like(p)
                        bufs[j] = buf
                    buf.lerp_(p.grad, 1 - momentum)
                    update = p.grad.lerp(buf, momentum) if nesterov else buf
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    update = update * (max(1, update.size(-2) / update.size(-1)) ** 0.5)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update.reshape(p.shape), alpha=-lr)
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
