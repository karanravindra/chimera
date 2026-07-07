"""
Muon optimizer (MomentUm Orthogonalized by Newton-schulz), with an auxiliary
AdamW for the parameters Muon should not touch.

Muon (Keller Jordan et al., https://kellerjordan.github.io/posts/muon/) updates
2D hidden weight matrices by orthogonalizing the momentum buffer via a few
Newton-Schulz iterations before applying it. It is not meant for <2D parameters
(biases, norm gains) or for the token embedding and output head, which are
conventionally optimized with AdamW instead.

:class:`Muon` is the single-device combined optimizer: it takes standard
``param_groups`` where each group carries a ``use_muon`` flag and dispatches to
the Muon or AdamW update accordingly. This keeps everything under one optimizer
(and therefore one LR scheduler), matching how the rest of ``chimera`` wires
training. Use :func:`muon_param_groups` to build the groups from a model with the
usual "matrices → Muon, everything else → AdamW" split.

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


def muon_update(
    grad: Tensor, momentum: Tensor, beta: float, ns_steps: int, nesterov: bool
) -> Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # flatten conv filters to a matrix
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # scale so the RMS of the update matches AdamW's (~1), independent of shape
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    bias_corr1 = 1 - betas[0] ** step
    bias_corr2 = 1 - betas[1] ** step
    return (exp_avg / bias_corr1) / ((exp_avg_sq / bias_corr2).sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Single-device Muon with an auxiliary AdamW.

    Expects ``param_groups`` (dicts) each carrying ``use_muon: bool``. Muon groups
    accept ``lr``, ``momentum``, ``weight_decay``, ``ns_steps``, ``nesterov``;
    AdamW groups accept ``lr``, ``betas``, ``eps``, ``weight_decay``. Missing keys
    are filled with sensible defaults.
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

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update.reshape(p.shape), alpha=-lr)
            else:
                betas, eps = group["betas"], group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        betas,
                        eps,
                    )
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
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
