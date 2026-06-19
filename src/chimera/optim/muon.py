"""Muon optimizer (momentum + Newton-Schulz orthogonalization).

Muon updates 2D hidden weight matrices by orthogonalizing the (momentum-smoothed)
gradient with a quintic Newton-Schulz iteration. Use it only for the transformer
matmul weights and pair it with AdamW for everything else (norms, biases,
embeddings, conv/projection layers).

Two ways to pair them:

* :class:`MuonWithAuxAdam` — a single ``torch.optim.Optimizer`` whose param groups
  are tagged ``use_muon=True/False``. Drops into frameworks that expect one
  optimizer (e.g. Lightning automatic optimization). Build its groups with
  :func:`muon_adam_param_groups`.
* :class:`chimera.optim.hybrid.HybridOptim` — wraps a standalone :class:`Muon` and a
  separate ``AdamW`` when you want to drive two real optimizers yourself.
"""

import math

import torch


@torch.no_grad()
def _newtonschulz5(G, steps=5, eps=1e-7):
    # quintic Newton-Schulz orthogonalization of the momentum matrix
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    # fp32 NS. bf16 NS (upstream Muon practice) was tried for throughput but
    # under the now-default non-deterministic mode TF32 already makes fp32 NS
    # cheap (~0.2% step-time difference), so precision is kept. See
    # projects/llm/OPTIMIZATION_LOG.md for the quality comparison.
    X = G.to(torch.float32)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


@torch.no_grad()
def _muon_update(p, state, *, lr, momentum, nesterov, ns_steps, weight_decay):
    """One in-place Muon step for a single 2D parameter ``p`` (grad assumed set)."""
    g = p.grad
    if "m" not in state:
        state["m"] = torch.zeros_like(g)
    buf = state["m"]
    buf.lerp_(g, 1 - momentum)
    g = g.lerp(buf, momentum) if nesterov else buf
    u = _newtonschulz5(g.reshape(g.size(0), -1), steps=ns_steps).view_as(g)
    u = u * max(1.0, g.size(-2) / g.size(-1)) ** 0.5  # aspect-ratio scale
    p.mul_(1 - lr * weight_decay)  # decoupled WD
    p.add_(u, alpha=-lr)


@torch.no_grad()
def _muon_update_batched(
    params, states, *, lr, momentum, nesterov, ns_steps, weight_decay
):
    """Muon step for a list of SAME-SHAPED 2D params, orthogonalized as one batch.

    Identical math to :func:`_muon_update` per slice, but the momentum updates
    run as foreach ops and the Newton-Schulz iteration runs once on a stacked
    ``[k, m, n]`` tensor (batched matmuls) instead of k sequential tiny matmuls
    — the per-param loop left the GPU idle between launches (profiled: 4-8 ms
    bubbles around the optimizer every step for llm_xs's 24 small matrices).
    """
    grads = [p.grad for p in params]
    for g, s in zip(grads, states):
        if "m" not in s:
            s["m"] = torch.zeros_like(g)
    bufs = [s["m"] for s in states]
    torch._foreach_lerp_(bufs, grads, 1 - momentum)
    gs = torch._foreach_lerp(grads, bufs, momentum) if nesterov else bufs
    G = torch.stack([g.reshape(g.size(0), -1) for g in gs])
    U = _newtonschulz5(G, steps=ns_steps)
    scale = max(1.0, G.size(-2) / G.size(-1)) ** 0.5
    torch._foreach_mul_(list(params), 1 - lr * weight_decay)  # decoupled WD
    for p, u in zip(params, U.unbind(0)):
        p.add_(u.view_as(p), alpha=-lr * scale)


@torch.no_grad()
def _adamw_update(p, state, *, lr, betas, eps, weight_decay):
    """One in-place decoupled-AdamW step for a single parameter ``p``."""
    g = p.grad
    if "step" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
    state["step"] += 1
    beta1, beta2 = betas
    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    exp_avg.lerp_(g, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    bias1 = 1 - beta1 ** state["step"]
    bias2 = 1 - beta2 ** state["step"]
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
    p.mul_(1 - lr * weight_decay)  # decoupled WD
    p.addcdiv_(exp_avg, denom, value=-lr / bias1)


@torch.no_grad()
def _adamw_update_foreach(params, states, *, lr, betas, eps, weight_decay):
    """Decoupled AdamW for a list of params via foreach ops (same math as
    :func:`_adamw_update`, ~8 kernel launches total instead of ~8 per param)."""
    grads = [p.grad for p in params]
    for p, s in zip(params, states):
        if "step" not in s:
            s["step"] = 0
            s["exp_avg"] = torch.zeros_like(p)
            s["exp_avg_sq"] = torch.zeros_like(p)
        s["step"] += 1
    beta1, beta2 = betas
    exp_avg = [s["exp_avg"] for s in states]
    exp_avg_sq = [s["exp_avg_sq"] for s in states]
    torch._foreach_lerp_(exp_avg, grads, 1 - beta1)
    torch._foreach_mul_(exp_avg_sq, beta2)
    torch._foreach_addcmul_(exp_avg_sq, grads, grads, value=1 - beta2)
    denom = torch._foreach_sqrt(exp_avg_sq)
    torch._foreach_div_(denom, [math.sqrt(1 - beta2 ** s["step"]) for s in states])
    torch._foreach_add_(denom, eps)
    torch._foreach_mul_(list(params), 1 - lr * weight_decay)  # decoupled WD
    torch._foreach_addcdiv_(
        list(params),
        exp_avg,
        denom,
        [-lr / (1 - beta1 ** s["step"]) for s in states],
    )


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden weights only. Pair with AdamW for everything else."""

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
            ),
        )

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                _muon_update(
                    p,
                    self.state[p],
                    lr=grp["lr"],
                    momentum=grp["momentum"],
                    nesterov=grp["nesterov"],
                    ns_steps=grp["ns_steps"],
                    weight_decay=grp["weight_decay"],
                )


def muon_adam_param_groups(
    model,
    *,
    muon_lr=0.02,
    adam_lr=8e-4,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.01,
):
    """Split ``model``'s parameters into Muon and AdamW groups for
    :class:`MuonWithAuxAdam`.

    Muon takes the 2D hidden matmul weights; AdamW takes everything else — the
    token/positional embeddings (matched by ``"embedding"`` in the name, so the
    tied output projection rides along), LayerNorm scale/bias, and all biases.
    Every ``requires_grad`` parameter lands in exactly one group.
    """
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embedding" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    groups = []
    if muon_params:
        groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
            )
        )
    if adam_params:
        groups.append(
            dict(
                params=adam_params,
                use_muon=False,
                lr=adam_lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        )
    return groups


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon + AdamW behind one ``optimizer.step()``.

    Each param group is tagged ``use_muon``: ``True`` groups get the Muon update
    (2D hidden weights), the rest get decoupled AdamW (embeddings, norms, biases).
    Presenting a single ``torch.optim.Optimizer`` lets it drop into Lightning's
    automatic optimization — gradient clipping and LR logging keep working — where
    a multi-optimizer setup would otherwise force manual optimization. Build the
    groups with :func:`muon_adam_param_groups`.
    """

    def __init__(self, param_groups):
        param_groups = list(param_groups)
        for g in param_groups:
            g["use_muon"] = bool(g.get("use_muon"))
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("nesterov", True)
                g.setdefault("ns_steps", 5)
                g.setdefault("weight_decay", 0.01)
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.01)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for grp in self.param_groups:
            params = [p for p in grp["params"] if p.grad is not None]
            if not params:
                continue
            if grp["use_muon"]:
                # Batch same-shaped params (e.g. the 6 layers' qkv projections)
                # through one stacked Newton-Schulz instead of sequential tiny
                # matmuls; llm_xs goes from 24 NS calls/step to 4.
                by_shape: dict[tuple, list] = {}
                for p in params:
                    by_shape.setdefault(tuple(p.shape), []).append(p)
                for shaped in by_shape.values():
                    _muon_update_batched(
                        shaped,
                        [self.state[p] for p in shaped],
                        lr=grp["lr"],
                        momentum=grp["momentum"],
                        nesterov=grp["nesterov"],
                        ns_steps=grp["ns_steps"],
                        weight_decay=grp["weight_decay"],
                    )
            else:
                _adamw_update_foreach(
                    params,
                    [self.state[p] for p in params],
                    lr=grp["lr"],
                    betas=grp["betas"],
                    eps=grp["eps"],
                    weight_decay=grp["weight_decay"],
                )
        return loss
