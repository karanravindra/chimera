"""GRPO algorithm core -- the task-agnostic math, kept pure so it is unit-testable in
isolation from the model, the dataset, and Lightning.

Group Relative Policy Optimization (DeepSeekMath, arXiv:2402.03300) replaces PPO's learned
value baseline with a *group* baseline: for each prompt we sample ``G`` completions, score
them, and turn each reward into an advantage by centering on the group mean (and, by
default, scaling by the group std). The policy is then nudged to raise the log-prob of the
above-average completions and lower the below-average ones.

Three pieces live here:

* :func:`compute_group_advantages` -- rewards (one per completion) -> advantages.
* :func:`selective_log_softmax` -- per-token log-prob of the realized tokens, computed
  memory-efficiently via a fused cross-entropy so we never materialize a vocab-wide fp32
  tensor (important on a 16 GB card with ~65k vocab).
* :func:`grpo_loss` -- the token-level (DAPO-normalized) policy-gradient loss with an
  optional KL-to-reference penalty.

We run a single optimizer update per generation (``num_iterations == mu == 1``), so the
PPO importance ratio ``pi_theta / pi_theta_old`` is exactly 1 in value at the point of
evaluation; we still write it as ``exp(logp - logp.detach())`` so the term *carries the
policy gradient* (its grad is ``A * d log pi``) and the formulation extends cleanly to
clipping should multiple inner steps ever be added.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    *,
    scale_rewards: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Center (and optionally std-scale) rewards within each prompt's group of completions.

    ``rewards`` is a 1-D tensor of length ``n_prompts * group_size`` laid out so each
    prompt's ``group_size`` completions are **contiguous** (the order
    ``model.generate(..., num_return_sequences=G)`` returns for a batched prompt). The
    advantage is ``A_i = (r_i - mean(group)) / (std(group) + eps)``.

    Setting ``scale_rewards=False`` keeps only the mean-centering. Dividing by the group
    std normalizes variance but, as shown in *Understanding R1-Zero-Like Training*
    (arXiv:2503.20783), introduces a question-level difficulty bias; disabling it makes
    update magnitudes track the raw reward scale instead.
    """
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} is not a multiple of group_size {group_size}"
        )
    grouped = rewards.view(-1, group_size)
    advantages = grouped - grouped.mean(dim=1, keepdim=True)
    if scale_rewards:
        advantages = advantages / (grouped.std(dim=1, keepdim=True) + eps)
    return advantages.reshape(-1)


def mgpo_difficulty_weights(
    rewards: torch.Tensor,
    group_size: int,
    *,
    lam: float = 2.0,
    p0: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-completion MGPO difficulty weights (VibeThinker, arXiv:2511.06221).

    MaxEnt-Guided Policy Optimization reweights each group's GRPO advantage by how *learnable*
    the prompt currently is: groups whose empirical pass rate ``p_c`` sits near the
    maximum-entropy point ``p0=0.5`` get weight ~1, while trivially-easy (``p_c->1``) and
    currently-impossible (``p_c->0``) groups are exponentially down-weighted:

        ``w(p_c) = exp(-lam * D_ME(p_c || p0))``,
        ``D_ME(p_c || p0) = KL(Bernoulli(p_c) || Bernoulli(p0))``.

    This bakes a smooth difficulty curriculum into the loss (a continuous version of dynamic
    sampling) so gradient mass concentrates on the frontier the model can still move. With a
    binary correctness reward, ``p_c`` is just the group mean. Returns a ``(n_prompts*G,)``
    tensor (the group weight broadcast to each of its completions); multiply it into the
    advantages before :func:`grpo_loss`. ``lam=0`` recovers plain GRPO (all weights 1).
    """
    grouped = rewards.view(-1, group_size)
    p_c = grouped.mean(dim=1, keepdim=True).clamp(eps, 1.0 - eps)
    kl = p_c * torch.log(p_c / p0) + (1.0 - p_c) * torch.log((1.0 - p_c) / (1.0 - p0))
    w = torch.exp(-lam * kl)
    return w.expand(-1, group_size).reshape(-1)


def selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Log-prob of each realized token under ``logits``: ``log softmax(logits)[target]``.

    Returns a ``(B, T)`` tensor of the log-probabilities of ``target_ids`` (also ``(B, T)``).
    Implemented as ``-cross_entropy`` rather than an explicit ``log_softmax(...).gather(...)``
    so the softmax normalization is fused: PyTorch's cross-entropy kernel accumulates the
    log-sum-exp in fp32 internally without ever allocating a ``(B, T, vocab)`` fp32 tensor,
    which would otherwise be the dominant activation on a small GPU.
    """
    batch, time, vocab = logits.shape
    neg_logp = F.cross_entropy(
        logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none"
    )
    return (-neg_logp).view(batch, time)


def grpo_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    ref_logprobs: torch.Tensor | None = None,
    beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The GRPO loss over a batch of completions, token-level (DAPO) normalized.

    Args:
        logprobs: ``(B, T)`` per-token log-probs of the completion tokens under the *current*
            policy (with grad).
        advantages: ``(B,)`` per-completion advantage from :func:`compute_group_advantages`.
        completion_mask: ``(B, T)`` 1 for real completion tokens (through the first EOS), 0
            for padding -- excludes pad tokens from both the loss and its normalizer.
        ref_logprobs: optional ``(B, T)`` per-token log-probs under the reference policy
            (the LoRA-disabled base model). Only used when ``beta > 0``.
        beta: KL-penalty coefficient. ``0`` (the modern GRPO default) drops the KL term and
            needs no reference model.

    Returns ``(loss, kl)`` where ``kl`` is the mean per-token KL estimate (or ``None`` when
    ``beta == 0``), logged for monitoring.

    The objective maximizes ``A_i * pi_theta(o_{i,t})`` summed over all completion tokens and
    divided by the total number of completion tokens (DAPO normalization, arXiv:2503.14476),
    which avoids the per-response length bias of dividing each sequence by its own length.
    """
    # mu == 1: this equals advantages in value but carries the policy gradient (d = A * d log pi).
    coef = torch.exp(logprobs - logprobs.detach())
    per_token = coef * advantages.unsqueeze(1)

    kl = None
    if beta > 0.0 and ref_logprobs is not None:
        # Schulman's k3 estimator: unbiased, non-negative, low variance.
        diff = ref_logprobs - logprobs
        kl_per_token = torch.exp(diff) - diff - 1.0
        per_token = per_token - beta * kl_per_token
        kl = _masked_mean(kl_per_token, completion_mask)

    loss = -_masked_mean(per_token, completion_mask)
    return loss, kl


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the nonzero entries of ``mask`` (token-level normalization)."""
    return (values * mask).sum() / mask.sum().clamp(min=1.0)
