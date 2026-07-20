"""Sampled-softmax cross entropy for the tied lm_head.

Approximates full-vocab CE by scoring the target against ``num_samples``
uniformly drawn negative classes instead of all ``V`` classes (Jean et al.
2015, "On Using Very Large Target Vocabulary for NMT"). The positive logit is
an elementwise dot with the target rows; negatives share one sampled index set
per call, so the matmul is [B*N, D] @ [D, k] instead of [B*N, D] @ [D, V].

The estimate is biased low as an NLL (the log-partition over k+1 candidates
lower-bounds the true logsumexp over V), so report/eval must still use the
full softmax; this is a train-time objective only.
"""

import torch
import torch.nn.functional as F


def _sampled_ce_from_negatives(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    neg_idx: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    """Deterministic core given pre-drawn negatives — the torch.compile target.

    (RNG stays outside: randint with a data-dependent size graph-breaks.)
    """
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)

    pos_logits = (flat_hidden * weight[flat_targets]).sum(-1)  # [T]
    neg_logits = flat_hidden @ weight[neg_idx].t()  # [T, k]

    if softcap is not None:
        pos_logits = softcap * torch.tanh(pos_logits / softcap)
        neg_logits = softcap * torch.tanh(neg_logits / softcap)

    # Mask per-token collisions between a sampled negative and the true target.
    collision = neg_idx.unsqueeze(0) == flat_targets.unsqueeze(1)  # [T, k]
    neg_logits = neg_logits.masked_fill(collision, float("-inf"))

    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
    return F.cross_entropy(
        logits.float(),
        torch.zeros_like(flat_targets),
    )


def sampled_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int,
    softcap: float | None = None,
    generator: torch.Generator | None = None,
    core=_sampled_ce_from_negatives,
) -> torch.Tensor:
    """Mean sampled-softmax CE of [B, N, D] hidden against a [V, D] head.

    Negatives are drawn uniformly with replacement, shared across all tokens
    in the batch. Negatives that collide with a token's own target are masked
    out for that token so the target never appears twice in its partition.
    Pass ``core=torch.compile(_sampled_ce_from_negatives, ...)`` to fuse the
    logit math.
    """
    neg_idx = torch.randint(
        weight.shape[0], (num_samples,), device=hidden.device, generator=generator
    )
    return core(hidden, weight, targets, neg_idx, softcap=softcap)
