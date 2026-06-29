"""Learning-rate schedule helpers shared across trainers."""

from __future__ import annotations

import math
from collections.abc import Callable


def cosine_with_floor(t_max: int, floor: float) -> Callable[[int], float]:
    """Per-epoch cosine-decay multiplier that bottoms out at ``floor`` x peak (not 0).

    Returns a factor for :class:`torch.optim.lr_scheduler.LambdaLR` so the single
    cosine factor scales each param group's own ``initial_lr`` -- flooring every
    group at ``floor`` of ITS base LR, which a scalar ``CosineAnnealingLR``
    ``eta_min`` (one absolute floor shared by all groups) cannot do. A 0.05
    minimum-LR ratio trains ViTs best (Southworth et al., arXiv:2605.24770).
    """
    span = max(t_max, 1)

    def factor(epoch: int) -> float:
        t = min(epoch, span) / span
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    return factor
