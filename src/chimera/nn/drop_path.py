"""Stochastic depth (DropPath): randomly drop whole residual branches per sample.

Standard ViT regularizer (Huang et al. 2016, "Deep Networks with Stochastic Depth"; the timm
``DropPath``). Applied to the *output of a residual branch* before it is added back, so a dropped
sample's block reduces to the identity that step. Surviving branches are rescaled by ``1/keep`` so
the expected activation is unchanged, which makes it a no-op at inference (and when ``drop_prob`` is
0). Networks usually ramp the rate linearly with depth -- callers build that schedule and pass each
block its own ``drop_prob``.
"""

from __future__ import annotations

import torch
from torch import nn


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    # One Bernoulli(keep) gate per sample, broadcast over all non-batch dims.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


class DropPath(nn.Module):
    """Per-sample stochastic depth on a residual branch.

    Identity in eval or when ``drop_prob == 0``; in training it zeroes a random ``drop_prob``
    fraction of the batch's branch outputs and rescales the survivors by ``1/(1 - drop_prob)``.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:g}"
