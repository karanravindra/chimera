"""Learning-rate schedulers shared across the training stages."""

import math

from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupCosineAnnealingLR(LambdaLR):
    """Per-step linear warmup then cosine decay to ``final_lr_frac * base_lr``.

    Reproduces the hand-rolled tinylm ``lr_factor`` exactly: a linear ramp for
    ``warmup_steps`` optimizer steps (factor ``(step + 1) / warmup_steps``), then a
    half-cosine from ``1.0`` down to ``final_lr_frac`` over the remaining
    ``max_steps - warmup_steps`` steps. The same factor multiplies every param
    group's base lr, so Muon's per-group lrs (hidden vs embedding/head) keep their
    ratio — matching the old ``base_lrs`` capture-and-scale loop.

    Step it with ``interval="step"`` (once per optimizer step) so ``last_epoch``
    tracks the optimizer-step index: during optimizer step ``n`` (0-indexed) the
    applied factor is ``_factor(n)``.
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps: int,
        max_steps: int,
        final_lr_frac: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_steps = int(max_steps)
        self.final_lr_frac = float(final_lr_frac)
        super().__init__(optimizer, self._factor, last_epoch=last_epoch)

    def _factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        t = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return self.final_lr_frac + (1 - self.final_lr_frac) * 0.5 * (
            1 + math.cos(math.pi * t)
        )
