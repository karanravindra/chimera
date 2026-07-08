from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim import Optimizer


def LinearWarmupCosineAnnealingLR(
    optimizer: Optimizer,
    warmup_steps: int,
    n_epochs: int,
    train_loader_length: int,
    eta_min: float = 1e-5,
    max_steps: int | None = None,
) -> SequentialLR:
    # The cosine anneals over the actual training horizon. By default that is the
    # full run (n_epochs * train_loader_length), but if the run is capped early by
    # max_steps the cosine must anneal over THAT many steps instead — otherwise a
    # capped run stops with the LR still near its peak (under-annealed), since the
    # cosine was scaled to a horizon it never reaches.
    total_steps = n_epochs * train_loader_length
    if max_steps is not None and max_steps > 0:
        total_steps = min(total_steps, max_steps)

    linear_schedule = LinearLR(optimizer, total_iters=warmup_steps)
    cosine_schedule = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[linear_schedule, cosine_schedule],
        milestones=[warmup_steps],
    )
    return scheduler
