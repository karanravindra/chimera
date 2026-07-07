from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim import Optimizer


def LinearWarmupCosineAnnealingLR(
    optimizer: Optimizer,
    warmup_steps: int,
    n_epochs: int,
    train_loader_length: int,
) -> SequentialLR:
    linear_schedule = LinearLR(optimizer, total_iters=warmup_steps)
    cosine_schedule = CosineAnnealingLR(
        optimizer, T_max=n_epochs * train_loader_length - warmup_steps, eta_min=1e-5
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[linear_schedule, cosine_schedule],
        milestones=[warmup_steps],
    )
    return scheduler
