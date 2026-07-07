from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger


def build_run_loggers(
    run_dir: str | Path,
    wandb_project: str,
    run_name: str | None = None,
    wandb_offline: bool = False,
    tags: list[str] | None = None,
) -> list[Logger]:
    """Standard pair of loggers for a training script.

    ``CSVLogger`` (first, so ``trainer.logger.log_dir`` stays the local metrics
    dir for scripts/notebooks that read ``metrics.csv``) plus ``WandbLogger``
    for dashboards, images, and tables. Both share ``run_dir``.
    """
    return [
        CSVLogger(save_dir=run_dir, name="csv"),
        WandbLogger(
            project=wandb_project,
            name=run_name,
            save_dir=run_dir,
            offline=wandb_offline,
            tags=tags,
        ),
    ]
