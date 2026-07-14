from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger


class TokenAxisCallback(pl.Callback):
    """Log cumulative tokens and chart every metric against them in wandb.

    Logs ``trainer/trained_tokens`` (= ``global_step * tokens_per_step``) on every
    train step and validation, and calls ``wandb.define_metric`` so all other
    metrics use ``trainer/trained_tokens`` as their x-axis instead of the raw
    optimizer step — the meaningful axis when comparing runs at different batch
    sizes / sequence lengths.
    """

    def __init__(self, tokens_per_step: int):
        super().__init__()
        self.tokens_per_step = int(tokens_per_step)

    @staticmethod
    def _wandb(trainer):
        for lg in trainer.loggers:
            exp = getattr(lg, "experiment", None)
            if exp is not None and type(exp).__module__.split(".")[0] == "wandb":
                return exp
        return None

    def on_fit_start(self, trainer, pl_module):
        w = self._wandb(trainer)
        if w is not None:
            w.define_metric("trainer/trained_tokens")
            w.define_metric("*", step_metric="trainer/trained_tokens")

    def _log_tokens(self, pl_module, trainer):
        tokens = trainer.global_step * self.tokens_per_step
        pl_module.log("trainer/trained_tokens", float(tokens))

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self._log_tokens(pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._log_tokens(pl_module, trainer)


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
