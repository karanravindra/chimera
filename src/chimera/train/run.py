"""The shared run harness: everything a training run *always* needs.

``run()`` owns seeding, checkpointing, loggers, the standard callbacks, Trainer
construction, fit, and the optional test-from-best-checkpoint. Task policy —
optimizer/scheduler construction, muP grouping, compile — stays in the project's
module/wiring; anything task-specific rides in via ``callbacks=`` or
``trainer_kwargs=``. If a third project passes the same thing through, it moves
into this layer.
"""

from dataclasses import dataclass, field
from pathlib import Path

from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from chimera.utils import EMACallback, ProgressPrinter, TokenAxisCallback, build_run_loggers

from .config import TrainConfig


@dataclass
class RunResult:
    best_ckpt: Path | None
    """Best checkpoint per the monitored metric (None if no val ran / nothing saved)."""
    wandb_id: str | None
    """wandb run id — lets bench/eval scripts backfill metrics without retraining."""
    metrics: dict[str, float] = field(default_factory=dict)
    """Final trainer.callback_metrics (post-test when test=True)."""


def run(
    cfg: TrainConfig,
    module: LightningModule,
    dm: LightningDataModule,
    *,
    monitor: str = "val/loss",
    mode: str = "min",
    ckpt_name: str = "best",
    callbacks: list[Callback] | tuple[Callback, ...] = (),
    test: bool = True,
    tokens_per_step: int = 0,
    trainer_kwargs: dict | None = None,
) -> RunResult:
    """Fit (and optionally test) ``module`` on ``dm`` under the standard harness.

    Args:
        monitor/mode: checkpoint selection metric (e.g. ``val/acc`` + ``max``).
        ckpt_name: checkpoint filename under ``cfg.run_dir/checkpoints``.
        callbacks: extra task-specific callbacks, appended after the standard ones.
        test: run ``trainer.test`` from the best checkpoint after fit.
        tokens_per_step: for LM runs — wires ProgressPrinter throughput and the
            TokenAxisCallback (tokens as the wandb x-axis). 0 = step-based only.
        trainer_kwargs: passthrough to ``Trainer`` (e.g. ``val_check_interval``,
            ``gradient_clip_val``); wins over the cfg-derived defaults.
    """
    seed_everything(cfg.seed, workers=True)

    run_dir = Path(cfg.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename=ckpt_name,
        monitor=monitor,
        mode=mode,
        enable_version_counter=False,
    )
    loggers = build_run_loggers(
        run_dir, cfg.wandb_project, cfg.run_name, cfg.wandb_offline, tags=list(cfg.tags)
    )

    all_callbacks: list[Callback] = [checkpoint, ProgressPrinter(tokens_per_step=tokens_per_step)]
    if tokens_per_step > 0:
        all_callbacks.append(TokenAxisCallback(tokens_per_step))
    if cfg.ema_decay is not None:
        all_callbacks.append(EMACallback(decay=cfg.ema_decay, warmup_steps=cfg.warmup_steps))
    all_callbacks.extend(callbacks)

    kwargs = dict(
        max_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        precision=cfg.precision,
        deterministic=cfg.deterministic,
        logger=loggers,
        callbacks=all_callbacks,
    )
    kwargs.update(trainer_kwargs or {})
    trainer = Trainer(**kwargs)

    trainer.fit(module, datamodule=dm)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}

    best = checkpoint.best_model_path or None
    if test:
        # trainer.test resets callback_metrics, so merge the two stages
        trainer.test(module, datamodule=dm, ckpt_path=best)
        metrics.update({k: float(v) for k, v in trainer.callback_metrics.items()})

    wandb_id = next(
        (lg.version for lg in trainer.loggers if isinstance(lg, WandbLogger)), None
    )
    if best:
        print("best checkpoint:", best)
    return RunResult(best_ckpt=Path(best) if best else None, wandb_id=wandb_id, metrics=metrics)
