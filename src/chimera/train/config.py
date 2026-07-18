"""Shared training config — the base every project's ``Config`` subclasses.

Projects extend it with task-specific fields and override the per-project
required knobs (``run_dir``, ``wandb_project``)::

    @dataclass
    class Config(TrainConfig):
        run_dir: Path = Path("/mnt/ai/runs/mnist/classifier")
        wandb_project: str = "mnist-classifier"
        arch: str = "small"

    cfg = tyro.cli(Config)

``tyro.cli`` turns the dataclass into the full CLI (``--run-dir``, ``--arch``,
...), with types, defaults, and ``--help`` derived from the fields and their
docstrings/comments.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # -- paths (per-project subclasses override the defaults) ---------------
    run_dir: Path
    """Checkpoints + local logs land here (convention: /mnt/ai/runs/<project>/<task>)."""
    wandb_project: str
    """wandb project name (convention: <project>-<task>)."""
    data_dir: Path = Path("/mnt/ai/data")
    """Root for datasets/caches (HF_HOME lives under here too)."""

    # -- schedule ------------------------------------------------------------
    epochs: int = 1
    max_steps: int = -1
    """Optimizer-step cap; -1 = no cap (run the full --epochs)."""
    batch_size: int = 128
    lr: float = 1e-3
    warmup_steps: int = 100
    seed: int = 42

    # -- precision / speed ---------------------------------------------------
    precision: str = "bf16-mixed"
    compile: bool = False
    """Consumed by the project's model/module wiring (run() does not compile)."""
    deterministic: bool = False
    """Trainer(deterministic=...); vision projects historically ran with True."""

    # -- logging ---------------------------------------------------------------
    run_name: str | None = None
    """wandb run name (None = wandb picks one)."""
    tags: tuple[str, ...] = ()
    wandb_offline: bool = False

    # -- opt-in extras ---------------------------------------------------------
    ema_decay: float | None = None
    """Enable the EMA callback with this decay ceiling (None = no EMA)."""
