"""Shared glue for the ``projects/<dataset>/<objective>/train.py`` scripts.

Each training script owns its model, ``LightningModule``, and step logic; everything
around that — the common argparse block, the WandbLogger resume dance, locating a run's
checkpoint (local copy or wandb artifact), the fit lifecycle (per-run checkpoint dir +
callback, Trainer, and the interrupt/crash-safe artifact upload in ``run_training``), and
tiling images into a log grid — is identical across projects and lives here.

Import the helpers directly (``from chimera.utils.experiment import ...``); they are
intentionally *not* re-exported from ``chimera.utils`` so a bare ``import chimera.utils``
stays free of the wandb / lightning import cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.utils import make_grid


def grid(images: torch.Tensor, nrow: int | None = None):
    """Tile a batch of [0,1] images into one grid, returned as an HxW (grayscale) or
    HxWx3 numpy array (channel layout chosen by the image's channel count)."""
    g = make_grid(images, nrow=nrow or images.shape[0], padding=2)
    g = g[0] if g.shape[0] == 1 else g.permute(1, 2, 0)
    return g.numpy()


def find_ckpt(run_id: str, project: str, outputs: Path) -> str:
    """Locate a run's checkpoint: prefer the local copy under ``outputs/<run_id>``, else
    download the latest wandb model artifact for the run."""
    local = outputs / run_id / "last.ckpt"
    if local.exists():
        print(f"[ckpt] using local checkpoint {local}")
        return str(local)
    print(f"[ckpt] no local checkpoint; downloading model artifact for run {run_id}")
    api = wandb.Api()
    run = api.run(f"{api.default_entity}/{project}/{run_id}")
    for artifact in reversed(list(run.logged_artifacts())):
        if artifact.type != "model":
            continue
        directory = Path(artifact.download())
        ckpts = sorted(directory.glob("*.ckpt"))
        if ckpts:
            print(f"[ckpt] downloaded {ckpts[0]}")
            return str(ckpts[0])
    raise FileNotFoundError(f"No checkpoint found locally or as an artifact for run {run_id}")


def add_common_args(parser: argparse.ArgumentParser, *, project: str, epochs: int) -> None:
    """Add the argument block shared by every training script. ``project`` and ``epochs``
    set the per-script defaults; the rest are common across projects."""
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (0 disables)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="/mnt/ai/data", help="where the dataset is downloaded/cached")
    parser.add_argument("--project", default=project)
    parser.add_argument("--resume", metavar="RUN_ID", default=None, help="wandb run id to resume")


def init_wandb_logger(
    project: str, config: dict, resume: str | None = None
) -> tuple[WandbLogger, str]:
    """Create the WandbLogger (continuing run ``resume`` if given) and return it with the
    run id. Accessing ``logger.experiment`` here initializes the run so the id is known
    before the checkpoint directory is created."""
    logger_kwargs = dict(project=project, config=config)
    if resume:
        logger_kwargs.update(id=resume, resume="must")
    logger = WandbLogger(**logger_kwargs)
    run_id = logger.experiment.id
    print(f"wandb run id: {run_id}{' (resumed)' if resume else ''}")
    return logger, run_id


def build_trainer(args, logger, callbacks) -> Trainer:
    """The Trainer configuration shared by every training script."""
    return Trainer(
        max_epochs=args.epochs,
        precision="bf16-mixed",
        accelerator="auto",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.grad_clip or None,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )


def upload_checkpoint_artifact(
    logger: WandbLogger, run_id: str, ckpt_path: Path, metadata: dict
) -> None:
    """Upload the final checkpoint as a wandb model artifact (aliased ``latest``) so any
    run can be rebuilt later. No-op if the checkpoint doesn't exist."""
    if not ckpt_path.exists():
        return
    artifact = wandb.Artifact(name=run_id, type="model", metadata=metadata)
    artifact.add_file(str(ckpt_path))
    logger.experiment.log_artifact(artifact, aliases=["latest"])
    print(f"logged checkpoint artifact for run {run_id}")


def run_training(
    *,
    module,
    datamodule,
    args,
    logger: WandbLogger,
    run_id: str,
    outputs: Path,
    resume_ckpt: str | None,
    artifact_metadata: dict,
    test: bool = False,
) -> None:
    """The training lifecycle shared by every script: set up the per-run checkpoint dir
    and callback, build the Trainer, and fit (resuming from ``resume_ckpt`` if given). When
    ``test`` is set, ``trainer.test`` runs on the just-trained weights after a clean fit.

    The final-checkpoint upload and ``wandb.finish()`` run in a ``finally`` so they happen
    on a clean finish, a ``KeyboardInterrupt`` (swallowed), *and* any other exception (which
    still propagates after the latest per-epoch ``last.ckpt`` is preserved as an artifact)."""
    ckpt_dir = outputs / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), save_last=True, save_top_k=0, every_n_epochs=1)
    trainer = build_trainer(args, logger, [ckpt_cb])
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
        if test:
            trainer.test(module, datamodule=datamodule)
    except KeyboardInterrupt:
        print("interrupted — uploading the latest checkpoint before exiting")
    finally:
        upload_checkpoint_artifact(logger, run_id, ckpt_dir / "last.ckpt", artifact_metadata)
        wandb.finish()
