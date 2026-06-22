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
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
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
    raise FileNotFoundError(
        f"No checkpoint found locally or as an artifact for run {run_id}"
    )


def add_common_args(
    parser: argparse.ArgumentParser, *, project: str, epochs: int
) -> None:
    """Add the argument block shared by every training script. ``project`` and ``epochs``
    set the per-script defaults; the rest are common across projects."""
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=7)
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="max grad norm (0 disables)"
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay for eval-time weights (0 disables EMA)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        default="/mnt/ai/data",
        help="where the dataset is downloaded/cached",
    )
    parser.add_argument("--project", default=project)
    parser.add_argument(
        "--resume", metavar="RUN_ID", default=None, help="wandb run id to resume"
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["off", "default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for the model ('off' disables)",
    )


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
    # Use TF32 for fp32 matmuls (the ones bf16-mixed autocast doesn't already cover) so the
    # GPU's Tensor Cores are utilized -- silences Lightning's warning at negligible precision cost.
    torch.set_float32_matmul_precision("high")
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


class EMA(Callback):
    """Maintains an exponential moving average of the LightningModule's trainable ``model``
    weights and evaluates with the averaged weights.

    Training proceeds on the live (raw) weights; after every optimizer step the shadow is
    nudged ``ema = decay * ema + (1 - decay) * live``. For validation and test the averaged
    weights are swapped into ``model`` and the raw weights restored afterward, so logged
    metrics and reconstruction images reflect the EMA model (which is typically smoother and
    scores better than the raw SGD iterate) while training continues unperturbed.

    The shadow is saved in the checkpoint as Lightning callback state (see ``state_dict`` /
    ``load_state_dict``), so a resumed run *continues* the average rather than restarting it.
    The model's own ``state_dict`` keeps the raw training weights, so resume and the artifact
    rebuild stay byte-for-byte what they were before EMA — only eval-time weights change.
    Non-float entries (e.g. ``num_batches_tracked``) are copied, not averaged."""

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema: dict | None = (
            None  # shadow of model.state_dict(); None until fit start
        )
        self._backup: dict | None = None  # raw weights stashed during an eval swap

    @staticmethod
    def _model(pl_module):
        """The trainable core that EMA tracks — the same ``.model`` that ``compile_model``
        targets — falling back to the module itself if it has no ``.model``."""
        return getattr(pl_module, "model", pl_module)

    def on_fit_start(self, trainer, pl_module) -> None:
        model = self._model(pl_module)
        if self.ema is None:
            self.ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:  # resumed: align the restored shadow with the (possibly GPU) model's devices
            reference = model.state_dict()
            self.ema = {k: v.to(reference[k].device) for k, v in self.ema.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args) -> None:
        for key, value in self._model(pl_module).state_dict().items():
            shadow = self.ema[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(
                    value
                )  # integer buffers (e.g. BN counts): track, don't average

    def _swap_in(self, pl_module) -> None:
        if self.ema is None:
            return
        model = self._model(pl_module)
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema)

    def _swap_out(self, pl_module) -> None:
        if self._backup is not None:
            self._model(pl_module).load_state_dict(self._backup)
            self._backup = None

    def on_validation_start(self, trainer, pl_module) -> None:
        self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._swap_out(pl_module)

    def on_test_start(self, trainer, pl_module) -> None:
        self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module) -> None:
        self._swap_out(pl_module)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "ema": self.ema}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.ema = state_dict["ema"]


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


# torch.compile modes that use CUDA graphs (static input shapes); an uneven last batch on
# any split forces an extra graph capture, so we drop it on all loaders (see run_training).
CUDA_GRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})


def compile_model(module, mode: str = "reduce-overhead") -> None:
    """Compile the LightningModule's core ``model`` in place with ``torch.compile`` (no-op
    when ``mode`` is ``"off"`` or the module has no ``model`` attribute).

    Uses ``nn.Module.compile`` (in place) rather than ``torch.compile(model)``: the latter
    wraps the model in an ``OptimizedModule`` that prefixes every ``state_dict`` key with
    ``_orig_mod.``, which would corrupt checkpoints and break the prefix-based weight loading
    in the rectified-flow project. In-place compile leaves keys and parameters untouched, so
    checkpoints, resume, and artifact rebuilds keep working unchanged. Only the trainable
    core ``model`` is compiled; loss/metric networks and logging stay eager."""
    if not mode or mode == "off":
        return
    model = getattr(module, "model", None)
    if model is None:
        print(f"[compile] skipped: {type(module).__name__} has no .model attribute")
        return
    model.compile(mode=mode)
    print(f"[compile] torch.compile(mode={mode!r}) on {type(model).__name__}")


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

    A ``KeyboardInterrupt`` during ``fit`` falls through to the test phase (when ``test`` is
    set) so an interrupted run is still evaluated on its current weights; a second interrupt
    during ``test`` just quits. The final-checkpoint upload and ``wandb.finish()`` run in a
    ``finally`` so they happen on a clean finish, either interrupt, *and* any other exception
    (which still propagates after the latest per-epoch ``last.ckpt`` is preserved)."""
    ckpt_dir = outputs / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), save_last=True, save_top_k=0, every_n_epochs=1
    )
    # Log the optimizer's learning rate to wandb as training progresses (every step, so the
    # scheduler's curve is visible); needs a logger, which every script passes in.
    lr_cb = LearningRateMonitor(logging_interval="step")
    callbacks = [ckpt_cb, lr_cb]
    # Maintain EMA weights and evaluate with them (0 disables). Append last so the eval-time
    # weight swap wraps the validation/test the other callbacks observe.
    ema_decay = getattr(args, "ema_decay", 0.0)
    if ema_decay:
        callbacks.append(EMA(decay=ema_decay))
        print(f"[ema] tracking eval weights with decay={ema_decay}")
    mode = getattr(args, "compile_mode", "reduce-overhead")
    compile_model(module, mode)
    if mode in CUDA_GRAPH_COMPILE_MODES and hasattr(datamodule, "drop_last"):
        datamodule.drop_last = (
            True  # avoid an extra CUDA-graph capture for the uneven last batch
        )
        print(
            f"[compile] {mode!r} uses CUDA graphs -> dropping the last (uneven) batch on every split"
        )
    trainer = build_trainer(args, logger, callbacks)
    logger.watch(module, log="gradients")
    try:
        try:
            trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
        except KeyboardInterrupt:
            # Interrupting training falls through to the test phase (below) on the
            # current weights rather than quitting outright.
            print(
                "training interrupted — running test on the current weights"
                if test
                else "training interrupted — exiting"
            )
        if test:
            try:
                trainer.test(module, datamodule=datamodule)
            except KeyboardInterrupt:
                # Interrupting the test just quits; there's nothing left to run.
                print("test interrupted — exiting")
    finally:
        upload_checkpoint_artifact(
            logger, run_id, ckpt_dir / "last.ckpt", artifact_metadata
        )
        wandb.finish()
