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
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
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
            # Prefer the exact-resume ``last.ckpt`` if the artifact carries it (it may also
            # hold the monitored best, named like ``epoch=NN-step=MMMM.ckpt``, so resuming
            # from the first ckpt could rewind to an older best epoch); fall back to the
            # first ckpt for older single-file artifacts.
            chosen = next((c for c in ckpts if c.name == "last.ckpt"), ckpts[0])
            print(f"[ckpt] downloaded {chosen}")
            return str(chosen)
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
    nudged ``ema = decay * ema + (1 - decay) * live``. The shadow starts at **zero** (not the
    random init), and eval-time weights are **bias-corrected** Adam-style by ``ema / (1 -
    decay**t)`` where ``t`` is the number of updates so far. Without this correction a high
    decay (e.g. 0.999, ~1-epoch half-life) leaves the shadow dominated by the zero start for
    many epochs, so validation lags training badly early on and only catches up once
    ``decay**t`` has decayed away; the correction makes the average unbiased from the very
    first step (at ``t = 1`` the corrected shadow equals the live weights). For validation and
    test the corrected weights are swapped into ``model`` and the raw weights restored
    afterward, so logged metrics and reconstruction images reflect the EMA model (typically
    smoother and better-scoring than the raw SGD iterate) while training continues unperturbed.

    The shadow and update count are saved in the checkpoint as Lightning callback state (see
    ``state_dict`` / ``load_state_dict``), so a resumed run *continues* the average rather than
    restarting it. The model's own ``state_dict`` keeps the raw training weights, so resume and
    the artifact rebuild stay byte-for-byte what they were before EMA — only eval-time weights
    change. Non-float entries (e.g. ``num_batches_tracked``) are tracked (copied), not averaged
    or corrected. Before the first optimizer step (``t = 0``, e.g. sanity-check validation) the
    swap is skipped, since no average exists yet."""

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema: dict | None = (
            None  # zero-init shadow of model.state_dict(); None until fit start
        )
        self._steps = 0  # optimizer steps folded into the shadow (the ``t`` in 1 - decay**t)
        self._backup: dict | None = None  # raw weights stashed during an eval swap

    @staticmethod
    def _model(pl_module):
        """The trainable core that EMA tracks — the same ``.model`` that ``compile_model``
        targets — falling back to the module itself if it has no ``.model``."""
        return getattr(pl_module, "model", pl_module)

    def on_fit_start(self, trainer, pl_module) -> None:
        model = self._model(pl_module)
        if self.ema is None:
            # Zero-init float accumulators so ``ema / (1 - decay**t)`` is exactly unbiased;
            # non-float buffers are tracked by copy, so seed them with the live value.
            self.ema = {
                k: (torch.zeros_like(v) if v.dtype.is_floating_point else v.detach().clone())
                for k, v in model.state_dict().items()
            }
        else:  # resumed: align the restored shadow with the (possibly GPU) model's devices
            reference = model.state_dict()
            self.ema = {k: v.to(reference[k].device) for k, v in self.ema.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args) -> None:
        self._steps += 1
        for key, value in self._model(pl_module).state_dict().items():
            shadow = self.ema[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(
                    value
                )  # integer buffers (e.g. BN counts): track, don't average

    def _corrected(self) -> dict:
        """Bias-corrected eval weights: float entries divided by ``1 - decay**t`` to undo the
        zero-init start; non-float entries (tracked by copy) passed through unchanged."""
        bias = 1.0 - self.decay**self._steps
        return {
            k: (v / bias if v.dtype.is_floating_point else v) for k, v in self.ema.items()
        }

    def _swap_in(self, pl_module) -> None:
        if self.ema is None or self._steps == 0:
            return  # no average yet (e.g. sanity-check val before the first step)
        model = self._model(pl_module)
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self._corrected())

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
        return {"decay": self.decay, "ema": self.ema, "steps": self._steps}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        if "steps" in state_dict:
            self.ema = state_dict["ema"]
            self._steps = state_dict["steps"]
        else:  # pre-bias-correction checkpoint stored real averaged weights, not a zero-init
            self.ema = None  # accumulator semantics changed; restart the average cleanly
            self._steps = 0


def upload_checkpoint_artifact(
    logger: WandbLogger,
    run_id: str,
    ckpt_path: Path,
    metadata: dict,
    best_path: Path | None = None,
) -> None:
    """Upload the final checkpoint(s) as a wandb model artifact (aliased ``latest``) so any
    run can be rebuilt or resumed later. ``ckpt_path`` is the primary (exact-resume)
    checkpoint -- usually ``last.ckpt``; when ``best_path`` is given and exists, the monitored
    best checkpoint is added to the same artifact alongside it (Lightning names it like
    ``epoch=NN-step=MMMM.ckpt``, so it won't collide with ``last.ckpt``). No-op if the
    primary checkpoint doesn't exist."""
    if not ckpt_path.exists():
        return
    artifact = wandb.Artifact(name=run_id, type="model", metadata=metadata)
    artifact.add_file(str(ckpt_path))
    if best_path is not None and best_path.exists() and best_path != ckpt_path:
        artifact.add_file(str(best_path))
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
    monitor: str | None = None,
    monitor_mode: str = "min",
    early_stop_patience: int = 0,
) -> None:
    """The training lifecycle shared by every script: set up the per-run checkpoint dir
    and callback, build the Trainer, and fit (resuming from ``resume_ckpt`` if given). When
    ``test`` is set, ``trainer.test`` runs after a clean fit.

    ``monitor`` opts into metric-based checkpoint selection: when given, the best-scoring epoch
    (per ``monitor`` / ``monitor_mode``) is kept *and* is what gets tested and uploaded, instead
    of the last epoch -- and ``early_stop_patience`` (> 0) stops training once ``monitor`` has not
    improved for that many epochs. Without ``monitor`` the behavior is unchanged: only ``last.ckpt``
    is kept and the final weights are tested. ``save_last`` stays on either way so resume always
    has a ``last.ckpt``.

    A ``KeyboardInterrupt`` during ``fit`` falls through to the test phase (when ``test`` is
    set) so an interrupted run is still evaluated on its current weights; a second interrupt
    during ``test`` just quits. The final-checkpoint upload and ``wandb.finish()`` run in a
    ``finally`` so they happen on a clean finish, either interrupt, *and* any other exception
    (which still propagates after the latest per-epoch ``last.ckpt`` is preserved)."""
    ckpt_dir = outputs / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        # With a monitor, keep the single best-scoring epoch (save_top_k=1) so the overfit
        # endpoint isn't all we retain; without one, keep the prior periodic-last-only behavior.
        save_top_k=1 if monitor else 0,
        monitor=monitor,
        mode=monitor_mode,
        every_n_epochs=1,
    )
    # Log the optimizer's learning rate to wandb as training progresses (every step, so the
    # scheduler's curve is visible); needs a logger, which every script passes in.
    lr_cb = LearningRateMonitor(logging_interval="step")
    callbacks = [ckpt_cb, lr_cb]
    if monitor and early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor, mode=monitor_mode, patience=early_stop_patience
            )
        )
        print(
            f"[early-stop] monitor={monitor!r} mode={monitor_mode} patience={early_stop_patience}"
        )
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
                # With metric-based selection, test the BEST epoch (reloaded from its checkpoint)
                # rather than the final / early-stopping-point weights. Needs a saved best; fall
                # back to current weights if none exists (e.g. interrupted before any validation).
                best = ckpt_cb.best_model_path if monitor else ""
                trainer.test(
                    module, datamodule=datamodule, ckpt_path="best" if best else None
                )
            except KeyboardInterrupt:
                # Interrupting the test just quits; there's nothing left to run.
                print("test interrupted — exiting")
    finally:
        # Always upload ``last.ckpt`` as the primary file so a remote resume rewinds to the
        # true latest epoch (not the older best); when monitoring, also bundle the best
        # checkpoint into the same artifact so the kept model is available for rebuilds.
        best_path = ckpt_cb.best_model_path if monitor else ""
        upload_checkpoint_artifact(
            logger,
            run_id,
            ckpt_dir / "last.ckpt",
            artifact_metadata,
            best_path=Path(best_path) if best_path else None,
        )
        wandb.finish()
