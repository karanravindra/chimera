"""Progress printing, a token x-axis callback, and the standard logger set.

Ported from the archived Lightning harness. ``build_run_loggers`` always wires a
``CSVLogger`` (so ``metrics.csv`` is written locally with no extra dependency) and
only adds a ``WandbLogger`` when ``wandb`` is installed — offline by default, so
the project's "CSV/offline first" tracking works without a hard ``wandb`` dep or a
login. Set ``wandb_offline=False`` (with ``wandb`` installed + logged in) to sync.
"""

import time
from collections import deque
from pathlib import Path
from statistics import median

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, Logger


class ProgressPrinter(pl.Callback):
    """Flush human-readable progress to stdout: throughput + metrics every
    ``print_every`` steps, and wall-clock per stage (train / val / test).

    Complements the wandb/CSV loggers, which aren't visible while a backgrounded
    run's stdout is block-buffered. Every print uses ``flush=True`` so the lines
    appear in the log file immediately. Metrics are read from
    ``trainer.callback_metrics`` (whatever the module logged: loss/bpt/bpb).

    Rate + ETA come from a rolling median of per-optimizer-step *training*
    durations (not the wall time since the last print): the interval straddling a
    validation pass is discarded and the compile-warmup steps age out of the
    window, so the ETA doesn't thrash (a naive "tokens since last print / wall
    since last print" counts validation + compile as training and swings wildly).
    ETA is training-only — it doesn't budget for remaining validation passes.
    """

    def __init__(
        self, print_every: int = 500, tokens_per_step: int = 0, rate_window: int = 50
    ):
        super().__init__()
        self.print_every = int(print_every)
        self.tokens_per_step = int(tokens_per_step)
        # window of recent per-step durations (secs) for a stable rate estimate
        self.rate_window = int(rate_window)

    @staticmethod
    def _fmt_dt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def _metrics(self, trainer, prefix: str) -> str:
        out = []
        for name in ("loss", "bpt", "bpb", "ppl"):
            v = trainer.callback_metrics.get(f"{prefix}/{name}")
            if v is not None:
                out.append(f"{name} {float(v):.4f}")
        return "  ".join(out)

    def _total_steps(self, trainer) -> int:
        if trainer.max_steps and trainer.max_steps > 0:
            return trainer.max_steps
        try:
            return int(trainer.estimated_stepping_batches)
        except Exception:
            return 0

    def on_train_start(self, trainer, pl_module):
        self._t0 = time.monotonic()
        self._prev_batch_t = self._t0
        self._last_step = trainer.global_step
        self._recent = deque(maxlen=self.rate_window)
        # discard the first interval (compile warmup / setup) — not a real step rate
        self._skip_dur = True
        total = self._total_steps(trainer)
        print(
            f"[progress] train start: target {total or '?'} steps, "
            f"{self.tokens_per_step} tokens/step",
            flush=True,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t = time.monotonic()

    def _sps(self) -> float:
        """Steady-state steps/sec from the rolling median per-step duration."""
        return (1.0 / median(self._recent)) if self._recent else 0.0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = trainer.global_step
        now = time.monotonic()
        # Record one duration per optimizer step (global_step advances). Skip the
        # interval flagged after start/validation (it includes compile/val time,
        # not steady-state training).
        if step != self._last_step:
            if self._skip_dur:
                self._skip_dur = False
            else:
                self._recent.append(
                    (now - self._prev_batch_t) / (step - self._last_step)
                )
            self._prev_batch_t = now
            self._last_step = step

        if self.print_every <= 0 or step == 0 or step % self.print_every != 0:
            return
        sps = self._sps()
        tps = sps * self.tokens_per_step
        total = self._total_steps(trainer)
        eta = self._fmt_dt((total - step) / sps) if (total and sps > 0) else "?"
        tok = step * self.tokens_per_step
        print(
            f"[progress] step {step}/{total or '?'}  {tok / 1e6:.0f}M tok  "
            f"{self._metrics(trainer, 'train')}  |  {sps:.2f} step/s  "
            f"{tps / 1e3:.0f}k tok/s  elapsed {self._fmt_dt(now - self._t0)}  eta {eta}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(self, "_epoch_t"):
            print(
                f"[progress] train epoch done in {self._fmt_dt(time.monotonic() - self._epoch_t)}",
                flush=True,
            )

    def on_validation_start(self, trainer, pl_module):
        self._val_t = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not hasattr(self, "_val_t"):
            return
        m = self._metrics(trainer, "val")
        print(
            f"[progress] val done in {self._fmt_dt(time.monotonic() - self._val_t)}"
            + (f"  |  {m}" if m else ""),
            flush=True,
        )
        # The next train interval straddles this val pass — discard it from the
        # step-rate estimate so ETA doesn't spike (mirrors the compile-warmup skip).
        if hasattr(self, "_recent"):
            self._skip_dur = True
            self._prev_batch_t = time.monotonic()

    def on_test_start(self, trainer, pl_module):
        self._test_t = time.monotonic()
        print("[progress] test/eval start", flush=True)

    def on_test_end(self, trainer, pl_module):
        if hasattr(self, "_test_t"):
            print(
                f"[progress] test/eval done in {self._fmt_dt(time.monotonic() - self._test_t)}",
                flush=True,
            )


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
    wandb_offline: bool = True,
    tags: list[str] | None = None,
) -> list[Logger]:
    """Standard logger set for a training script.

    Always includes ``CSVLogger`` first (so ``trainer.logger.log_dir`` stays the
    local metrics dir for scripts/notebooks reading ``metrics.csv``). A
    ``WandbLogger`` is appended only when ``wandb`` is importable — offline by
    default, so tracking works with no hard dependency and no login. Pass
    ``wandb_offline=False`` (wandb installed + logged in) to sync online.
    """
    loggers: list[Logger] = [CSVLogger(save_dir=run_dir, name="csv")]
    try:
        from lightning.pytorch.loggers import WandbLogger  # noqa: PLC0415

        import wandb  # noqa: F401,PLC0415  — probe: absent -> CSV-only
    except ImportError:
        print("[loggers] wandb not installed -> CSV-only tracking", flush=True)
        return loggers
    loggers.append(
        WandbLogger(
            project=wandb_project,
            name=run_name,
            save_dir=run_dir,
            offline=wandb_offline,
            tags=tags,
        )
    )
    return loggers
