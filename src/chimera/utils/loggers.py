import time
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger


class ProgressPrinter(pl.Callback):
    """Flush human-readable progress to stdout: throughput + metrics every
    ``print_every`` steps, and wall-clock per stage (train / val / test).

    Complements the wandb/CSV loggers, which aren't visible while a backgrounded
    run's stdout is block-buffered. Every print uses ``flush=True`` so the lines
    appear in the log file immediately. Metrics are read from
    ``trainer.callback_metrics`` (whatever the module logged: loss/bpt/bpb).
    """

    def __init__(self, print_every: int = 500, tokens_per_step: int = 0):
        super().__init__()
        self.print_every = int(print_every)
        self.tokens_per_step = int(tokens_per_step)

    @staticmethod
    def _fmt_dt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def _metrics(self, trainer, prefix: str) -> str:
        out = []
        for name in ("loss", "bpt", "bpb"):
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
        self._last_t = self._t0
        self._last_step = trainer.global_step
        total = self._total_steps(trainer)
        print(f"[progress] train start: target {total or '?'} steps, "
              f"{self.tokens_per_step} tokens/step", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = trainer.global_step
        if self.print_every <= 0 or step == 0 or step % self.print_every != 0:
            return
        if step == self._last_step:  # same optimizer step (grad accum) -> skip dup
            return
        now = time.monotonic()
        dstep = step - self._last_step
        dt = max(now - self._last_t, 1e-9)
        sps = dstep / dt
        tps = sps * self.tokens_per_step
        total = self._total_steps(trainer)
        eta = self._fmt_dt((total - step) / sps) if (total and sps > 0) else "?"
        tok = step * self.tokens_per_step
        print(f"[progress] step {step}/{total or '?'}  {tok/1e6:.0f}M tok  "
              f"{self._metrics(trainer, 'train')}  |  {sps:.2f} step/s  "
              f"{tps/1e3:.0f}k tok/s  elapsed {self._fmt_dt(now - self._t0)}  eta {eta}",
              flush=True)
        self._last_t, self._last_step = now, step

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(self, "_epoch_t"):
            print(f"[progress] train epoch done in {self._fmt_dt(time.monotonic() - self._epoch_t)}",
                  flush=True)

    def on_validation_start(self, trainer, pl_module):
        self._val_t = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not hasattr(self, "_val_t"):
            return
        m = self._metrics(trainer, "val")
        print(f"[progress] val done in {self._fmt_dt(time.monotonic() - self._val_t)}"
              + (f"  |  {m}" if m else ""), flush=True)

    def on_test_start(self, trainer, pl_module):
        self._test_t = time.monotonic()
        print("[progress] test/eval start", flush=True)

    def on_test_end(self, trainer, pl_module):
        if hasattr(self, "_test_t"):
            print(f"[progress] test/eval done in {self._fmt_dt(time.monotonic() - self._test_t)}",
                  flush=True)


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
